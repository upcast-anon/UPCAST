import copy
import math
import os
from typing import Any, Optional, Tuple
import torch
from torch import nn, Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from omegaconf import DictConfig
from einops import rearrange
from ..modules.embeddings import (
    RandomDropoutPatchEmbed,
)
from .u_vit3d import UViT3D
from upcast.state_compat import (
    PREFERENCE_POLICY_VERSION,
    PreferencePolicySpec,
    StableWorldStatePreferencePolicy,
)


def _scalar_sinusoidal_embedding(values: Tensor, dim: int) -> Tensor:
    """Embed one scalar per sample without introducing a learned lookup table."""
    values = values.float().reshape(-1, 1)
    half = dim // 2
    if half == 0:
        return values.new_zeros((values.shape[0], dim))
    frequencies = torch.exp(
        torch.arange(half, device=values.device, dtype=values.dtype)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    angles = values * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if embedding.shape[-1] < dim:
        embedding = F.pad(embedding, (0, dim - embedding.shape[-1]))
    return embedding


class PhysicalTransitionConditioner(nn.Module):
    """Inject a paired shared physical transition at its DFoT source layer.

    The video path is an exact identity at initialization.  A separate
    physical-velocity head makes the shared UPCAST code part of the denoising
    state instead of using it only as a reconstruction target.  The read path
    compares the video and fused UPCAST tokens explicitly, so a shuffled fused
    state can be rejected instead of looking like another generic residual.
    """

    VERSION = 4

    def __init__(
        self,
        hidden_dim: int,
        physical_dim: int = 64,
        num_heads: int = 8,
        spatial_grid: int = 8,
        max_read_scale: float = 0.05,
        local_read_fraction: float = 0.5,
        local_read_radius: int = 1,
        pair_encoder_layers: int = 2,
        pair_gate_prior: float = 0.8,
        pair_update_scale: float = 0.25,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            num_heads = math.gcd(hidden_dim, num_heads)
        if num_heads < 1:
            raise ValueError(
                f"No valid attention head count for hidden_dim={hidden_dim}"
            )
        self.hidden_dim = int(hidden_dim)
        self.physical_dim = int(physical_dim)
        self.spatial_grid = int(spatial_grid)
        self.max_read_scale = float(max_read_scale)
        self.local_read_fraction = float(local_read_fraction)
        self.local_read_radius = int(local_read_radius)
        self.pair_encoder_layers = int(pair_encoder_layers)
        self.pair_gate_prior = float(pair_gate_prior)
        self.pair_update_scale = float(pair_update_scale)
        if self.spatial_grid < 1:
            raise ValueError("generative_unit.spatial_grid must be positive")
        if self.max_read_scale <= 0:
            raise ValueError("generative_unit.max_read_scale must be positive")
        if not 0.0 <= self.local_read_fraction <= 1.0:
            raise ValueError(
                "generative_unit.local_read_fraction must be in [0, 1]"
            )
        if not 0 <= self.local_read_radius < self.spatial_grid:
            raise ValueError(
                "generative_unit.local_read_radius must be in "
                "[0, spatial_grid)"
            )
        if self.pair_encoder_layers < 1:
            raise ValueError(
                "generative_unit.pair_encoder_layers must be positive"
            )
        if not 0.0 < self.pair_gate_prior < 1.0:
            raise ValueError(
                "generative_unit.pair_gate_prior must be in (0, 1)"
            )
        if self.pair_update_scale < 0.0:
            raise ValueError(
                "generative_unit.pair_update_scale must be non-negative"
            )

        token_count = self.spatial_grid**2
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.physical_norm = nn.LayerNorm(self.physical_dim)
        self.physical_position = nn.Parameter(
            torch.zeros(1, token_count, self.physical_dim)
        )
        self.noise_embedding = nn.Sequential(
            nn.Linear(self.physical_dim, self.physical_dim * 2),
            nn.SiLU(),
            nn.Linear(self.physical_dim * 2, self.physical_dim),
        )
        pair_heads = math.gcd(self.physical_dim, int(num_heads))
        pair_heads = max(pair_heads, 1)
        self.pair_video_norm = nn.LayerNorm(self.physical_dim)
        self.pair_fused_norm = nn.LayerNorm(self.physical_dim)
        self.pair_input_projection = nn.Linear(
            self.physical_dim * 4,
            self.physical_dim,
        )
        pair_layer = nn.TransformerEncoderLayer(
            d_model=self.physical_dim,
            nhead=pair_heads,
            dim_feedforward=self.physical_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.pair_encoder = nn.TransformerEncoder(
            pair_layer,
            num_layers=self.pair_encoder_layers,
            norm=nn.LayerNorm(self.physical_dim),
            enable_nested_tensor=False,
        )
        self.pair_update_norm = nn.LayerNorm(self.physical_dim)
        self.pair_update = nn.Linear(self.physical_dim, self.physical_dim)
        self.pair_gate_norm = nn.LayerNorm(self.physical_dim)
        self.pair_gate = nn.Linear(self.physical_dim, 1)
        nn.init.zeros_(self.pair_update.weight)
        nn.init.zeros_(self.pair_update.bias)
        nn.init.zeros_(self.pair_gate.weight)
        nn.init.zeros_(self.pair_gate.bias)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=int(num_heads),
            kdim=self.physical_dim,
            vdim=self.physical_dim,
            batch_first=True,
        )
        # A scalar zero gate gives an exact no-op while still allowing the gate
        # itself to receive a first-step gradient.
        self.read_gate = nn.Parameter(torch.zeros(()))

        self.velocity_feature = nn.Conv2d(
            self.hidden_dim,
            self.physical_dim,
            kernel_size=1,
        )
        self.velocity_norm = nn.LayerNorm(self.physical_dim)
        self.velocity_head = nn.Sequential(
            nn.Linear(self.physical_dim * 2, self.physical_dim * 4),
            nn.SiLU(),
            nn.Linear(self.physical_dim * 4, self.physical_dim),
        )
        self.register_buffer(
            "architecture_version",
            torch.tensor(self.VERSION, dtype=torch.long),
            persistent=True,
        )
        self.last_diagnostics: dict[str, Tensor] = {}

    def _encode_physical_pair(
        self,
        video_physical: Tensor,
        fused_physical: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return a paired read, a spatial compatibility gate, and its update."""
        if video_physical.shape != fused_physical.shape:
            raise ValueError(
                "Video and fused UPCAST tokens must have identical shapes, got "
                f"{tuple(video_physical.shape)} and "
                f"{tuple(fused_physical.shape)}"
            )
        video = self.pair_video_norm(video_physical)
        fused = self.pair_fused_norm(fused_physical)
        pair_features = torch.cat(
            (video, fused, fused - video, video * fused),
            dim=-1,
        )
        pair_context = self.pair_encoder(
            self.pair_input_projection(pair_features)
        )
        update = self.pair_update_scale * torch.tanh(
            self.pair_update(self.pair_update_norm(pair_context))
        )
        innovation = fused_physical - video_physical
        read_physical = innovation + update.to(innovation)

        # Dividing by the prior makes the initial compatibility exactly one,
        # while retaining a smooth path to zero for mismatched pairs.  The
        # upper bound is only 1 / prior, so the scalar max_read_scale remains
        # a meaningful safety bound.
        prior_logit = math.log(
            self.pair_gate_prior / (1.0 - self.pair_gate_prior)
        )
        compatibility = torch.sigmoid(
            self.pair_gate(self.pair_gate_norm(pair_context)) + prior_logit
        ) / self.pair_gate_prior
        return read_physical, compatibility, update

    def _stable_local_cross_attention(
        self,
        query: Tensor,
        key_value: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Evaluate the small masked read in FP32.

        PyTorch's fused half-precision SDPA kernels can occasionally produce a
        non-finite backward result for the rectangular ``1024 x 64`` boolean
        mask used here.  Keep the same MultiheadAttention parameters and local
        receptive field, but make the projections, logits, softmax, and output
        projection explicit in FP32.  The returned tensor is cast back to the
        surrounding activation dtype; gradients still update the original MHA
        parameters.
        """
        attention = self.cross_attention
        if attention._qkv_same_embed_dim:
            raise RuntimeError(
                "Stable local Physical conditioner attention expects separate "
                "query/key/value projections"
            )
        if bool(attention_mask.all(dim=-1).any()):
            raise RuntimeError(
                "Physical conditioner local attention contains an all-masked query"
            )

        output_dtype = query.dtype
        device_type = query.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            query_fp32 = query.float()
            key_value_fp32 = key_value.float()
            projection_bias = attention.in_proj_bias
            if projection_bias is None:
                query_bias = key_bias = value_bias = None
            else:
                query_bias, key_bias, value_bias = projection_bias.float().chunk(3)

            projected_query = F.linear(
                query_fp32,
                attention.q_proj_weight.float(),
                query_bias,
            )
            projected_key = F.linear(
                key_value_fp32,
                attention.k_proj_weight.float(),
                key_bias,
            )
            projected_value = F.linear(
                key_value_fp32,
                attention.v_proj_weight.float(),
                value_bias,
            )

            batch, query_tokens, _ = projected_query.shape
            key_tokens = projected_key.shape[1]
            heads = attention.num_heads
            head_dim = self.hidden_dim // heads
            projected_query = projected_query.reshape(
                batch, query_tokens, heads, head_dim
            ).transpose(1, 2)
            projected_key = projected_key.reshape(
                batch, key_tokens, heads, head_dim
            ).transpose(1, 2)
            projected_value = projected_value.reshape(
                batch, key_tokens, heads, head_dim
            ).transpose(1, 2)

            scores = torch.matmul(
                projected_query,
                projected_key.transpose(-2, -1),
            ) * (head_dim**-0.5)
            scores = scores.masked_fill(
                attention_mask[None, None],
                -1.0e4,
            )
            probabilities = scores.softmax(dim=-1)
            attended = torch.matmul(probabilities, projected_value)
            attended = attended.transpose(1, 2).reshape(
                batch, query_tokens, self.hidden_dim
            )
            attended = F.linear(
                attended,
                attention.out_proj.weight.float(),
                (
                    None
                    if attention.out_proj.bias is None
                    else attention.out_proj.bias.float()
                ),
            )
        return attended.to(dtype=output_dtype)

    def forward(
        self,
        latent: Tensor,
        noisy_physical: Tensor,
        future_indices: Tensor,
        physical_logsnr: Tensor,
        *,
        video_physical: Tensor,
        fused_physical: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if latent.ndim != 5:
            raise ValueError(
                f"Expected latent [B,T,C,H,W], got {tuple(latent.shape)}"
            )
        batch, frames, channels, height, width = latent.shape
        expected_tokens = self.spatial_grid**2
        if noisy_physical.shape != (
            batch,
            expected_tokens,
            self.physical_dim,
        ):
            raise ValueError(
                "Noisy UPCAST tokens must have shape "
                f"[{batch},{expected_tokens},{self.physical_dim}], got "
                f"{tuple(noisy_physical.shape)}"
            )
        if video_physical.shape != noisy_physical.shape:
            raise ValueError(
                "Physical conditioner video tokens must match noisy tokens, got "
                f"{tuple(video_physical.shape)} and "
                f"{tuple(noisy_physical.shape)}"
            )
        if fused_physical.shape != noisy_physical.shape:
            raise ValueError(
                "Physical conditioner fused tokens must match noisy tokens, got "
                f"{tuple(fused_physical.shape)} and "
                f"{tuple(noisy_physical.shape)}"
            )
        if future_indices.shape != (batch,):
            raise ValueError(
                f"future_indices must have shape [{batch}], got "
                f"{tuple(future_indices.shape)}"
            )
        if bool(((future_indices < 0) | (future_indices >= frames)).any()):
            raise ValueError("future_indices contains an out-of-range frame")

        batch_indices = torch.arange(batch, device=latent.device)
        future = latent[batch_indices, future_indices]
        query = rearrange(future, "b c h w -> b (h w) c")
        time_embedding = self.noise_embedding(
            _scalar_sinusoidal_embedding(
                physical_logsnr,
                self.physical_dim,
            ).to(noisy_physical)
        )
        velocity_physical = self.physical_norm(
            noisy_physical
            + self.physical_position.to(noisy_physical)
            + time_embedding[:, None]
        )
        read_physical, pair_compatibility, pair_update = (
            self._encode_physical_pair(video_physical, fused_physical)
        )
        read_tokens = self.physical_norm(
            read_physical
            + self.physical_position.to(read_physical)
            + time_embedding[:, None].to(read_physical)
        )
        normalized_query = self.query_norm(query)
        global_residual = None
        if self.local_read_fraction < 1.0:
            global_residual, _ = self.cross_attention(
                normalized_query,
                read_tokens,
                read_tokens,
                need_weights=False,
            )
        local_residual = None
        if self.local_read_fraction > 0.0:
            # Each video location attends within its aligned UPCAST neighborhood.
            # The global path remains available for camera motion and
            # disocclusion, while this path prevents a 64-token permutation
            # from being an almost identical condition. A neighborhood rather
            # than one key keeps the read query-dependent.
            query_y = (
                torch.arange(height, device=latent.device)
                * self.spatial_grid
                // height
            )
            query_x = (
                torch.arange(width, device=latent.device)
                * self.spatial_grid
                // width
            )
            centers_y = query_y[:, None].expand(height, width).reshape(-1)
            centers_x = query_x[None, :].expand(height, width).reshape(-1)
            token_y = torch.arange(
                self.spatial_grid,
                device=latent.device,
            ).repeat_interleave(self.spatial_grid)
            token_x = torch.arange(
                self.spatial_grid,
                device=latent.device,
            ).repeat(self.spatial_grid)
            local_mask = (
                (centers_y[:, None] - token_y[None]).abs()
                > self.local_read_radius
            ) | (
                (centers_x[:, None] - token_x[None]).abs()
                > self.local_read_radius
            )
            local_residual = self._stable_local_cross_attention(
                normalized_query,
                read_tokens,
                local_mask,
            )
        if global_residual is None:
            residual = local_residual
        elif local_residual is None:
            residual = global_residual
        else:
            residual = (
                (1.0 - self.local_read_fraction) * global_residual
                + self.local_read_fraction * local_residual
            )
        if residual is None:
            raise RuntimeError("Physical conditioner produced no read residual")
        compatibility_map = rearrange(
            pair_compatibility,
            "b (h w) c -> b c h w",
            h=self.spatial_grid,
            w=self.spatial_grid,
        )
        compatibility_map = F.interpolate(
            compatibility_map.float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).to(residual)
        compatibility_tokens = rearrange(
            compatibility_map,
            "b c h w -> b (h w) c",
        )
        residual = residual * compatibility_tokens
        scale = self.max_read_scale * torch.tanh(self.read_gate.float())
        residual = residual * scale.to(residual)
        residual_map = rearrange(
            residual,
            "b (h w) c -> b c h w",
            h=height,
            w=width,
        )
        frame_mask = F.one_hot(
            future_indices.long(),
            num_classes=frames,
        ).to(latent)[:, :, None, None, None]
        conditioned = latent + frame_mask * residual_map[:, None]

        conditioned_future = future + residual_map
        feature_tokens = F.adaptive_avg_pool2d(
            self.velocity_feature(conditioned_future),
            (self.spatial_grid, self.spatial_grid),
        )
        feature_tokens = rearrange(feature_tokens, "b c h w -> b (h w) c")
        velocity = self.velocity_head(
            torch.cat(
                    (
                        self.velocity_norm(feature_tokens),
                        velocity_physical,
                ),
                dim=-1,
            )
        )

        with torch.no_grad():
            query_rms = query.float().square().mean().sqrt().clamp_min(1e-8)
            residual_rms = residual.float().square().mean().sqrt()
            innovation = fused_physical.float() - video_physical.float()
            innovation_rms = innovation.square().mean().sqrt().clamp_min(1e-8)
            self.last_diagnostics = {
                "generative_unit_gate": torch.tanh(self.read_gate.float()).detach(),
                "generative_unit_residual_rms": residual_rms.detach(),
                "generative_unit_residual_ratio": (
                    residual_rms / query_rms
                ).detach(),
                "generative_unit_physical_rms": noisy_physical.float()
                .square()
                .mean()
                .sqrt()
                .detach(),
                "generative_unit_read_physical_rms": read_physical.float()
                .square()
                .mean()
                .sqrt()
                .detach(),
                "generative_unit_pair_cosine": F.cosine_similarity(
                    video_physical.float(),
                    fused_physical.float(),
                    dim=-1,
                ).mean().detach(),
                "generative_unit_pair_gate_mean": pair_compatibility.float()
                .mean()
                .detach(),
                "generative_unit_pair_gate_std": pair_compatibility.float()
                .std(unbiased=False)
                .detach(),
                "generative_unit_pair_gate_min": pair_compatibility.float()
                .amin()
                .detach(),
                "generative_unit_pair_gate_max": pair_compatibility.float()
                .amax()
                .detach(),
                "generative_unit_pair_update_ratio": (
                    pair_update.float().square().mean().sqrt()
                    / innovation_rms
                ).detach(),
                "generative_unit_local_read_fraction": residual_rms.new_tensor(
                    self.local_read_fraction
                ),
                "generative_unit_local_read_radius": residual_rms.new_tensor(
                    self.local_read_radius
                ),
            }
        return conditioned, velocity, pair_compatibility


class GatedWorldStateAdapter(nn.Module):
    """Pose-aware read and update rules for a persistent factorized state.

    Each memory token carries a camera-ray anchor and a confidence value. The
    read path therefore retrieves state by world-space ray compatibility rather
    than by the screen-space token index used by V6. Both residual heads are
    zero initialized, preserving the loaded video model exactly before V7
    adapter training.
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 768,
        num_heads: int = 12,
        max_read_scale: float = 0.15,
        appearance_read_scale: float = 0.25,
        physical_update_rate: float = 0.5,
        appearance_update_rate: float = 0.1,
        logsnr_scale: float = 8.0,
        anchor_dim: int = 9,
        memory_grid_size: int = 4,
        max_state_tokens: int = 160,
        visibility_threshold: float = 0.35,
        visibility_temperature: float = 0.10,
        preference_gate_hidden_dim: int = 64,
        preference_policy_version: int = PREFERENCE_POLICY_VERSION,
        preference_content_dim: int = 32,
        preference_max_correction: float = 0.5,
        physical_age_half_life: float = 8.0,
        appearance_age_half_life: float = 1.5,
        transport_mode: str = "additive",
        transport_max_residual_ratio: float = 0.006,
        transport_max_relation_drift: float = 0.0001,
        transport_min_scale: float = 0.0,
        transport_preserve_token_norm: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            num_heads = math.gcd(hidden_dim, num_heads)
        if num_heads < 1:
            raise ValueError(
                f"No valid attention head count for hidden_dim={hidden_dim}"
            )
        self.hidden_dim = int(hidden_dim)
        self.state_dim = int(state_dim)
        self.read_num_heads = int(num_heads)
        self.max_read_scale = float(max_read_scale)
        self.appearance_read_scale = float(appearance_read_scale)
        self.physical_update_rate = float(physical_update_rate)
        self.appearance_update_rate = float(appearance_update_rate)
        self.logsnr_scale = float(logsnr_scale)
        self.anchor_dim = int(anchor_dim)
        self.memory_grid_size = int(memory_grid_size)
        self.max_state_tokens = int(max_state_tokens)
        self.visibility_threshold = float(visibility_threshold)
        self.visibility_temperature = float(visibility_temperature)
        self.preference_policy_version = int(preference_policy_version)
        self.preference_content_dim = int(preference_content_dim)
        self.preference_max_correction = float(preference_max_correction)
        self.physical_age_half_life = float(physical_age_half_life)
        self.appearance_age_half_life = float(appearance_age_half_life)
        self.transport_mode = str(transport_mode or "additive").lower()
        self.transport_max_residual_ratio = float(transport_max_residual_ratio)
        self.transport_max_relation_drift = float(
            transport_max_relation_drift
        )
        self.transport_min_scale = float(transport_min_scale)
        self.transport_preserve_token_norm = bool(
            transport_preserve_token_norm
        )
        if self.anchor_dim != 9:
            raise ValueError("V7 world-state ray anchors must have 9 channels")
        if self.memory_grid_size < 1 or self.max_state_tokens < 1:
            raise ValueError("World-state memory dimensions must be positive")
        if self.physical_age_half_life <= 0 or self.appearance_age_half_life <= 0:
            raise ValueError("World-state age half-lives must be positive")
        if self.preference_policy_version != PREFERENCE_POLICY_VERSION:
            raise ValueError(
                "Only stable-window world-state preference policy version "
                f"{PREFERENCE_POLICY_VERSION} "
                f"is supported, got {self.preference_policy_version}"
            )
        if self.preference_content_dim < 4:
            raise ValueError("preference_content_dim must be at least 4")
        if not 0.0 < self.preference_max_correction <= 1.0:
            raise ValueError("preference_max_correction must be in (0,1]")
        if self.transport_mode not in {"additive", "relation_trust"}:
            raise ValueError(
                "world_state.transport_mode must be additive or relation_trust"
            )
        if self.transport_max_residual_ratio <= 0.0:
            raise ValueError(
                "world_state.transport_max_residual_ratio must be positive"
            )
        if self.transport_max_relation_drift <= 0.0:
            raise ValueError(
                "world_state.transport_max_relation_drift must be positive"
            )
        if not 0.0 <= self.transport_min_scale <= 1.0:
            raise ValueError("world_state.transport_min_scale must be in [0,1]")

        self.physical_state_norm = nn.LayerNorm(state_dim)
        self.appearance_state_norm = nn.LayerNorm(state_dim)
        self.physical_update_attention = nn.MultiheadAttention(
            state_dim,
            num_heads=max(1, math.gcd(state_dim, num_heads)),
            dropout=0.0,
            batch_first=True,
        )
        self.appearance_update_attention = nn.MultiheadAttention(
            state_dim,
            num_heads=max(1, math.gcd(state_dim, num_heads)),
            dropout=0.0,
            batch_first=True,
        )
        self.physical_update_mlp = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, state_dim * 2),
            nn.SiLU(),
            nn.Linear(state_dim * 2, state_dim),
        )
        self.appearance_update_mlp = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, state_dim * 2),
            nn.SiLU(),
            nn.Linear(state_dim * 2, state_dim),
        )
        self.physical_update_gate = nn.Sequential(
            nn.LayerNorm(state_dim * 2),
            nn.Linear(state_dim * 2, 1),
        )
        self.appearance_update_gate = nn.Sequential(
            nn.LayerNorm(state_dim * 2),
            nn.Linear(state_dim * 2, 1),
        )

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.query_anchor_projection = nn.Sequential(
            nn.LayerNorm(anchor_dim),
            nn.Linear(anchor_dim, hidden_dim),
        )
        self.source_anchor_projection = nn.Sequential(
            nn.LayerNorm(anchor_dim),
            nn.Linear(anchor_dim, hidden_dim),
        )
        self.physical_read_projection = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
        )
        self.appearance_read_projection = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
        )
        self.state_type_embedding = nn.Parameter(torch.zeros(2, hidden_dim))
        self.physical_read_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.appearance_read_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.read_output = nn.Linear(hidden_dim, hidden_dim)
        self.state_refiner = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.read_gate_logit = nn.Parameter(torch.tensor(0.0))
        self.preference_policy = StableWorldStatePreferencePolicy(
            PreferencePolicySpec(
                hidden_dim=self.hidden_dim,
                content_dim=self.preference_content_dim,
                gate_hidden_dim=max(8, int(preference_gate_hidden_dim)),
                max_correction=self.preference_max_correction,
            )
        )

        # Kept only so V10.2/V10.3 checkpoints remain strictly loadable. Age is
        # now an input feature; multiplying the policy by an independently
        # learned one-sided decay made the supervised probability cease to be
        # the action actually applied to the video model.
        self.physical_age_decay_strength = nn.Parameter(torch.tensor(0.0))
        self.appearance_age_decay_strength = nn.Parameter(torch.tensor(0.0))

        nn.init.normal_(self.state_type_embedding, std=0.02)
        nn.init.zeros_(self.read_output.weight)
        nn.init.zeros_(self.read_output.bias)
        nn.init.zeros_(self.state_refiner[-1].weight)
        nn.init.zeros_(self.state_refiner[-1].bias)
        # Start from a deterministic exponential update of the observed
        # spatial tokens. Attention and MLP learn cross-region corrections;
        # their zero-output initialization prevents the migrated V5 features
        # from being distorted by random update rules.
        nn.init.zeros_(self.physical_update_attention.out_proj.weight)
        nn.init.zeros_(self.physical_update_attention.out_proj.bias)
        nn.init.zeros_(self.appearance_update_attention.out_proj.weight)
        nn.init.zeros_(self.appearance_update_attention.out_proj.bias)
        nn.init.zeros_(self.physical_update_mlp[-1].weight)
        nn.init.zeros_(self.physical_update_mlp[-1].bias)
        nn.init.zeros_(self.appearance_update_mlp[-1].weight)
        nn.init.zeros_(self.appearance_update_mlp[-1].bias)
        nn.init.zeros_(self.physical_update_gate[-1].weight)
        nn.init.zeros_(self.appearance_update_gate[-1].weight)
        nn.init.zeros_(self.physical_update_gate[-1].bias)
        nn.init.zeros_(self.appearance_update_gate[-1].bias)
        object.__setattr__(self, "_runtime_state", None)
        object.__setattr__(self, "_runtime_query_anchors", None)
        object.__setattr__(self, "_last_gate_probabilities", None)
        object.__setattr__(self, "_cached_preference_outputs", None)
        object.__setattr__(self, "_last_preference_inputs", None)
        object.__setattr__(self, "_runtime_diagnostic_sums", {})
        object.__setattr__(self, "_runtime_diagnostic_maxima", {})
        object.__setattr__(self, "_runtime_diagnostic_minima", {})
        self._runtime_diagnostic_count = 0
        self._runtime_scale = 1.0
        self._runtime_branch_mode = "adaptive"
        self.last_diagnostics: dict[str, Tensor] = {}

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Migrate V10.1-V10.3 adapters while keeping strict loading.

        Only preference-policy parameters are synthesized. Any unrelated
        architecture mismatch still fails under strict loading.
        """
        preference_prefixes = (
            "preference_policy.",
            "physical_age_decay_strength",
            "appearance_age_decay_strength",
        )
        for name, value in self.state_dict().items():
            if name.startswith(preference_prefixes):
                state_dict.setdefault(prefix + name, value.detach().clone())
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def preference_parameters(self) -> list[nn.Parameter]:
        """Return exactly the parameters used by policy-only calibration."""
        return list(self.preference_policy.parameters())

    @staticmethod
    def _bounded_decay_strength(value: Tensor) -> Tensor:
        """Deprecated clamp retained for V10.3 checkpoint-level tests."""
        bounded = value.clamp(0.0, 10.0)
        return value + (bounded - value).detach()

    @property
    def gate_probabilities(self) -> dict[str, Tensor]:
        probabilities = self._last_gate_probabilities
        return {} if probabilities is None else probabilities

    @property
    def gate_corrections(self) -> dict[str, Tensor]:
        outputs = self._cached_preference_outputs
        if outputs is None:
            return {}
        return {
            "physical": outputs["physical_correction"],
            "appearance": outputs["appearance_correction"],
        }

    @property
    def preference_cache_inputs(self) -> dict[str, Tensor]:
        inputs = self._last_preference_inputs
        return {} if inputs is None else inputs

    @property
    def window_diagnostics(self) -> dict[str, Tensor]:
        """Aggregate diagnostics across all denoising calls in one window."""
        count = int(self._runtime_diagnostic_count)
        if count == 0:
            return self.last_diagnostics
        summary = {
            name: total / float(count)
            for name, total in self._runtime_diagnostic_sums.items()
        }
        # Keep insertion order deterministic across DDP ranks. Validation logs
        # these scalars with collectives, so a set here can silently reduce
        # different metric names together under different hash seeds.
        extrema_names = (
            "world_state_residual_ratio_raw",
            "world_state_residual_ratio_applied",
            "world_state_relation_drift_raw",
            "world_state_relation_drift_applied",
            "world_state_transport_scale",
        )
        for name in extrema_names:
            if name in self._runtime_diagnostic_maxima:
                summary[f"{name}_max"] = self._runtime_diagnostic_maxima[name]
                summary[f"{name}_min"] = self._runtime_diagnostic_minima[name]
        reference = next(iter(summary.values()))
        summary["world_state_denoising_diagnostic_calls"] = reference.new_tensor(
            float(count)
        )
        return summary

    def _reset_runtime_diagnostics(self) -> None:
        object.__setattr__(self, "_runtime_diagnostic_sums", {})
        object.__setattr__(self, "_runtime_diagnostic_maxima", {})
        object.__setattr__(self, "_runtime_diagnostic_minima", {})
        self._runtime_diagnostic_count = 0

    @torch.no_grad()
    def _record_runtime_diagnostics(self) -> None:
        if not self.last_diagnostics:
            return
        for name, value in self.last_diagnostics.items():
            if not torch.is_tensor(value) or value.numel() != 1:
                continue
            value = value.detach().float()
            previous_sum = self._runtime_diagnostic_sums.get(name)
            self._runtime_diagnostic_sums[name] = (
                value if previous_sum is None else previous_sum + value
            )
            previous_max = self._runtime_diagnostic_maxima.get(name)
            previous_min = self._runtime_diagnostic_minima.get(name)
            self._runtime_diagnostic_maxima[name] = (
                value if previous_max is None else torch.maximum(previous_max, value)
            )
            self._runtime_diagnostic_minima[name] = (
                value if previous_min is None else torch.minimum(previous_min, value)
            )
        self._runtime_diagnostic_count += 1

    def advance_state_age(
        self,
        previous_state: dict[str, Tensor] | None,
    ) -> dict[str, Tensor] | None:
        """Advance memory metadata once per completed autoregressive window."""
        if previous_state is None:
            return None
        physical = previous_state["physical"]
        state = dict(previous_state)
        age = previous_state.get("age")
        if age is None:
            age = physical.new_zeros(*physical.shape[:2], 1)
        state["age"] = age + 1.0
        window_count = previous_state.get("window_count")
        if window_count is None:
            window_count = physical.new_ones(physical.shape[0], 1)
        state["window_count"] = window_count + 1.0
        return state

    @staticmethod
    def _validate_observation(physical: Tensor, appearance: Tensor) -> None:
        if physical.ndim != 3 or physical.shape != appearance.shape:
            raise ValueError(
                "World-state observations must be matching [B,N,D] tensors, got "
                f"{tuple(physical.shape)} and {tuple(appearance.shape)}"
            )

    def _compress_observation(
        self,
        physical: Tensor,
        appearance: Tensor,
        anchors: Tensor,
        confidence: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        token_count = physical.shape[1]
        source_grid = math.isqrt(token_count)
        target_grid = self.memory_grid_size
        if source_grid * source_grid != token_count or source_grid <= target_grid:
            return physical, appearance, anchors, confidence

        def pool(tokens: Tensor) -> Tensor:
            channels = tokens.shape[-1]
            feature_map = rearrange(
                tokens,
                "b (h w) c -> b c h w",
                h=source_grid,
                w=source_grid,
            )
            pooled = F.adaptive_avg_pool2d(
                feature_map.float(),
                (target_grid, target_grid),
            ).to(tokens)
            return rearrange(pooled, "b c h w -> b (h w) c", c=channels)

        return pool(physical), pool(appearance), pool(anchors), pool(confidence)

    def _validate_metadata(
        self,
        physical: Tensor,
        anchors: Tensor,
        confidence: Tensor,
    ) -> None:
        expected_anchor = (*physical.shape[:2], self.anchor_dim)
        expected_confidence = (*physical.shape[:2], 1)
        if anchors.shape != expected_anchor:
            raise ValueError(
                f"Expected world-state anchors {expected_anchor}, got {tuple(anchors.shape)}"
            )
        if confidence.shape != expected_confidence:
            raise ValueError(
                "Expected world-state confidence "
                f"{expected_confidence}, got {tuple(confidence.shape)}"
            )

    def update_state(
        self,
        physical_observation: Tensor,
        appearance_observation: Tensor,
        previous_state: dict[str, Tensor] | None = None,
        *,
        anchors: Tensor | None = None,
        confidence: Tensor | None = None,
        detach_previous: bool = True,
    ) -> dict[str, Tensor]:
        """Append one anchored observation to a bounded recurrent memory bank."""
        self._validate_observation(physical_observation, appearance_observation)
        if anchors is None:
            anchors = physical_observation.new_zeros(
                *physical_observation.shape[:2], self.anchor_dim
            )
        if confidence is None:
            confidence = physical_observation.new_ones(
                *physical_observation.shape[:2], 1
            )
        anchors = anchors.to(physical_observation)
        confidence = confidence.to(physical_observation).clamp(0.0, 1.0)
        self._validate_metadata(physical_observation, anchors, confidence)
        physical_observation = self.physical_state_norm(physical_observation)
        appearance_observation = self.appearance_state_norm(appearance_observation)
        (
            physical_observation,
            appearance_observation,
            anchors,
            confidence,
        ) = self._compress_observation(
            physical_observation,
            appearance_observation,
            anchors,
            confidence,
        )
        observation_age = physical_observation.new_zeros(
            *physical_observation.shape[:2], 1
        )
        if previous_state is None:
            return {
                "physical": physical_observation,
                "appearance": appearance_observation,
                "anchors": anchors,
                "confidence": confidence,
                "age": observation_age,
                "window_count": physical_observation.new_ones(
                    physical_observation.shape[0], 1
                ),
            }

        previous_physical = previous_state["physical"]
        previous_appearance = previous_state["appearance"]
        self._validate_observation(previous_physical, previous_appearance)
        previous_anchors = previous_state.get("anchors")
        previous_confidence = previous_state.get("confidence")
        if previous_anchors is None:
            previous_anchors = previous_physical.new_zeros(
                *previous_physical.shape[:2], self.anchor_dim
            )
        if previous_confidence is None:
            previous_confidence = previous_physical.new_ones(
                *previous_physical.shape[:2], 1
            )
        previous_age = previous_state.get("age")
        if previous_age is None:
            previous_age = previous_physical.new_zeros(
                *previous_physical.shape[:2], 1
            )
        previous_window_count = previous_state.get("window_count")
        if previous_window_count is None:
            previous_window_count = previous_physical.new_ones(
                previous_physical.shape[0], 1
            )
        self._validate_metadata(
            previous_physical,
            previous_anchors,
            previous_confidence,
        )
        if previous_age.shape != (*previous_physical.shape[:2], 1):
            raise ValueError(
                "Expected world-state age metadata "
                f"{(*previous_physical.shape[:2], 1)}, got {tuple(previous_age.shape)}"
            )
        if previous_window_count.shape != (previous_physical.shape[0], 1):
            raise ValueError(
                "Expected world-state window_count [B,1], got "
                f"{tuple(previous_window_count.shape)}"
            )
        if detach_previous:
            previous_physical = previous_physical.detach()
            previous_appearance = previous_appearance.detach()
            previous_anchors = previous_anchors.detach()
            previous_confidence = previous_confidence.detach()
            previous_age = previous_age.detach()
            previous_window_count = previous_window_count.detach()

        physical_delta, _ = self.physical_update_attention(
            physical_observation,
            self.physical_state_norm(previous_physical),
            self.physical_state_norm(previous_physical),
            need_weights=False,
        )
        appearance_delta, _ = self.appearance_update_attention(
            appearance_observation,
            self.appearance_state_norm(previous_appearance),
            self.appearance_state_norm(previous_appearance),
            need_weights=False,
        )
        physical_candidate = physical_observation + physical_delta
        physical_candidate = physical_candidate + self.physical_update_mlp(
            physical_candidate
        )
        appearance_candidate = appearance_observation + appearance_delta
        appearance_candidate = appearance_candidate + self.appearance_update_mlp(
            appearance_candidate
        )
        physical_gate = self.physical_update_rate * torch.sigmoid(
            self.physical_update_gate(
                torch.cat((physical_observation, physical_candidate), dim=-1)
            )
        )
        appearance_gate = self.appearance_update_rate * torch.sigmoid(
            self.appearance_update_gate(
                torch.cat((appearance_observation, appearance_candidate), dim=-1)
            )
        )
        physical_gate = physical_gate.to(physical_candidate)
        appearance_gate = appearance_gate.to(appearance_candidate)
        enriched_physical = torch.lerp(
            physical_observation,
            physical_candidate,
            physical_gate,
        )
        enriched_appearance = torch.lerp(
            appearance_observation,
            appearance_candidate,
            appearance_gate,
        )
        state = {
            "physical": torch.cat((previous_physical, enriched_physical), dim=1),
            "appearance": torch.cat((previous_appearance, enriched_appearance), dim=1),
            "anchors": torch.cat((previous_anchors, anchors), dim=1),
            "confidence": torch.cat(
                (previous_confidence, confidence), dim=1
            ),
            "age": torch.cat((previous_age, observation_age), dim=1),
        }
        if state["physical"].shape[1] > self.max_state_tokens:
            state = {
                key: value[:, -self.max_state_tokens :]
                for key, value in state.items()
            }
        state["window_count"] = previous_window_count
        for key in ("reference_origin", "reference_scale"):
            if key in previous_state:
                state[key] = previous_state[key]
        return state

    def set_runtime_state(
        self,
        state: dict[str, Tensor] | None,
        *,
        query_anchors: Tensor | None = None,
        scale: float = 1.0,
        detach: bool = True,
        branch_mode: str = "adaptive",
    ) -> None:
        branch_mode = str(branch_mode or "adaptive").lower()
        self._reset_runtime_diagnostics()
        if branch_mode not in {
            "adaptive",
            "full",
            "physical",
            "appearance",
            "none",
        }:
            raise ValueError(
                "World-state branch_mode must be adaptive, full, physical, "
                "appearance, or none; "
                f"got {branch_mode!r}"
            )
        if state is None:
            object.__setattr__(self, "_runtime_state", None)
            object.__setattr__(self, "_runtime_query_anchors", None)
            object.__setattr__(self, "_last_gate_probabilities", None)
            object.__setattr__(self, "_cached_preference_outputs", None)
            object.__setattr__(self, "_last_preference_inputs", None)
            self._runtime_scale = float(scale)
            self._runtime_branch_mode = branch_mode
            return
        physical = state["physical"]
        appearance = state["appearance"]
        self._validate_observation(physical, appearance)
        anchors = state.get("anchors")
        confidence = state.get("confidence")
        if anchors is None:
            anchors = physical.new_zeros(*physical.shape[:2], self.anchor_dim)
        if confidence is None:
            confidence = physical.new_ones(*physical.shape[:2], 1)
        age = state.get("age")
        if age is None:
            age = physical.new_zeros(*physical.shape[:2], 1)
        window_count = state.get("window_count")
        if window_count is None:
            window_count = physical.new_ones(physical.shape[0], 1)
        self._validate_metadata(physical, anchors, confidence)
        if age.shape != (*physical.shape[:2], 1):
            raise ValueError(
                f"World-state age must be [B,N,1], got {tuple(age.shape)}"
            )
        if window_count.shape != (physical.shape[0], 1):
            raise ValueError(
                "World-state window_count must be [B,1], got "
                f"{tuple(window_count.shape)}"
            )
        if query_anchors is not None:
            if query_anchors.ndim != 4 or query_anchors.shape[-1] != self.anchor_dim:
                raise ValueError(
                    "World-state query anchors must be [B,T,N,9], got "
                    f"{tuple(query_anchors.shape)}"
                )
        if detach:
            physical = physical.detach()
            appearance = appearance.detach()
            anchors = anchors.detach()
            confidence = confidence.detach()
            age = age.detach()
            window_count = window_count.detach()
            if query_anchors is not None:
                query_anchors = query_anchors.detach()
        runtime_state = {
            "physical": physical,
            "appearance": appearance,
            "anchors": anchors,
            "confidence": confidence,
            "age": age,
            "window_count": window_count,
        }
        object.__setattr__(self, "_runtime_state", runtime_state)
        object.__setattr__(self, "_runtime_query_anchors", query_anchors)
        self._runtime_scale = float(scale)
        self._runtime_branch_mode = branch_mode
        self._cache_window_preference(
            runtime_state,
            query_anchors,
            branch_mode=branch_mode,
        )

    def clear_runtime_state(self) -> None:
        object.__setattr__(self, "_runtime_state", None)
        object.__setattr__(self, "_runtime_query_anchors", None)
        object.__setattr__(self, "_last_gate_probabilities", None)
        object.__setattr__(self, "_cached_preference_outputs", None)
        object.__setattr__(self, "_last_preference_inputs", None)
        self._runtime_branch_mode = "adaptive"
        self.last_diagnostics = {}
        self._reset_runtime_diagnostics()

    @staticmethod
    def _match_batch(tokens: Tensor, batch_size: int) -> Tensor:
        if tokens.shape[0] == batch_size:
            return tokens
        if batch_size % tokens.shape[0] != 0:
            raise ValueError(
                "World-state batch cannot match denoising batch: "
                f"state={tokens.shape[0]}, denoising={batch_size}"
            )
        return tokens.repeat_interleave(batch_size // tokens.shape[0], dim=0)

    def _resize_query_anchors(
        self,
        anchors: Tensor,
        *,
        height: int,
        width: int,
    ) -> Tensor:
        source_tokens = anchors.shape[2]
        source_grid = math.isqrt(source_tokens)
        if source_grid * source_grid != source_tokens:
            raise ValueError(
                "World-state query-anchor token count must form a square grid, "
                f"got {source_tokens}"
            )
        if (source_grid, source_grid) == (height, width):
            return anchors
        feature_map = rearrange(
            anchors,
            "b t (h w) c -> (b t) c h w",
            h=source_grid,
            w=source_grid,
        )
        resized = F.interpolate(
            feature_map.float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).to(anchors)
        return rearrange(
            resized,
            "(b t) c h w -> b t (h w) c",
            b=anchors.shape[0],
            t=anchors.shape[1],
        )

    def _ray_compatibility(
        self,
        query_anchors: Tensor,
        source_anchors: Tensor,
        confidence: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        query_direction = F.normalize(query_anchors[..., :3].float(), dim=-1)
        source_direction = F.normalize(source_anchors[..., :3].float(), dim=-1)
        query_moment = query_anchors[..., 3:6].float()
        source_moment = source_anchors[..., 3:6].float()
        direction_score = 0.5 * (
            torch.einsum("btqd,bkd->btqk", query_direction, source_direction)
            + 1.0
        )
        reciprocal = (
            torch.einsum("btqd,bkd->btqk", query_direction, source_moment)
            + torch.einsum("bkd,btqd->btqk", source_direction, query_moment)
        ).abs()
        line_score = torch.exp(-reciprocal / 0.25)
        compatibility = 0.55 * direction_score + 0.45 * line_score
        compatibility = compatibility * confidence[:, None, None, :, 0].float()
        best = compatibility.amax(dim=-1)
        visibility = torch.sigmoid(
            (best - self.visibility_threshold)
            / max(self.visibility_temperature, 1e-4)
        )
        return visibility[..., None], compatibility

    def _fallback_policy_query_anchors(
        self,
        source_anchors: Tensor,
    ) -> Tensor:
        """Use the newest memory grid when a caller has no future pose grid."""
        target_tokens = self.memory_grid_size * self.memory_grid_size
        if source_anchors.shape[1] >= target_tokens:
            anchors = source_anchors[:, -target_tokens:]
        else:
            repeats = math.ceil(target_tokens / source_anchors.shape[1])
            anchors = source_anchors.repeat(1, repeats, 1)[:, -target_tokens:]
        return anchors[:, None]

    @torch.no_grad()
    def _stable_preference_inputs(
        self,
        state: dict[str, Tensor],
        query_anchors: Tensor | None,
    ) -> dict[str, Tensor]:
        """Summarize one rollout window without consulting the noisy query.

        Pose embeddings query the two frozen memory branches once.  The
        resulting features are fixed for every denoising step in this window
        and can therefore be cached and trained offline.
        """
        physical_state = state["physical"]
        appearance_state = state["appearance"]
        source_anchors = state["anchors"].to(physical_state)
        confidence = state["confidence"].to(physical_state)
        age = state["age"].to(physical_state)
        window_count = state["window_count"].to(physical_state)
        if query_anchors is None:
            query_anchors = self._fallback_policy_query_anchors(source_anchors)
        else:
            query_anchors = query_anchors.to(physical_state)
            query_anchors = self._resize_query_anchors(
                query_anchors,
                height=self.memory_grid_size,
                width=self.memory_grid_size,
            )
        batch_size, temporal_length, query_tokens, _ = query_anchors.shape
        if physical_state.shape[0] != batch_size:
            raise ValueError(
                "Stable policy state/query batches must match, got "
                f"{physical_state.shape[0]} and {batch_size}"
            )

        anchor_embedding = self.source_anchor_projection(source_anchors)
        physical = (
            self.physical_read_projection(physical_state)
            + anchor_embedding
            + self.state_type_embedding[0].to(physical_state)[None, None]
        ) * confidence
        appearance = (
            self.appearance_read_projection(appearance_state)
            + anchor_embedding
            + self.state_type_embedding[1].to(appearance_state)[None, None]
        ) * confidence
        query = self.query_norm(self.query_anchor_projection(query_anchors))
        visibility, compatibility = self._ray_compatibility(
            query_anchors,
            source_anchors,
            confidence,
        )
        metadata_weights = compatibility / compatibility.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        confidence_summary = torch.einsum(
            "btqk,bkc->btqc",
            metadata_weights,
            confidence.float(),
        )
        age_summary = torch.einsum(
            "btqk,bkc->btqc",
            metadata_weights,
            age.float(),
        )
        attention_bias = rearrange(
            compatibility.clamp_min(1.0e-4).log(),
            "b t q k -> (b t) q k",
        ).repeat_interleave(self.read_num_heads, dim=0).to(query)
        query_flat = rearrange(query, "b t n c -> (b t) n c")
        physical = physical.repeat_interleave(temporal_length, dim=0)
        appearance = appearance.repeat_interleave(temporal_length, dim=0)
        attention_backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
        with sdpa_kernel(attention_backends):
            physical_read, _ = self.physical_read_attention(
                query_flat,
                physical,
                physical,
                attn_mask=attention_bias,
                need_weights=False,
            )
            appearance_read, _ = self.appearance_read_attention(
                query_flat,
                appearance,
                appearance,
                attn_mask=attention_bias,
                need_weights=False,
            )
        query = query.detach().float()
        physical_read = rearrange(
            physical_read,
            "(b t) n c -> b t n c",
            b=batch_size,
            t=temporal_length,
        ).detach().float()
        appearance_read = rearrange(
            appearance_read,
            "(b t) n c -> b t n c",
            b=batch_size,
            t=temporal_length,
        ).detach().float()

        query_unit = F.normalize(query, dim=-1, eps=1.0e-6)
        physical_unit = F.normalize(physical_read, dim=-1, eps=1.0e-6)
        appearance_unit = F.normalize(appearance_read, dim=-1, eps=1.0e-6)
        query_physical = (query_unit * physical_unit).sum(dim=-1)
        query_appearance = (query_unit * appearance_unit).sum(dim=-1)
        physical_appearance = (physical_unit * appearance_unit).sum(dim=-1)

        def mean_std(values: Tensor) -> tuple[Tensor, Tensor]:
            return (
                values.mean(dim=(1, 2)),
                values.std(dim=(1, 2), unbiased=False),
            )

        qp_mean, qp_std = mean_std(query_physical)
        qa_mean, qa_std = mean_std(query_appearance)
        pa_mean = physical_appearance.mean(dim=(1, 2))
        query_rms = query.square().mean(dim=(1, 2, 3)).sqrt().clamp_min(1.0e-6)
        physical_rms = physical_read.square().mean(dim=(1, 2, 3)).sqrt()
        appearance_rms = appearance_read.square().mean(dim=(1, 2, 3)).sqrt()
        disagreement = (
            (physical_read - appearance_read)
            .square()
            .mean(dim=(1, 2, 3))
            .sqrt()
            / (physical_rms + appearance_rms).clamp_min(1.0e-6)
        )
        relations = torch.stack(
            (
                qp_mean,
                qp_std,
                qa_mean,
                qa_std,
                pa_mean,
                physical_rms / query_rms,
                appearance_rms / query_rms,
                disagreement,
            ),
            dim=-1,
        )

        camera_origin = query_anchors[..., 6:9].float().mean(dim=2)
        camera_direction = query_anchors[..., :3].float().mean(dim=2)
        origin_delta = (camera_origin[:, -1] - camera_origin[:, 0]).norm(dim=-1)
        direction_delta = (
            camera_direction[:, -1] - camera_direction[:, 0]
        ).norm(dim=-1)
        metadata = torch.stack(
            (
                torch.tanh(origin_delta),
                torch.tanh(direction_delta),
                visibility.float().mean(dim=(1, 2, 3)),
                confidence_summary.mean(dim=(1, 2, 3)),
                (
                    torch.log1p(age_summary).mean(dim=(1, 2, 3))
                    / math.log(17.0)
                ).clamp(0.0, 2.0),
                (
                    torch.log1p(window_count.float().squeeze(-1))
                    / math.log(6.0)
                ).clamp(0.0, 2.0),
            ),
            dim=-1,
        )
        return {
            "query_frames": query.mean(dim=2),
            "physical_frames": physical_read.mean(dim=2),
            "appearance_frames": appearance_read.mean(dim=2),
            "metadata": metadata,
            "relations": relations,
        }

    def _cache_window_preference(
        self,
        state: dict[str, Tensor],
        query_anchors: Tensor | None,
        *,
        branch_mode: str,
    ) -> None:
        batch_size = state["physical"].shape[0]
        if branch_mode == "adaptive":
            inputs = self._stable_preference_inputs(state, query_anchors)
            with torch.autocast(
                device_type=state["physical"].device.type,
                enabled=False,
            ):
                outputs = self.preference_policy(inputs)
            object.__setattr__(self, "_last_preference_inputs", inputs)
        else:
            zero = state["physical"].new_zeros(
                batch_size,
                dtype=torch.float32,
            )
            outputs = {
                "physical_correction": zero,
                "appearance_correction": zero.clone(),
                "physical_probability": zero + 0.5,
                "appearance_probability": zero + 0.5,
            }
            object.__setattr__(self, "_last_preference_inputs", None)
        object.__setattr__(self, "_cached_preference_outputs", outputs)
        object.__setattr__(
            self,
            "_last_gate_probabilities",
            {
                "physical": outputs["physical_probability"],
                "appearance": outputs["appearance_probability"],
            },
        )

    @staticmethod
    def _local_relation_drift(
        reference: Tensor,
        candidate: Tensor,
        *,
        batch_size: int,
        temporal_length: int,
        height: int,
        width: int,
    ) -> Tensor:
        """Measure local spatial and temporal cosine-relation distortion."""
        if reference.shape != candidate.shape:
            raise ValueError(
                "World-state relation tensors must match, got "
                f"{tuple(reference.shape)} and {tuple(candidate.shape)}"
            )
        reference = rearrange(
            F.normalize(reference.float(), dim=-1, eps=1.0e-6),
            "(b t) (h w) c -> b t h w c",
            b=batch_size,
            t=temporal_length,
            h=height,
            w=width,
        )
        candidate = rearrange(
            F.normalize(candidate.float(), dim=-1, eps=1.0e-6),
            "(b t) (h w) c -> b t h w c",
            b=batch_size,
            t=temporal_length,
            h=height,
            w=width,
        )

        components: list[tuple[float, Tensor]] = []

        def append_relation(
            weight: float,
            ref_a: Tensor,
            ref_b: Tensor,
            cand_a: Tensor,
            cand_b: Tensor,
        ) -> None:
            ref_relation = (ref_a * ref_b).sum(dim=-1)
            cand_relation = (cand_a * cand_b).sum(dim=-1)
            drift = (cand_relation - ref_relation).abs().flatten(1).mean(dim=1)
            components.append((weight, drift))

        if width > 1:
            append_relation(
                0.25,
                reference[:, :, :, :-1],
                reference[:, :, :, 1:],
                candidate[:, :, :, :-1],
                candidate[:, :, :, 1:],
            )
        if height > 1:
            append_relation(
                0.25,
                reference[:, :, :-1],
                reference[:, :, 1:],
                candidate[:, :, :-1],
                candidate[:, :, 1:],
            )
        if temporal_length > 1:
            append_relation(
                0.35,
                reference[:, :-1],
                reference[:, 1:],
                candidate[:, :-1],
                candidate[:, 1:],
            )
        if temporal_length > 2:
            append_relation(
                0.15,
                reference[:, :1],
                reference[:, -1:],
                candidate[:, :1],
                candidate[:, -1:],
            )
        if not components:
            return reference.new_zeros(batch_size)
        total_weight = sum(weight for weight, _ in components)
        return sum(weight * value for weight, value in components) / total_weight

    def _preserve_query_token_norm(
        self,
        query: Tensor,
        candidate: Tensor,
    ) -> Tensor:
        if not self.transport_preserve_token_norm:
            return candidate
        query_fp32 = query.float()
        candidate_fp32 = candidate.float()
        query_norm = query_fp32.norm(dim=-1, keepdim=True)
        candidate_norm = candidate_fp32.norm(dim=-1, keepdim=True)
        normalized = candidate_fp32 * (
            query_norm / candidate_norm.clamp_min(1.0e-6)
        )
        normalized = torch.where(
            query_norm > 1.0e-6,
            normalized,
            candidate_fp32,
        )
        return normalized.to(candidate)

    def _transport_residual(
        self,
        query: Tensor,
        residual: Tensor,
        *,
        batch_size: int,
        temporal_length: int,
        height: int,
        width: int,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Apply a bounded, relation-preserving world-state update.

        The trust scale is a detached runtime safeguard, not a learned utility
        predictor. Gradients still pass through the retained residual and the
        norm-preserving reorganization if the adapter is later fine-tuned.
        """
        query_grouped = rearrange(
            query.float(),
            "(b t) n c -> b t n c",
            b=batch_size,
            t=temporal_length,
        )
        residual_grouped = rearrange(
            residual.float(),
            "(b t) n c -> b t n c",
            b=batch_size,
            t=temporal_length,
        )
        query_rms = query_grouped.square().mean(dim=(1, 2, 3)).sqrt()
        residual_rms = residual_grouped.square().mean(dim=(1, 2, 3)).sqrt()
        residual_ratio = residual_rms / query_rms.clamp_min(1.0e-6)

        with torch.no_grad():
            raw_candidate = query + residual
            raw_relation_drift = self._local_relation_drift(
                query,
                raw_candidate,
                batch_size=batch_size,
                temporal_length=temporal_length,
                height=height,
                width=width,
            )

        if self.transport_mode == "additive":
            scale = residual_ratio.new_ones(batch_size)
            output = query + residual
        else:
            with torch.no_grad():
                norm_scale = (
                    self.transport_max_residual_ratio
                    / residual_ratio.clamp_min(1.0e-8)
                ).clamp(max=1.0)
                relation_scale = (
                    self.transport_max_relation_drift
                    / raw_relation_drift.clamp_min(1.0e-8)
                ).clamp(max=1.0)
                scale = torch.minimum(norm_scale, relation_scale)
                scale = scale.clamp(min=self.transport_min_scale, max=1.0)
            scale_tokens = scale.to(query).repeat_interleave(
                temporal_length,
                dim=0,
            )[:, None, None]
            transported = self._preserve_query_token_norm(
                query,
                query + scale_tokens * residual,
            )
            safe_candidate = query + residual
            clipped = scale < 1.0 - 1.0e-6
            output = torch.where(
                clipped.to(device=query.device).repeat_interleave(
                    temporal_length,
                    dim=0,
                )[:, None, None],
                transported,
                safe_candidate,
            )

            # Cosine drift is only approximately linear in the residual scale.
            # A second detached correction enforces the configured budget.
            with torch.no_grad():
                first_drift = self._local_relation_drift(
                    query,
                    output,
                    batch_size=batch_size,
                    temporal_length=temporal_length,
                    height=height,
                    width=width,
                )
                correction = (
                    self.transport_max_relation_drift
                    / first_drift.clamp_min(1.0e-8)
                ).clamp(max=1.0)
                corrected_scale = (scale * correction).clamp(
                    min=self.transport_min_scale,
                    max=1.0,
                )
            scale = corrected_scale
            scale_tokens = scale.to(query).repeat_interleave(
                temporal_length,
                dim=0,
            )[:, None, None]
            transported = self._preserve_query_token_norm(
                query,
                query + scale_tokens * residual,
            )
            clipped = scale < 1.0 - 1.0e-6
            output = torch.where(
                clipped.to(device=query.device).repeat_interleave(
                    temporal_length,
                    dim=0,
                )[:, None, None],
                transported,
                safe_candidate,
            )

        with torch.no_grad():
            applied_relation_drift = self._local_relation_drift(
                query,
                output,
                batch_size=batch_size,
                temporal_length=temporal_length,
                height=height,
                width=width,
            )
            applied_residual = output.float() - query.float()
            applied_grouped = rearrange(
                applied_residual,
                "(b t) n c -> b t n c",
                b=batch_size,
                t=temporal_length,
            )
            applied_rms = applied_grouped.square().mean(
                dim=(1, 2, 3)
            ).sqrt()
            query_norm = query.float().norm(dim=-1)
            output_norm = output.float().norm(dim=-1)
            token_norm_error = (
                (output_norm - query_norm).abs()
                / query_norm.clamp_min(1.0e-6)
            ).mean()
        diagnostics = {
            "world_state_residual_rms_raw": residual_rms.mean(),
            "world_state_residual_ratio_raw": residual_ratio.mean(),
            "world_state_residual_ratio_applied": (
                applied_rms / query_rms.clamp_min(1.0e-6)
            ).mean(),
            "world_state_transport_scale": scale.mean(),
            "world_state_transport_clipped_fraction": (
                scale < 1.0 - 1.0e-6
            ).float().mean(),
            "world_state_relation_drift_raw": raw_relation_drift.mean(),
            "world_state_relation_drift_applied": applied_relation_drift.mean(),
            "world_state_token_norm_relative_error": token_norm_error,
            "world_state_transport_mode": residual.new_tensor(
                {"additive": 0.0, "relation_trust": 1.0}[
                    self.transport_mode
                ]
            ),
        }
        return output, diagnostics

    def forward(
        self,
        x: Tensor,
        noise_levels: Tensor,
        temporal_length: int,
    ) -> Tensor:
        state = self._runtime_state
        if state is None or self._runtime_scale == 0.0:
            return x
        if x.ndim != 4 or x.shape[0] % temporal_length != 0:
            raise ValueError(
                f"Expected [B*T,C,H,W] with T={temporal_length}, got {tuple(x.shape)}"
            )
        batch_size = x.shape[0] // temporal_length
        physical = self._match_batch(state["physical"], batch_size).to(x)
        appearance = self._match_batch(state["appearance"], batch_size).to(x)
        source_anchors = self._match_batch(state["anchors"], batch_size).to(x)
        confidence = self._match_batch(state["confidence"], batch_size).to(x)
        age = self._match_batch(state["age"], batch_size).to(x)
        window_count = self._match_batch(
            state["window_count"], batch_size
        ).to(x)
        anchor_embedding = self.source_anchor_projection(source_anchors)
        physical = (
            self.physical_read_projection(physical)
            + anchor_embedding
            + self.state_type_embedding[0].to(physical)[None, None]
        ) * confidence
        appearance = (
            self.appearance_read_projection(appearance)
            + anchor_embedding
            + self.state_type_embedding[1].to(appearance)[None, None]
        ) * confidence
        height, width = x.shape[-2:]
        query = rearrange(x, "bt c h w -> bt (h w) c")
        query_anchors = self._runtime_query_anchors
        if query_anchors is None:
            visibility = query.new_ones(
                batch_size, temporal_length, height * width, 1
            )
            normalized_query = self.query_norm(query)
            attention_bias = None
            compatibility_mean = query.new_ones(())
            confidence_summary = confidence.mean(dim=1)[:, None, None, :].expand(
                -1,
                temporal_length,
                height * width,
                -1,
            )
            age_summary = age.mean(dim=1)[:, None, None, :].expand_as(
                confidence_summary
            )
        else:
            query_anchors = self._match_batch(query_anchors, batch_size).to(x)
            if query_anchors.shape[1] != temporal_length:
                raise ValueError(
                    "World-state query-anchor temporal mismatch: "
                    f"{query_anchors.shape[1]} != {temporal_length}"
                )
            query_anchors = self._resize_query_anchors(
                query_anchors,
                height=height,
                width=width,
            )
            visibility, compatibility = self._ray_compatibility(
                query_anchors,
                source_anchors,
                confidence,
            )
            visibility = visibility.to(query)
            compatibility_mean = compatibility.mean().to(query)
            metadata_weights = compatibility / compatibility.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            confidence_summary = torch.einsum(
                "btqk,bkc->btqc",
                metadata_weights,
                confidence.float(),
            ).to(query)
            age_summary = torch.einsum(
                "btqk,bkc->btqc",
                metadata_weights,
                age.float(),
            ).to(query)
            attention_bias = compatibility.clamp_min(1e-4).log()
            attention_bias = rearrange(
                attention_bias,
                "b t q k -> (b t) q k",
            ).repeat_interleave(self.read_num_heads, dim=0).to(query)
            query_anchor_embedding = self.query_anchor_projection(query_anchors)
            normalized_query = self.query_norm(
                query + rearrange(
                    query_anchor_embedding,
                    "b t n c -> (b t) n c",
                )
            )
        physical = physical.repeat_interleave(temporal_length, dim=0)
        appearance = appearance.repeat_interleave(temporal_length, dim=0)
        # PyTorch 2.5 may dispatch a 3-D additive MHA mask to cuDNN SDPA,
        # although that backend cannot plan this graph. Keep the restriction
        # local to anchored reads so the rest of DFoT can still use cuDNN.
        attention_backends = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
        with sdpa_kernel(attention_backends):
            physical_read, _ = self.physical_read_attention(
                normalized_query,
                physical,
                physical,
                attn_mask=attention_bias,
                need_weights=False,
            )
            appearance_read, _ = self.appearance_read_attention(
                normalized_query,
                appearance,
                appearance,
                attn_mask=attention_bias,
                need_weights=False,
            )
        if noise_levels.shape != (batch_size, temporal_length):
            raise ValueError(
                "World-state logSNR shape mismatch: "
                f"expected {(batch_size, temporal_length)}, got {tuple(noise_levels.shape)}"
            )
        signal_gate = torch.sigmoid(
            noise_levels.float() * self.logsnr_scale
        ).to(query)
        signal_gate = rearrange(signal_gate, "b t -> (b t) 1 1")
        physical_noise_gate = 1.0 - 0.5 * signal_gate
        appearance_noise_gate = 0.1 + 0.9 * signal_gate
        visibility_flat = rearrange(visibility, "b t n c -> (b t) n c")
        query_tokens = height * width
        branch_mode = self._runtime_branch_mode
        if branch_mode == "adaptive":
            outputs = self._cached_preference_outputs
            if outputs is None:
                raise RuntimeError(
                    "Adaptive world-state read has no stable window policy action"
                )
            physical_clip_probability_fp32 = self._match_batch(
                outputs["physical_probability"][:, None],
                batch_size,
            )
            appearance_clip_probability_fp32 = self._match_batch(
                outputs["appearance_probability"][:, None],
                batch_size,
            )
            physical_clip_correction_fp32 = self._match_batch(
                outputs["physical_correction"][:, None],
                batch_size,
            )
            appearance_clip_correction_fp32 = self._match_batch(
                outputs["appearance_correction"][:, None],
                batch_size,
            )
        else:
            physical_clip_probability_fp32 = query.new_full(
                (batch_size, 1), 0.5, dtype=torch.float32
            )
            appearance_clip_probability_fp32 = query.new_full(
                (batch_size, 1), 0.5, dtype=torch.float32
            )
            physical_clip_correction_fp32 = torch.zeros_like(
                physical_clip_probability_fp32
            )
            appearance_clip_correction_fp32 = torch.zeros_like(
                appearance_clip_probability_fp32
            )
        physical_clip_probability = physical_clip_probability_fp32.to(query)
        appearance_clip_probability = appearance_clip_probability_fp32.to(query)
        physical_clip_correction = physical_clip_correction_fp32.to(query)
        appearance_clip_correction = appearance_clip_correction_fp32.to(query)
        physical_probability = physical_clip_probability[:, None].expand(
            -1, temporal_length * query_tokens, -1
        )
        appearance_probability = appearance_clip_probability[:, None].expand(
            -1, temporal_length * query_tokens, -1
        )
        physical_probability = rearrange(
            physical_probability,
            "b (t n) c -> (b t) n c",
            t=temporal_length,
            n=query_tokens,
        )
        appearance_probability = rearrange(
            appearance_probability,
            "b (t n) c -> (b t) n c",
            t=temporal_length,
            n=query_tokens,
        )
        physical_correction = physical_clip_correction[:, None].expand(
            -1, temporal_length * query_tokens, -1
        )
        appearance_correction = appearance_clip_correction[:, None].expand(
            -1, temporal_length * query_tokens, -1
        )
        physical_correction = rearrange(
            physical_correction,
            "b (t n) c -> (b t) n c",
            t=temporal_length,
            n=query_tokens,
        )
        appearance_correction = rearrange(
            appearance_correction,
            "b (t n) c -> (b t) n c",
            t=temporal_length,
            n=query_tokens,
        )
        physical_decay = torch.ones_like(physical_probability)
        appearance_decay = torch.ones_like(appearance_probability)
        if branch_mode == "adaptive":
            physical_multiplier = 1.0 + physical_correction
            appearance_multiplier = 1.0 + appearance_correction
        elif branch_mode == "full":
            physical_multiplier = torch.ones_like(physical_probability)
            appearance_multiplier = torch.ones_like(appearance_probability)
        elif branch_mode == "physical":
            physical_multiplier = torch.ones_like(physical_probability)
            appearance_multiplier = torch.zeros_like(appearance_probability)
        elif branch_mode == "appearance":
            physical_multiplier = torch.zeros_like(physical_probability)
            appearance_multiplier = torch.ones_like(appearance_probability)
        elif branch_mode == "none":
            physical_multiplier = torch.zeros_like(physical_probability)
            appearance_multiplier = torch.zeros_like(appearance_probability)
        else:
            raise RuntimeError(f"Invalid runtime world-state mode {branch_mode!r}")
        attended = (
            physical_multiplier * physical_noise_gate * physical_read
            + self.appearance_read_scale
            * appearance_multiplier
            * appearance_noise_gate
            * appearance_read
            * visibility_flat
        )
        if branch_mode == "none":
            raw_residual = torch.zeros_like(query)
        else:
            raw_residual = self.read_output(attended) + self.state_refiner(attended)
        learned_gate = (
            self.max_read_scale
            * torch.sigmoid(self.read_gate_logit)
            * self._runtime_scale
        )
        # Physical memory must remain available while future frames are still
        # highly noisy; that is when layout and motion are established.  The
        # per-branch gates above already delay private appearance information.
        # Multiplying the combined residual by signal_gate here would cancel
        # the intended high-noise physical path.
        raw_residual = raw_residual * learned_gate * visibility_flat
        output, transport_diagnostics = self._transport_residual(
            query,
            raw_residual,
            batch_size=batch_size,
            temporal_length=temporal_length,
            height=height,
            width=width,
        )
        residual = output - query
        object.__setattr__(
            self,
            "_last_gate_probabilities",
            {
                "physical": physical_clip_probability_fp32.squeeze(-1),
                "appearance": appearance_clip_probability_fp32.squeeze(-1),
            },
        )
        self.last_diagnostics = {
            "world_state_read_gate": learned_gate.detach().to(residual),
            "world_state_signal_gate": signal_gate.mean().detach(),
            "world_state_physical_noise_gate": physical_noise_gate.mean().detach(),
            "world_state_appearance_noise_gate": appearance_noise_gate.mean().detach(),
            "world_state_residual_rms": residual.float().square().mean().sqrt().detach(),
            "world_state_visibility_mean": visibility.mean().detach(),
            "world_state_visibility_min": visibility.amin().detach(),
            "world_state_confidence_mean": confidence.mean().detach(),
            "world_state_age_mean": age.mean().detach(),
            "world_state_age_max": age.amax().detach(),
            "world_state_window_count": window_count.mean().detach(),
            "world_state_physical_gate_probability": physical_clip_probability.mean().detach(),
            "world_state_appearance_gate_probability": appearance_clip_probability.mean().detach(),
            "world_state_physical_effective_probability": (
                physical_probability
            ).mean().detach(),
            "world_state_appearance_effective_probability": (
                appearance_probability
            ).mean().detach(),
            "world_state_physical_multiplier": physical_multiplier.mean().detach(),
            "world_state_appearance_multiplier": appearance_multiplier.mean().detach(),
            "world_state_physical_policy_correction": physical_correction.mean().detach(),
            "world_state_appearance_policy_correction": appearance_correction.mean().detach(),
            "world_state_physical_age_decay": physical_decay.mean().detach(),
            "world_state_appearance_age_decay": appearance_decay.mean().detach(),
            "world_state_physical_age_decay_strength": self._bounded_decay_strength(
                self.physical_age_decay_strength
            ).detach().to(residual),
            "world_state_appearance_age_decay_strength": self._bounded_decay_strength(
                self.appearance_age_decay_strength
            ).detach().to(residual),
            "world_state_branch_mode": residual.new_tensor(
                {
                    "none": 0.0,
                    "physical": 1.0,
                    "appearance": 2.0,
                    "full": 3.0,
                    "adaptive": 4.0,
                }[branch_mode]
            ),
            "world_state_compatibility_mean": compatibility_mean.detach(),
            "world_state_memory_tokens": residual.new_tensor(
                float(state["physical"].shape[1])
            ),
            **{
                name: value.detach().to(residual)
                for name, value in transport_diagnostics.items()
            },
        }
        self._record_runtime_diagnostics()
        return rearrange(
            output,
            "bt (h w) c -> bt c h w",
            h=height,
            w=width,
        )


class FrozenUViTTail(nn.Module):
    """Frozen U-ViT suffix starting after one captured encoder/decoder block."""

    def __init__(
        self,
        backbone: "UViT3DPose",
        *,
        activation_checkpointing: bool = True,
    ):
        super().__init__()
        self.activation_checkpointing = bool(activation_checkpointing)
        stage = backbone._latent_return_stage()
        if stage in {"down", "encoder"}:
            self.capture_stage = "down"
            capture_is_up = False
        elif stage in {"up", "decoder"}:
            self.capture_stage = "up"
            capture_is_up = True
        else:
            raise ValueError(
                "FrozenUViTTail supports encoder/down or decoder/up cuts; "
                f"got {stage!r}."
            )
        self.temporal_length = int(backbone.temporal_length)
        self.num_levels = int(backbone.num_levels)
        self.capture_level = int(backbone._latent_return_level())

        level_blocks = backbone._level_blocks(
            self.capture_level,
            is_up=capture_is_up,
        )
        capture_indices = sorted(
            backbone._normalize_capture_indices(
                backbone._latent_return_blocks(),
                len(level_blocks),
            )
        )
        if len(capture_indices) != 1:
            raise ValueError(
                "UPCAST frozen-tail supervision requires exactly "
                f"one captured block; got {capture_indices}."
            )
        self.capture_block_index = int(capture_indices[0])
        self.current_blocks = copy.deepcopy(
            nn.ModuleList(list(level_blocks[self.capture_block_index + 1 :]))
        )

        self.current_up_index = None
        self.current_downsample = None
        self.current_downsample_index = None
        self.remaining_down_levels: tuple[int, ...] = ()
        self.remaining_down_blocks = nn.ModuleList()
        self.mid_blocks = nn.ModuleList()
        if self.capture_stage == "up":
            self.current_up_index = self.num_levels - 2 - self.capture_level
            if not 0 <= self.current_up_index < len(backbone.up_blocks):
                raise ValueError(
                    f"Invalid up capture level {self.capture_level} for "
                    f"{len(backbone.up_blocks)} up levels."
                )
            remaining_up_indices = list(
                range(self.current_up_index + 1, len(backbone.up_blocks))
            )
        else:
            if not 0 <= self.capture_level < len(backbone.down_blocks):
                raise ValueError(
                    f"Invalid down capture level {self.capture_level} for "
                    f"{len(backbone.down_blocks)} down levels."
                )
            self.current_downsample_index = (
                len(backbone.down_blocks[self.capture_level]) - 1
            )
            self.current_downsample = copy.deepcopy(
                backbone.down_blocks[self.capture_level][
                    self.current_downsample_index
                ]
            )
            self.remaining_down_levels = tuple(
                range(self.capture_level + 1, len(backbone.down_blocks))
            )
            self.remaining_down_blocks = copy.deepcopy(
                nn.ModuleList(
                    [backbone.down_blocks[level] for level in self.remaining_down_levels]
                )
            )
            self.mid_blocks = copy.deepcopy(backbone.mid_blocks)
            remaining_up_indices = list(range(len(backbone.up_blocks)))

        self.remaining_up_indices = tuple(remaining_up_indices)
        self.remaining_levels = tuple(
            self.num_levels - 2 - up_index for up_index in remaining_up_indices
        )
        self.remaining_up_blocks = copy.deepcopy(
            nn.ModuleList([backbone.up_blocks[index] for index in remaining_up_indices])
        )
        self.project_output = copy.deepcopy(backbone.project_output)
        self.is_transformers = tuple(bool(value) for value in backbone.is_transformers)
        self.pos_emb_type = str(backbone.pos_emb_type)
        self.pos_embs = nn.ModuleDict()
        if self.pos_emb_type == "learned_1d":
            required_levels = {
                self.capture_level,
                *self.remaining_down_levels,
                *self.remaining_levels,
            }
            if self.capture_stage == "down":
                required_levels.add(self.num_levels - 1)
            for level in required_levels:
                self.pos_embs[str(level)] = copy.deepcopy(backbone.pos_embs[str(level)])
        self.eval()
        self.requires_grad_(False)

    def _run_module(self, module: nn.Module, *args: Tensor) -> Tensor:
        should_checkpoint = (
            self.activation_checkpointing
            and torch.is_grad_enabled()
            and any(value.requires_grad for value in args)
        )
        if should_checkpoint:
            return checkpoint(
                module,
                *args,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return module(*args)

    @staticmethod
    def _load_source_submodule(
        module: nn.Module,
        source_state: dict[str, Tensor],
        prefix: str,
    ) -> None:
        prefix_with_dot = prefix + "."
        state = {
            key[len(prefix_with_dot) :]: value
            for key, value in source_state.items()
            if key.startswith(prefix_with_dot)
        }
        expected = module.state_dict()
        if expected and not state:
            raise KeyError(f"Frozen-tail source checkpoint is missing {prefix}.*")
        module.load_state_dict(state, strict=True)

    def load_from_backbone_state_dict(self, source_state: dict[str, Tensor]) -> None:
        """Load this suffix from an uncompiled U-ViT backbone state dict."""
        if self.capture_stage == "up":
            current_prefix = f"up_blocks.{self.current_up_index}"
            # U-ViT up level index 0 is the upsample; _level_blocks omits it.
            current_start = self.capture_block_index + 2
        else:
            current_prefix = f"down_blocks.{self.capture_level}"
            current_start = self.capture_block_index + 1
        for offset, block in enumerate(self.current_blocks):
            self._load_source_submodule(
                block,
                source_state,
                f"{current_prefix}.{current_start + offset}",
            )

        if self.capture_stage == "down":
            self._load_source_submodule(
                self.current_downsample,
                source_state,
                f"down_blocks.{self.capture_level}.{self.current_downsample_index}",
            )
            for level, down_block in zip(
                self.remaining_down_levels,
                self.remaining_down_blocks,
            ):
                self._load_source_submodule(
                    down_block,
                    source_state,
                    f"down_blocks.{level}",
                )
            self._load_source_submodule(
                self.mid_blocks,
                source_state,
                "mid_blocks",
            )

        for up_index, up_block in zip(
            self.remaining_up_indices,
            self.remaining_up_blocks,
        ):
            self._load_source_submodule(
                up_block,
                source_state,
                f"up_blocks.{up_index}",
            )
        self._load_source_submodule(
            self.project_output,
            source_state,
            "project_output",
        )
        for level, positional_embedding in self.pos_embs.items():
            self._load_source_submodule(
                positional_embedding,
                source_state,
                f"pos_embs.{level}",
            )
        self.eval()
        self.requires_grad_(False)

    def _run_blocks(
        self,
        x: Tensor,
        emb: Tensor,
        level: int,
        blocks: nn.ModuleList,
        *,
        add_position: bool,
    ) -> Tensor:
        is_transformer = self.is_transformers[level]
        if is_transformer:
            height, width = x.shape[-2:]
            x = rearrange(
                x,
                "(b t) c h w -> b (t h w) c",
                t=self.temporal_length,
            )
            emb = rearrange(
                emb,
                "(b t) c h w -> b (t h w) c",
                t=self.temporal_length,
            )
            # A captured transformer latent already contains its learned
            # positional embedding. Subsequent levels have not seen theirs yet.
            if add_position and self.pos_emb_type == "learned_1d":
                x = self.pos_embs[str(level)](x)
        for block in blocks:
            x = self._run_module(block, x, emb)
        if is_transformer:
            x = rearrange(
                x,
                "b (t h w) c -> (b t) c h w",
                t=self.temporal_length,
                h=height,
                w=width,
            )
        return x

    def _run_up_path(
        self,
        x: Tensor,
        embs: list[Tensor],
        hs_before: list[Tensor],
        hs_after: list[Tensor],
    ) -> Tensor:
        for level, up_block in zip(self.remaining_levels, self.remaining_up_blocks):
            x = x - hs_after.pop()
            x = self._run_module(up_block[0], x) + hs_before.pop()
            x = self._run_blocks(
                x,
                embs[level],
                level,
                up_block[1:],
                add_position=True,
            )
        return x

    def forward(self, replacement_latent: Tensor, tail_state: dict[str, Any]) -> Tensor:
        state_stage = str(tail_state.get("stage", "up" if tail_state.get("is_up") else "down"))
        if state_stage != self.capture_stage:
            raise ValueError(
                f"Frozen tail stage mismatch: state={state_stage!r}, "
                f"tail={self.capture_stage!r}."
            )
        if int(tail_state["i_level"]) != self.capture_level:
            raise ValueError(
                "Frozen tail capture-level mismatch: "
                f"state={tail_state['i_level']}, tail={self.capture_level}."
            )
        if int(tail_state["next_block_idx"]) != self.capture_block_index + 1:
            raise ValueError(
                "Frozen tail block mismatch: "
                f"state starts at {tail_state['next_block_idx']}, "
                f"tail starts at {self.capture_block_index + 1}."
            )

        x = rearrange(replacement_latent, "b t c h w -> (b t) c h w")
        embs = tail_state["embs"]
        hs_before = list(tail_state["hs_before"])
        hs_after = list(tail_state["hs_after"])
        x = self._run_blocks(
            x,
            embs[self.capture_level],
            self.capture_level,
            self.current_blocks,
            add_position=False,
        )

        if self.capture_stage == "up":
            state_up_index = tail_state.get("current_up_index")
            if state_up_index is None or int(state_up_index) != self.current_up_index:
                raise ValueError(
                    "Frozen tail up-index mismatch: "
                    f"state={state_up_index}, tail={self.current_up_index}."
                )
        else:
            hs_before.append(x)
            x = self._run_module(self.current_downsample, x)
            hs_after.append(x)
            for level, down_block in zip(
                self.remaining_down_levels,
                self.remaining_down_blocks,
            ):
                x = self._run_blocks(
                    x,
                    embs[level],
                    level,
                    down_block[:-1],
                    add_position=True,
                )
                hs_before.append(x)
                x = self._run_module(down_block[-1], x)
                hs_after.append(x)
            middle_level = self.num_levels - 1
            x = self._run_blocks(
                x,
                embs[middle_level],
                middle_level,
                self.mid_blocks,
                add_position=True,
            )

        x = self._run_up_path(x, embs, hs_before, hs_after)
        if hs_before or hs_after:
            raise RuntimeError(
                "Frozen U-ViT tail did not consume all skip tensors: "
                f"before={len(hs_before)}, after={len(hs_after)}"
            )
        x = self._run_module(self.project_output, x)
        return rearrange(
            x,
            "(b t) c h w -> b t c h w",
            t=self.temporal_length,
        )


class UViT3DPose(UViT3D):
    """
    U-ViT with pose embedding.
    """

    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        max_tokens: int,
        external_cond_dim: int,
        use_causal_mask=True,
    ):
        self.conditioning_dropout = cfg.external_cond_dropout
        super().__init__(
            cfg,
            x_shape,
            max_tokens,
            cfg.conditioning.dim,
            use_causal_mask,
        )
        world_state_cfg = getattr(cfg, "world_state", None)
        self.world_state_enabled = bool(
            world_state_cfg is not None
            and getattr(world_state_cfg, "enable", False)
        )
        self.world_state_inject_level = int(
            getattr(world_state_cfg, "inject_level", 2)
            if world_state_cfg is not None
            else 2
        )
        if self.world_state_enabled:
            if not 0 <= self.world_state_inject_level < self.num_levels - 1:
                raise ValueError(
                    "world_state.inject_level must select an up/down level, got "
                    f"{self.world_state_inject_level}"
                )
            self.world_state_adapter = GatedWorldStateAdapter(
                hidden_dim=int(cfg.channels[self.world_state_inject_level]),
                state_dim=int(getattr(world_state_cfg, "dim", 768)),
                num_heads=int(getattr(world_state_cfg, "num_heads", 12)),
                max_read_scale=float(
                    getattr(world_state_cfg, "max_read_scale", 0.15)
                ),
                appearance_read_scale=float(
                    getattr(world_state_cfg, "appearance_read_scale", 0.25)
                ),
                physical_update_rate=float(
                    getattr(world_state_cfg, "physical_update_rate", 0.5)
                ),
                appearance_update_rate=float(
                    getattr(world_state_cfg, "appearance_update_rate", 0.1)
                ),
                logsnr_scale=float(
                    getattr(world_state_cfg, "logsnr_scale", 8.0)
                ),
                anchor_dim=int(getattr(world_state_cfg, "anchor_dim", 9)),
                memory_grid_size=int(
                    getattr(world_state_cfg, "memory_grid_size", 4)
                ),
                max_state_tokens=int(
                    getattr(world_state_cfg, "max_state_tokens", 160)
                ),
                visibility_threshold=float(
                    getattr(world_state_cfg, "visibility_threshold", 0.35)
                ),
                visibility_temperature=float(
                    getattr(world_state_cfg, "visibility_temperature", 0.10)
                ),
                preference_gate_hidden_dim=int(
                    getattr(world_state_cfg, "preference_gate_hidden_dim", 64)
                ),
                preference_policy_version=int(
                    getattr(
                        world_state_cfg,
                        "preference_policy_version",
                        PREFERENCE_POLICY_VERSION,
                    )
                ),
                preference_content_dim=int(
                    getattr(world_state_cfg, "preference_content_dim", 32)
                ),
                preference_max_correction=float(
                    getattr(world_state_cfg, "preference_max_correction", 0.5)
                ),
                physical_age_half_life=float(
                    getattr(world_state_cfg, "physical_age_half_life", 8.0)
                ),
                appearance_age_half_life=float(
                    getattr(world_state_cfg, "appearance_age_half_life", 1.5)
                ),
                transport_mode=str(
                    getattr(world_state_cfg, "transport_mode", "additive")
                ),
                transport_max_residual_ratio=float(
                    getattr(
                        world_state_cfg,
                        "transport_max_residual_ratio",
                        0.006,
                    )
                ),
                transport_max_relation_drift=float(
                    getattr(
                        world_state_cfg,
                        "transport_max_relation_drift",
                        0.0001,
                    )
                ),
                transport_min_scale=float(
                    getattr(world_state_cfg, "transport_min_scale", 0.0)
                ),
                transport_preserve_token_norm=bool(
                    getattr(
                        world_state_cfg,
                        "transport_preserve_token_norm",
                        True,
                    )
                ),
            )
        else:
            self.world_state_adapter = None

        generative_cfg = getattr(cfg, "generative_unit", None)
        self.generative_unit_enabled = bool(
            generative_cfg is not None
            and getattr(generative_cfg, "enable", False)
        )
        if self.generative_unit_enabled:
            capture_stage = self._latent_return_stage()
            capture_level = self._latent_return_level()
            if capture_stage not in {"down", "encoder"}:
                raise ValueError(
                    "Physical conditioner requires latent_return_stage=down so its "
                    "read and injection distributions match"
                )
            if not 0 <= capture_level < self.num_levels - 1:
                raise ValueError(
                    "Physical conditioner requires a down-path capture level, got "
                    f"{capture_level}"
                )
            self.generative_unit_conditioner = PhysicalTransitionConditioner(
                hidden_dim=int(cfg.channels[capture_level]),
                physical_dim=int(
                    getattr(generative_cfg, "physical_dim", 64)
                ),
                num_heads=int(getattr(generative_cfg, "num_heads", 8)),
                spatial_grid=int(
                    getattr(generative_cfg, "spatial_grid", 8)
                ),
                max_read_scale=float(
                    getattr(generative_cfg, "max_read_scale", 0.05)
                ),
                local_read_fraction=float(
                    getattr(generative_cfg, "local_read_fraction", 0.5)
                ),
                local_read_radius=int(
                    getattr(generative_cfg, "local_read_radius", 1)
                ),
                pair_encoder_layers=int(
                    getattr(generative_cfg, "pair_encoder_layers", 2)
                ),
                pair_gate_prior=float(
                    getattr(generative_cfg, "pair_gate_prior", 0.8)
                ),
                pair_update_scale=float(
                    getattr(generative_cfg, "pair_update_scale", 0.25)
                ),
            )
        else:
            self.generative_unit_conditioner = None

    def set_world_state_condition(
        self,
        state: dict[str, Tensor] | None,
        *,
        query_anchors: Tensor | None = None,
        scale: float = 1.0,
        detach: bool = True,
        branch_mode: str = "adaptive",
    ) -> None:
        if self.world_state_adapter is None:
            if state is not None:
                raise RuntimeError("World-state conditioning is disabled in the backbone")
            return
        self.world_state_adapter.set_runtime_state(
            state,
            query_anchors=query_anchors,
            scale=scale,
            detach=detach,
            branch_mode=branch_mode,
        )

    def clear_world_state_condition(self) -> None:
        if self.world_state_adapter is not None:
            self.world_state_adapter.clear_runtime_state()

    def update_world_state(
        self,
        physical_observation: Tensor,
        appearance_observation: Tensor,
        previous_state: dict[str, Tensor] | None = None,
        *,
        anchors: Tensor | None = None,
        confidence: Tensor | None = None,
        detach_previous: bool = True,
    ) -> dict[str, Tensor]:
        if self.world_state_adapter is None:
            raise RuntimeError("World-state conditioning is disabled in the backbone")
        return self.world_state_adapter.update_state(
            physical_observation,
            appearance_observation,
            previous_state,
            anchors=anchors,
            confidence=confidence,
            detach_previous=detach_previous,
        )

    def advance_world_state_age(
        self,
        previous_state: dict[str, Tensor] | None,
    ) -> dict[str, Tensor] | None:
        if self.world_state_adapter is None:
            raise RuntimeError("World-state conditioning is disabled in the backbone")
        return self.world_state_adapter.advance_state_age(previous_state)

    @property
    def world_state_gate_probabilities(self) -> dict[str, Tensor]:
        if self.world_state_adapter is None:
            return {}
        return self.world_state_adapter.gate_probabilities

    @property
    def world_state_diagnostics(self) -> dict[str, Tensor]:
        if self.world_state_adapter is None:
            return {}
        return self.world_state_adapter.last_diagnostics

    @property
    def world_state_window_diagnostics(self) -> dict[str, Tensor]:
        if self.world_state_adapter is None:
            return {}
        return self.world_state_adapter.window_diagnostics

    @property
    def generative_unit_diagnostics(self) -> dict[str, Tensor]:
        if self.generative_unit_conditioner is None:
            return {}
        return self.generative_unit_conditioner.last_diagnostics

    def condition_generative_unit(
        self,
        latent: Tensor,
        noisy_physical: Tensor,
        future_indices: Tensor,
        physical_logsnr: Tensor,
        *,
        video_physical: Tensor,
        fused_physical: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.generative_unit_conditioner is None:
            raise RuntimeError("Physical conditioner conditioning is disabled")
        return self.generative_unit_conditioner(
            latent,
            noisy_physical,
            future_indices,
            physical_logsnr,
            video_physical=video_physical,
            fused_physical=fused_physical,
        )

    def _build_external_cond_embedding(self) -> Optional[nn.Module]:
        return RandomDropoutPatchEmbed(
            dropout_prob=self.conditioning_dropout,
            img_size=self.x_shape[1],
            patch_size=self.cfg.patch_size,
            in_chans=self.external_cond_dim,
            embed_dim=self.external_cond_emb_dim,
            bias=True,
            flatten=False,
        )

    def _latent_return_mode(self) -> str:
        return str(getattr(self.cfg, "latent_return_mode", "multi_scale") or "multi_scale").lower()

    def _latent_return_level(self) -> int:
        return int(getattr(self.cfg, "latent_return_level", 2))

    def _latent_return_stage(self) -> str:
        return str(getattr(self.cfg, "latent_return_stage", "up") or "up").lower()

    def _latent_return_blocks(self) -> list[int]:
        raw = getattr(self.cfg, "latent_return_blocks", [-2])
        return [int(x) for x in raw]

    def _rearrange_and_add_pos_emb_if_transformer(
        self, x: Tensor, emb: Tensor, i_level: int
    ) -> Tuple[Tensor, Tensor]:
        is_transformer = self.is_transformers[i_level]
        if not is_transformer:
            return x, emb
        x, emb = map(
            lambda y: rearrange(
                y, "(b t) c h w -> b (t h w) c", t=self.temporal_length
            ),
            (x, emb),
        )
        if self.pos_emb_type == "learned_1d":
            x = self.pos_embs[f"{i_level}"](x)
        return x, emb

    def _run_level_from_block(
        self,
        x: Tensor,
        emb: Tensor,
        i_level: int,
        is_up: bool,
        start_block_idx: int,
    ) -> Tensor:
        is_transformer = self.is_transformers[i_level]
        if is_transformer:
            x, emb = map(
                lambda value: rearrange(
                    value,
                    "(b t) c h w -> b (t h w) c",
                    t=self.temporal_length,
                ),
                (x, emb),
            )
        blocks = self._level_blocks(i_level, is_up)
        use_checkpointing = self.use_checkpointing[i_level]
        for block in blocks[int(start_block_idx) :]:
            x = self._checkpointed_forward(
                block,
                x,
                emb,
                use_checkpointing=use_checkpointing,
            )
        return self._unrearrange_if_transformer(x, i_level)

    def build_frozen_tail(
        self,
        *,
        activation_checkpointing: bool = True,
    ) -> FrozenUViTTail:
        return FrozenUViTTail(
            self,
            activation_checkpointing=activation_checkpointing,
        )

    def _run_level_with_tail_capture(
        self,
        x: Tensor,
        emb: Tensor,
        i_level: int,
        is_up: bool,
        capture_block_indices: list[int],
        captured_latents: list[Tensor],
        embs: list[Tensor],
        hs_before: list[Tensor],
        hs_after: list[Tensor],
        current_up_index: int | None,
    ) -> tuple[Tensor, dict[str, Any] | None]:
        x, emb = self._rearrange_and_add_pos_emb_if_transformer(x, emb, i_level)
        blocks = self._level_blocks(i_level, is_up)
        capture_set = self._normalize_capture_indices(capture_block_indices, len(blocks))
        use_checkpointing = self.use_checkpointing[i_level]
        tail_state = None

        for block_idx, block in enumerate(blocks):
            x = self._checkpointed_forward(
                block,
                x,
                emb,
                use_checkpointing=use_checkpointing,
            )
            if block_idx in capture_set:
                latent = self._unrearrange_if_transformer(x, i_level)
                captured_latents.append(latent)
                tail_state = {
                    "stage": "up" if is_up else ("mid" if i_level == self.num_levels - 1 else "down"),
                    "i_level": int(i_level),
                    "is_up": bool(is_up),
                    "next_block_idx": int(block_idx + 1),
                    "current_up_index": None if current_up_index is None else int(current_up_index),
                    "embs": embs,
                    "hs_before": list(hs_before),
                    "hs_after": list(hs_after),
                }

        x = self._unrearrange_if_transformer(x, i_level)
        return x, tail_state

    def forward_tail_from_state(
        self,
        replacement_latent: Tensor,
        tail_state: dict[str, Any],
    ) -> Tensor:
        """Continue the live U-ViT suffix from a captured block latent.

        Both encoder and decoder cuts are supported.  This is the trainable
        counterpart of :class:`FrozenUViTTail` and lets Physical conditioner reuse
        the exact prefix activations and skip tensors from the base pass.
        """
        x = rearrange(replacement_latent, "b t c h w -> (b t) c h w")
        embs = tail_state["embs"]
        hs_before = list(tail_state["hs_before"])
        hs_after = list(tail_state["hs_after"])
        capture_level = int(tail_state["i_level"])
        is_up = bool(tail_state.get("is_up", False))
        x = self._run_level_from_block(
            x,
            embs[capture_level],
            i_level=capture_level,
            is_up=is_up,
            start_block_idx=int(tail_state["next_block_idx"]),
        )

        if is_up:
            current_up_index = tail_state.get("current_up_index", None)
            if current_up_index is None:
                raise RuntimeError("Missing current_up_index in U-ViT tail_state")
            first_up_index = int(current_up_index) + 1
        else:
            if not 0 <= capture_level < len(self.down_blocks):
                raise ValueError(
                    f"Invalid down-path tail capture level {capture_level}"
                )
            hs_before.append(x)
            x = self.down_blocks[capture_level][-1](x)
            hs_after.append(x)
            for level in range(capture_level + 1, len(self.down_blocks)):
                down_block = self.down_blocks[level]
                x = self._run_level(x, embs[level], level, is_up=False)
                hs_before.append(x)
                x = down_block[-1](x)
                hs_after.append(x)
            x = self._run_level(
                x,
                embs[-1],
                self.num_levels - 1,
                is_up=False,
            )
            first_up_index = 0

        for up_index in range(first_up_index, len(self.up_blocks)):
            up_block = self.up_blocks[up_index]
            level = self.num_levels - 2 - up_index
            x = x - hs_after.pop()
            x = up_block[0](x) + hs_before.pop()
            x = self._run_level(x, embs[level], level, is_up=True)

        if hs_before or hs_after:
            raise RuntimeError(
                "Live U-ViT tail did not consume all skip tensors: "
                f"before={len(hs_before)}, after={len(hs_after)}"
            )
        x = self.project_output(x)
        return rearrange(
            x,
            "(b t) c h w -> b t c h w",
            t=self.temporal_length,
        )

    def forward(
        self,
        x: Tensor,
        noise_levels: Tensor,
        external_cond: Optional[Tensor] = None,
        external_cond_mask: Optional[Tensor] = None,
        return_latents: Optional[bool] = False,
        return_tail_state: Optional[bool] = False,
    ) -> Tensor:
        """
        Forward pass of the U-ViT backbone, with pose conditioning.
        Args:
            x: Input tensor of shape (B, T, C, H, W).
            noise_levels: Noise level tensor of shape (B, T).
            external_cond: External conditioning tensor of shape (B, T, C', H, W).
        Returns:
            Output tensor of shape (B, T, C, H, W).
        """
        # import pdb; pdb.set_trace()
        assert (
            x.shape[1] == self.temporal_length
        ), f"Temporal length of U-ViT is set to {self.temporal_length}, but input has temporal length {x.shape[1]}."

        assert (
            external_cond is not None
        ), "External condition (camera pose) is required for U-ViT3DPose model."

        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.embed_input(x)

        # Embeddings
        external_cond = self.external_cond_embedding(external_cond, external_cond_mask)
        emb = self.noise_level_pos_embedding(noise_levels)
        emb = rearrange(
            rearrange(emb, "b t c -> b t c 1 1") + external_cond,
            "b t c h w -> (b t) c h w",
        )

        # Down-sample embeddings for each level
        embs = [
            (
                emb
                if i_level == 0
                # pylint: disable-next=not-callable
                else F.avg_pool2d(emb, kernel_size=2**i_level, stride=2**i_level)
            )
            for i_level in range(self.num_levels)
        ]
        hs_before = []  # hidden states before downsampling
        hs_after = []  # hidden states after downsampling
        # print(f"original shape x: {x.shape}") #  x: torch.Size([64, 128, 128, 128])
        # Down-sampling blocks
        if hasattr(self, "latents_savename") and self.latents_savename is not None:
            original_latents = return_latents
            return_latents = True
            # print(f"[Info] set return_latents to True for latents saving at {self.latents_savename}.")
        latent_return_mode = self._latent_return_mode()
        capture_same_level_blocks = return_latents and latent_return_mode in {
            "same_level_blocks",
            "same_level",
            "block",
            "blocks",
        }
        capture_level = self._latent_return_level()
        capture_stage = self._latent_return_stage()
        capture_blocks = self._latent_return_blocks()
        if return_latents:
            latents_list = []
        tail_state = None
        for i_level, down_block in enumerate(
            self.down_blocks,
        ):
            capture_here = (
                capture_same_level_blocks
                and capture_stage in {"down", "encoder"}
                and i_level == capture_level
            )
            if capture_here and return_tail_state:
                x, maybe_tail_state = self._run_level_with_tail_capture(
                    x,
                    embs[i_level],
                    i_level,
                    is_up=False,
                    capture_block_indices=capture_blocks,
                    captured_latents=latents_list,
                    embs=embs,
                    hs_before=hs_before,
                    hs_after=hs_after,
                    current_up_index=None,
                )
                tail_state = maybe_tail_state or tail_state
            else:
                x = self._run_level(
                    x,
                    embs[i_level],
                    i_level,
                    capture_block_indices=capture_blocks if capture_here else None,
                    captured_latents=latents_list if capture_here else None,
                )
            # print(f"down sample i_level: {i_level}, x.shape: {x.shape}")
            hs_before.append(x)
            if return_latents and not capture_same_level_blocks:
                latents_list.append(x)
            x = down_block[-1](x)
            hs_after.append(x)
        # Middle blocks
        capture_here = (
            capture_same_level_blocks
            and capture_stage in {"mid", "middle", "bottleneck"}
            and self.num_levels - 1 == capture_level
        )
        if capture_here and return_tail_state:
            x, maybe_tail_state = self._run_level_with_tail_capture(
                x,
                embs[-1],
                self.num_levels - 1,
                is_up=False,
                capture_block_indices=capture_blocks,
                captured_latents=latents_list,
                embs=embs,
                hs_before=hs_before,
                hs_after=hs_after,
                current_up_index=None,
            )
            tail_state = maybe_tail_state or tail_state
        else:
            x = self._run_level(
                x,
                embs[-1],
                self.num_levels - 1,
                capture_block_indices=capture_blocks if capture_here else None,
                captured_latents=latents_list if capture_here else None,
            )
        if return_latents and not capture_same_level_blocks:
            latents_list.append(x)
        # import pdb; pdb.set_trace()
        # Up-sampling blocks
        for _i_level, up_block in enumerate(self.up_blocks):
            i_level = self.num_levels - 2 - _i_level
            x = x - hs_after.pop()
            x = up_block[0](x) + hs_before.pop()
            capture_here = (
                capture_same_level_blocks
                and capture_stage in {"up", "decoder"}
                and i_level == capture_level
            )
            if capture_here and return_tail_state:
                x, maybe_tail_state = self._run_level_with_tail_capture(
                    x,
                    embs[i_level],
                    i_level,
                    is_up=True,
                    capture_block_indices=capture_blocks,
                    captured_latents=latents_list,
                    embs=embs,
                    hs_before=hs_before,
                    hs_after=hs_after,
                    current_up_index=_i_level,
                )
                tail_state = maybe_tail_state or tail_state
            else:
                x = self._run_level(
                    x,
                    embs[i_level],
                    i_level,
                    is_up=True,
                    capture_block_indices=capture_blocks if capture_here else None,
                    captured_latents=latents_list if capture_here else None,
                )
            if (
                self.world_state_adapter is not None
                and i_level == self.world_state_inject_level
            ):
                x = self.world_state_adapter(
                    x,
                    noise_levels,
                    self.temporal_length,
                )
            if return_latents and not capture_same_level_blocks:
                latents_list.append(x)
        if hasattr(self, "latents_savename") and self.latents_savename is not None:
            self.latents_savename_hit_counter+=1
            if self.latents_savename_hit_counter >= self.save_at_hit_n:
                latent_to_save = latents_list[self.layer_to_save]
                torch.save(latent_to_save, self.latents_savename)
                print(f"[Info] Latents of shape {latent_to_save.shape} saved to {self.latents_savename} saved at hit counter {self.latents_savename_hit_counter}.")
                self.latents_savename = None  # reset to avoid saving multiple time
                self.layer_to_save = None
            if original_latents is False:
                del latents_list
                return_latents = original_latents  # reset to original value
        x = self.project_output(x)
        model_output=rearrange(x, "(b t) c h w -> b t c h w", t=self.temporal_length)
        if return_latents:
            if capture_same_level_blocks and len(latents_list) == 0:
                raise RuntimeError(
                    "latent_return_mode=same_level_blocks did not capture any latent. "
                    f"Check latent_return_stage={capture_stage!r}, "
                    f"latent_return_level={capture_level}, "
                    f"latent_return_blocks={capture_blocks}."
                )
            latents_list = [rearrange(latent, "(b t) c h w -> b t c h w", t=self.temporal_length) for latent in latents_list]
            if return_tail_state:
                return model_output, latents_list, tail_state
            return model_output, latents_list
        else:
            return model_output
    # add a latents savename set function, no relation with other algorithm but visualization
    def set_latents_savename(self, savename: str , save_at_hit_n: int = 49, layer_to_save=-1):
        """
        Set the savename for the latents.
        Args:
            savename (str): The name to save the latents.
        """
        try:
            os.makedirs(os.path.dirname(savename), exist_ok=True)
        except Exception as e:
            print(f"[Warning] Failed to create directory for latents: {e}")
        self.latents_savename_hit_counter = 0
        self.save_at_hit_n = save_at_hit_n
        self.latents_savename = savename
        self.layer_to_save = layer_to_save  # for future use if needed
