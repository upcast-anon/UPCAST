"""Factorized physical and appearance transition learning for UPCAST.

The module learns a PAGE-anchored discrete physical transition together with a
continuous private appearance transition. The adapter exposes the frozen video-side
transition encoder to a pose-aware recurrent adapter; PAGE and DINO remain
training-only teachers.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from upcast.geometry_teacher import GeometryTeacherOutput, build_geometry_teacher
from upcast.factorized_transitions import (
    ConditionalGeometryDecoder,
    ConditionalVisualDecoder,
    DenseFeaturePairProjector,
    GeometryAnchorEncoder,
    GeometryPairProjector,
    ResidualTokenAdapter,
    SpatialFusionEncoder,
    SpatialResidualVQ,
    SpatialTransitionEncoder,
    VideoPairProjector,
    VisualAnchorEncoder,
    WorldTokenFusion,
    gather_frames,
    resize_map,
)


UNIT_ARCH_VERSION = 7


@dataclass
class PreparedUnitTeacherContext:
    teacher: GeometryTeacherOutput
    current: Tensor
    future: Tensor
    delta: Tensor
    pair_dino: Tensor
    current_page: Tensor
    current_points: Tensor


@dataclass
class PreparedPhysicalTeacherContext:
    teacher: GeometryTeacherOutput
    current: Tensor
    future: Tensor
    delta: Tensor


@dataclass
class GenerativePhysicalTokens:
    video: Tensor
    fused: Tensor
    current: Tensor
    future: Tensor
    delta: Tensor
    future_logsnr: Tensor
    sample_mask: Tensor
    video_indices: Tensor
    fused_indices: Tensor


@dataclass
class PreparedGenerationTargets:
    dino: Tensor | None
    vjepa: Tensor | None
    num_frames: int


@dataclass
class PreparedPreferenceTargets:
    dino: Tensor
    vjepa: Tensor
    num_frames: int


def _gather_bt(x: Tensor, indices: Tensor) -> Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected [B,T], got {tuple(x.shape)}")
    return x.gather(1, indices[:, None]).squeeze(1)


def _weighted_batch_mean(values: Tensor, weights: Tensor | None = None) -> Tensor:
    if values.ndim == 0:
        return values
    if values.ndim > 1:
        values = values.flatten(1).mean(dim=1)
    if weights is None:
        return values.mean()
    weights = weights.to(device=values.device, dtype=values.dtype)
    if weights.shape != values.shape:
        raise ValueError(f"Weight shape {tuple(weights.shape)} != loss shape {tuple(values.shape)}")
    # Divide by batch size, rather than sum(weights), so low-SNR samples really
    # contribute less instead of merely changing the relative normalization.
    return (values * weights).mean()


def _masked_batch_mean(values: Tensor, mask: Tensor | None = None) -> Tensor:
    """Average over accepted samples without shrinking the whole objective."""
    if values.ndim == 0:
        return values
    if values.ndim > 1:
        values = values.flatten(1).mean(dim=1)
    if mask is None:
        return values.mean()
    mask = mask.to(device=values.device, dtype=values.dtype)
    if mask.shape != values.shape:
        raise ValueError(f"Mask shape {tuple(mask.shape)} != loss shape {tuple(values.shape)}")
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _detach_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {key: _detach_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    return value


class FrozenDINOv2PatchExtractor(nn.Module):
    """Frozen DINOv2 patch-token extractor preserving the spatial grid."""

    def __init__(self, model_name: str = "dinov2_vits14", output_size: tuple[int, int] = (16, 16)):
        super().__init__()
        self.output_size = tuple(int(value) for value in output_size)
        cached_repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        if cached_repo.exists():
            self.model = torch.hub.load(str(cached_repo), model_name, source="local", verbose=False)
        else:
            self.model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
        self.output_dim = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }.get(model_name, 384)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.model.eval()
        self.model.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def forward(self, images: Tensor, allow_input_grad: bool = False) -> Tensor:
        batch_size, num_frames = images.shape[:2]
        x = rearrange(images, "b t c h w -> (b t) c h w").float().clamp(0.0, 1.0)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean.to(x)) / self.std.to(x)
        track_input_gradient = bool(allow_input_grad and images.requires_grad)
        with torch.set_grad_enabled(track_input_gradient):
            if track_input_gradient:
                output = activation_checkpoint(
                    self.model.forward_features,
                    x,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                output = self.model.forward_features(x)
        tokens = output.get("x_norm_patchtokens") if isinstance(output, dict) else output
        if tokens is None and isinstance(output, dict):
            tokens = output.get("x_prenorm")
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]
        if tokens is None or tokens.ndim != 3:
            raise RuntimeError(f"Unexpected DINOv2 output: {type(tokens)!r}")
        side = int(tokens.shape[1] ** 0.5)
        if side * side != tokens.shape[1]:
            raise RuntimeError(f"Cannot reshape {tokens.shape[1]} DINO patch tokens to a square grid")
        features = rearrange(tokens, "n (h w) c -> n c h w", h=side, w=side)
        features = resize_map(features, self.output_size)
        return rearrange(features, "(b t) c h w -> b t c h w", b=batch_size, t=num_frames)


class FrozenVJEPATokenExtractor(nn.Module):
    """Frozen V-JEPA encoder returning [B,T',H',W',D] tubelet tokens."""

    CHECKPOINTS = {
        "vjepa_vitl16": ("vit_large", "vitl16.pth.tar", 224),
        "vjepa_vith16": ("vit_huge", "vith16.pth.tar", 224),
        "vjepa_vith16_384": ("vit_huge", "vith16-384.pth.tar", 384),
    }

    def __init__(self, model_name: str = "vjepa_vitl16", num_frames: int = 16):
        super().__init__()
        name = str(model_name or "vjepa_vitl16")
        if ":" in name:
            prefix, name = name.split(":", 1)
            if prefix.lower() not in {"vjepa", "jepa"}:
                raise ValueError(f"Unsupported temporal encoder prefix {prefix!r}")
        if name not in self.CHECKPOINTS:
            raise ValueError(f"UPCAST generation protection requires V-JEPA, got {model_name!r}")
        architecture, checkpoint_name, self.input_size = self.CHECKPOINTS[name]
        self.num_frames = int(num_frames)
        self.tubelet_size = 2
        self.patch_size = 16

        checkpoint_candidates = [
            Path("./metric_assets/jepa_checkpoints") / checkpoint_name,
            Path(__file__).resolve().parents[1] / "external" / "jepa_checkpoints" / checkpoint_name,
        ]
        checkpoint_path = next((path for path in checkpoint_candidates if path.exists()), None)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"V-JEPA checkpoint {checkpoint_name!r} not found under metric_assets or external/jepa_checkpoints"
            )
        jepa_root = Path(__file__).resolve().parents[1] / "external" / "jepa"
        if str(jepa_root) not in sys.path:
            sys.path.insert(0, str(jepa_root))
        try:
            import src.models.vision_transformer as vit
        except Exception as exc:
            raise ImportError(f"V-JEPA source is not importable from {jepa_root}") from exc

        self.model = vit.__dict__[architecture](
            img_size=self.input_size,
            patch_size=self.patch_size,
            num_frames=self.num_frames,
            tubelet_size=self.tubelet_size,
        )
        # This V-JEPA source revision returns tokens only for configured
        # out_layers. Use the normalized final transformer block.
        self.model.out_layers = [len(self.model.blocks) - 1]
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", mmap=True)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        state = payload.get("target_encoder", payload.get("encoder", payload))
        state = {key.replace("module.", "").replace("backbone.", ""): value for key, value in state.items()}
        incompatible = self.model.load_state_dict(state, strict=False)
        model_state = self.model.state_dict()
        loaded_numel = sum(
            value.numel()
            for key, value in model_state.items()
            if key not in set(incompatible.missing_keys)
        )
        total_numel = sum(value.numel() for value in model_state.values())
        loaded_fraction = loaded_numel / max(total_numel, 1)
        required_prefixes = ("patch_embed.", "blocks.", "norm.")
        missing_required = [
            prefix
            for prefix in required_prefixes
            if not any(
                key.startswith(prefix) and key not in set(incompatible.missing_keys)
                for key in model_state
            )
        ]
        if loaded_fraction < 0.95 or missing_required:
            raise RuntimeError(
                "V-JEPA checkpoint did not cover the requested encoder: "
                f"loaded_fraction={loaded_fraction:.3f}, "
                f"missing_required={missing_required}, "
                f"missing_keys={incompatible.missing_keys[:10]}."
            )
        del payload, state
        self.model.eval()
        self.model.requires_grad_(False)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def _sample_frames(self, videos: Tensor) -> Tensor:
        if videos.shape[1] == self.num_frames:
            return videos
        indices = torch.linspace(0, videos.shape[1] - 1, self.num_frames, device=videos.device).round().long()
        return videos.index_select(1, indices)

    def forward(self, videos: Tensor, allow_input_grad: bool = False) -> Tensor:
        videos = self._sample_frames(videos).float().clamp(0.0, 1.0)
        batch_size = videos.shape[0]
        x = rearrange(videos, "b t c h w -> (b t) c h w")
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        x = rearrange(x, "(b t) c h w -> b t c h w", b=batch_size, t=self.num_frames)
        x = (x - self.mean.to(x)) / self.std.to(x)
        x = rearrange(x, "b t c h w -> b c t h w")
        track_input_gradient = bool(allow_input_grad and videos.requires_grad)
        with torch.set_grad_enabled(track_input_gradient):
            if track_input_gradient:
                tokens = activation_checkpoint(
                    self.model,
                    x,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                tokens = self.model(x)
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected V-JEPA [B,N,D] tokens, got {tuple(tokens.shape)}")
        temporal = self.num_frames // self.tubelet_size
        spatial = self.input_size // self.patch_size
        expected = temporal * spatial * spatial
        if tokens.shape[1] == expected + 1:
            tokens = tokens[:, 1:]
        if tokens.shape[1] != expected:
            raise RuntimeError(
                f"Cannot reshape V-JEPA tokens: got {tokens.shape[1]}, expected {expected} "
                f"for ({temporal},{spatial},{spatial})"
            )
        tokens = rearrange(tokens, "b (t h w) d -> b t h w d", t=temporal, h=spatial, w=spatial)
        return F.normalize(tokens, p=2, dim=-1, eps=1e-6)


class FactorizedTransitionLearning(nn.Module):
    """UPCAST world-transition tokenizer using checkpoint schema 7."""

    def __init__(
        self,
        alignment_context_length: int = 16,
        keyframe_stride: int = 1,
        alignment_mode: str = "unit_world_latent",
        unit_arch_version: int = UNIT_ARCH_VERSION,
        geometry_teacher: str = "page4d",
        geometry_teacher_repo_id: str = "zhouk777/PAGE4D",
        geometry_teacher_checkpoint: str | None = None,
        geometry_representation: str = "point_map",
        geometry_feature_info: list[int] | tuple[int, int, int] = (3, 128, 128),
        geometry_normalize: bool = True,
        latents_info: list | tuple = ((576, 32, 32),),
        unit_dim: int = 768,
        unit_spatial_grid: list[int] | tuple[int, int] = (8, 8),
        unit_anchor_grid: list[int] | tuple[int, int] = (16, 16),
        unit_mformer_layers: int = 4,
        unit_fusion_layers: int = 4,
        unit_appearance_layers: int = 2,
        unit_appearance_fusion_layers: int = 2,
        unit_visual_decoder_layers: int = 8,
        unit_geometry_decoder_layers: int = 8,
        unit_num_heads: int = 12,
        unit_ffn_dim: int = 3072,
        unit_vq_dim: int = 64,
        unit_vq_stages: int = 2,
        unit_vq_codebook_size: int = 256,
        unit_commitment_beta: float = 0.25,
        unit_vq_ema_decay: float = 0.8,
        unit_vq_dead_code_threshold: float = 2.0,
        unit_vq_dead_code_restart_interval: int = 100,
        unit_geometry_feature_channels: int = 2048,
        unit_visual_loss_weight: float = 1.0,
        unit_geometry_loss_weight: float = 1.0,
        unit_tail_fm_loss_weight: float = 1.0,
        unit_rvq_loss_weight: float = 1.0,
        unit_code_assignment_loss_weight: float = 1.0,
        unit_soft_code_loss_weight: float = 0.5,
        unit_pre_vq_alignment_loss_weight: float = 0.5,
        unit_code_usage_loss_weight: float = 0.25,
        unit_student_spread_loss_weight: float = 0.25,
        unit_code_level_weights: list[float] | tuple[float, ...] = (2.0, 1.0),
        unit_physical_contrast_loss_weight: float = 0.1,
        unit_appearance_contrast_loss_weight: float = 0.1,
        unit_variance_loss_weight: float = 0.1,
        unit_shuffle_loss_weight: float = 0.5,
        unit_shuffle_margin: float = 0.05,
        unit_shuffle_loss_freq: int = 4,
        unit_shuffle_loss_phase: int = 2,
        unit_anchor_dropout: float = 0.25,
        unit_anchor_sample_dropout: float = 0.25,
        unit_q_only_loss_weight: float = 0.2,
        unit_q_only_loss_freq: int = 4,
        unit_q_only_loss_phase: int = 0,
        unit_contrastive_temperature: float = 0.1,
        unit_contrastive_memory_size: int = 256,
        unit_code_temperature: float = 0.1,
        unit_code_usage_temperature: float = 0.01,
        unit_joint_visual_loss_weight: float = 0.1,
        unit_joint_geometry_loss_weight: float = 0.1,
        unit_joint_tail_fm_loss_weight: float = 1.0,
        unit_joint_rvq_loss_weight: float = 0.1,
        unit_joint_code_assignment_loss_weight: float = 0.02,
        unit_joint_soft_code_loss_weight: float = 0.02,
        unit_joint_pre_vq_alignment_loss_weight: float = 0.02,
        unit_joint_physical_contrast_loss_weight: float = 0.02,
        unit_joint_appearance_contrast_loss_weight: float = 0.02,
        unit_noise_signal_threshold: float = 0.01,
        unit_dino_model_name: str = "dinov2_vits14",
        unit_physical_only_continuation: bool = False,
        unit_dino_feature_dim: int = 384,
        unit_video_tail_checkpoint: str | None = None,
        unit_tail_activation_checkpointing: bool = True,
        generation_dino_patch_loss_weight: float = 0.0,
        generation_vjepa_spatial_loss_weight: float = 0.0,
        generation_vjepa_temporal_loss_weight: float = 0.0,
        generation_vjepa_global_loss_weight: float = 0.0,
        generation_color_moment_loss_weight: float = 0.0,
        generation_rgb_temporal_loss_weight: float = 0.0,
        generation_noise_signal_threshold: float = 0.1,
        generation_video_encoder_name: str = "vjepa_vitl16",
        generation_video_num_frames: int = 16,
        generation_pixel_loss_weight: float = 0.0,
        unit_video_only_inference: bool = False,
        unit_enable_preference_scorers: bool = False,
        unit_world_state_distill_loss_weight: float = 0.1,
        unit_joint_world_state_distill_loss_weight: float = 0.02,
    ):
        super().__init__()
        if int(unit_arch_version) != UNIT_ARCH_VERSION:
            raise ValueError(
                f"UPCAST requires unit_arch_version={UNIT_ARCH_VERSION}; "
                f"got {unit_arch_version}. Older checkpoints are intentionally incompatible."
            )
        mode = str(alignment_mode or "unit_world_latent").lower()
        if mode != "unit_world_latent":
            raise ValueError(f"UPCAST only supports alignment_mode=unit_world_latent, got {mode!r}")
        teacher_name = str(geometry_teacher or "page4d").lower()
        if teacher_name not in {"page4d", "page-4d", "page_4d"}:
            raise ValueError(
                "UPCAST requires PAGE-4D patch features; "
                f"geometry_teacher={geometry_teacher!r} is not supported."
            )
        if float(generation_pixel_loss_weight) != 0.0:
            raise ValueError("UPCAST requires generation_pixel_loss_weight=0")
        self.alignment_mode = mode
        self.is_unit_world_latent_mode = True
        self.unit_video_only_inference = bool(unit_video_only_inference)
        self.unit_physical_only_continuation = bool(
            unit_physical_only_continuation
        )
        if self.unit_video_only_inference and self.unit_physical_only_continuation:
            raise ValueError(
                "video-only inference and physical-only continuation are "
                "mutually exclusive runtime reductions"
            )
        self.unit_enable_preference_scorers = bool(
            unit_enable_preference_scorers
        )
        self._preference_dino_model_name = str(
            unit_dino_model_name or "dinov2_vits14"
        )
        self._preference_video_model_name = str(
            generation_video_encoder_name or "vjepa_vitl16"
        )
        self._preference_video_num_frames = int(generation_video_num_frames)
        # Deliberately bypass nn.Module registration. These frozen scorers are
        # lazy training utilities, not checkpoint state or DDP parameters.
        object.__setattr__(self, "_preference_dino_extractor", None)
        object.__setattr__(self, "_preference_vjepa_extractor", None)
        self.uses_teacher_geometry = not self.unit_video_only_inference
        self.alignment_context_length = int(alignment_context_length)
        self.keyframe_stride = max(1, int(keyframe_stride))
        if self.alignment_context_length != 16 or self.keyframe_stride != 1:
            raise ValueError(
                "UPCAST is defined for the same 16-frame clip seen by DFoT, "
                f"got context_length={self.alignment_context_length}, stride={self.keyframe_stride}."
            )
        self.unit_dim = int(unit_dim)
        self.unit_spatial_grid = tuple(int(value) for value in unit_spatial_grid)
        self.unit_anchor_grid = tuple(int(value) for value in unit_anchor_grid)
        if self.unit_spatial_grid != (8, 8):
            raise ValueError(f"UPCAST bitrate contract requires unit_spatial_grid=[8,8], got {self.unit_spatial_grid}")
        if self.unit_anchor_grid != (16, 16):
            raise ValueError(f"UPCAST decoder contract requires unit_anchor_grid=[16,16], got {self.unit_anchor_grid}")
        self.unit_is_frozen = False
        self.unit_noise_signal_threshold = float(unit_noise_signal_threshold)
        if not 0.0 <= self.unit_noise_signal_threshold <= 1.0:
            raise ValueError(
                "unit_noise_signal_threshold must be in [0,1], got "
                f"{self.unit_noise_signal_threshold}"
            )
        self.unit_visual_loss_weight = float(unit_visual_loss_weight)
        self.unit_geometry_loss_weight = float(unit_geometry_loss_weight)
        self.unit_tail_fm_loss_weight = float(unit_tail_fm_loss_weight)
        self.unit_rvq_loss_weight = float(unit_rvq_loss_weight)
        self.unit_code_assignment_loss_weight = float(unit_code_assignment_loss_weight)
        self.unit_soft_code_loss_weight = float(unit_soft_code_loss_weight)
        self.unit_pre_vq_alignment_loss_weight = float(
            unit_pre_vq_alignment_loss_weight
        )
        self.unit_code_usage_loss_weight = float(unit_code_usage_loss_weight)
        self.unit_student_spread_loss_weight = float(
            unit_student_spread_loss_weight
        )
        self.unit_code_level_weights = tuple(
            float(value) for value in unit_code_level_weights
        )
        if len(self.unit_code_level_weights) != int(unit_vq_stages):
            raise ValueError(
                "unit_code_level_weights must match unit_vq_stages, got "
                f"{self.unit_code_level_weights} and {unit_vq_stages}"
            )
        if any(value < 0 for value in self.unit_code_level_weights) or sum(
            self.unit_code_level_weights
        ) <= 0:
            raise ValueError(
                "unit_code_level_weights must be non-negative with a positive sum"
            )
        self.unit_physical_contrast_loss_weight = float(unit_physical_contrast_loss_weight)
        self.unit_appearance_contrast_loss_weight = float(unit_appearance_contrast_loss_weight)
        self.unit_variance_loss_weight = float(unit_variance_loss_weight)
        self.unit_shuffle_loss_weight = float(unit_shuffle_loss_weight)
        self.unit_shuffle_margin = float(unit_shuffle_margin)
        self.unit_shuffle_loss_freq = max(1, int(unit_shuffle_loss_freq))
        self.unit_shuffle_loss_phase = (
            int(unit_shuffle_loss_phase) % self.unit_shuffle_loss_freq
        )
        self.unit_anchor_dropout = float(unit_anchor_dropout)
        self.unit_anchor_sample_dropout = float(unit_anchor_sample_dropout)
        self.unit_q_only_loss_weight = float(unit_q_only_loss_weight)
        self.unit_q_only_loss_freq = max(1, int(unit_q_only_loss_freq))
        self.unit_q_only_loss_phase = (
            int(unit_q_only_loss_phase) % self.unit_q_only_loss_freq
        )
        self.unit_contrastive_temperature = float(unit_contrastive_temperature)
        self.unit_contrastive_memory_size = max(
            0,
            int(unit_contrastive_memory_size),
        )
        self.unit_code_temperature = float(unit_code_temperature)
        self.unit_code_usage_temperature = float(unit_code_usage_temperature)
        self.unit_joint_visual_loss_weight = float(unit_joint_visual_loss_weight)
        self.unit_joint_geometry_loss_weight = float(unit_joint_geometry_loss_weight)
        self.unit_joint_tail_fm_loss_weight = float(unit_joint_tail_fm_loss_weight)
        self.unit_joint_rvq_loss_weight = float(unit_joint_rvq_loss_weight)
        self.unit_joint_code_assignment_loss_weight = float(
            unit_joint_code_assignment_loss_weight
        )
        self.unit_joint_soft_code_loss_weight = float(
            unit_joint_soft_code_loss_weight
        )
        self.unit_joint_pre_vq_alignment_loss_weight = float(
            unit_joint_pre_vq_alignment_loss_weight
        )
        self.unit_joint_physical_contrast_loss_weight = float(
            unit_joint_physical_contrast_loss_weight
        )
        self.unit_joint_appearance_contrast_loss_weight = float(
            unit_joint_appearance_contrast_loss_weight
        )
        self.unit_world_state_distill_loss_weight = float(
            unit_world_state_distill_loss_weight
        )
        self.unit_joint_world_state_distill_loss_weight = float(
            unit_joint_world_state_distill_loss_weight
        )
        self.unit_video_tail_checkpoint = (
            None
            if unit_video_tail_checkpoint in (None, "", "null")
            else str(unit_video_tail_checkpoint)
        )
        self.unit_tail_activation_checkpointing = bool(
            unit_tail_activation_checkpointing
        )
        if (
            not self.unit_video_only_inference
            and not self.unit_physical_only_continuation
            and max(
                self.unit_tail_fm_loss_weight,
                self.unit_joint_tail_fm_loss_weight,
            )
            > 0
            and self.unit_video_tail_checkpoint is None
        ):
            raise ValueError(
                "Frozen-tail FM requires alignment.unit_video_tail_checkpoint"
            )
        if not 0.0 <= self.unit_anchor_dropout < 1.0:
            raise ValueError(f"unit_anchor_dropout must be in [0,1), got {self.unit_anchor_dropout}")
        if not 0.0 <= self.unit_anchor_sample_dropout < 1.0:
            raise ValueError(
                "unit_anchor_sample_dropout must be in [0,1), got "
                f"{self.unit_anchor_sample_dropout}"
            )
        self.generation_dino_patch_loss_weight = float(generation_dino_patch_loss_weight)
        self.generation_vjepa_spatial_loss_weight = float(generation_vjepa_spatial_loss_weight)
        self.generation_vjepa_temporal_loss_weight = float(generation_vjepa_temporal_loss_weight)
        self.generation_vjepa_global_loss_weight = float(generation_vjepa_global_loss_weight)
        self.generation_color_moment_loss_weight = float(
            generation_color_moment_loss_weight
        )
        self.generation_rgb_temporal_loss_weight = float(
            generation_rgb_temporal_loss_weight
        )
        self.generation_noise_signal_threshold = float(
            generation_noise_signal_threshold
        )
        if not 0.0 <= self.generation_noise_signal_threshold < 1.0:
            raise ValueError(
                "generation_noise_signal_threshold must be in [0,1)"
            )
        self.requires_denoising_terms = bool(
            not self.unit_video_only_inference
            and max(
                self.unit_tail_fm_loss_weight,
                self.unit_joint_tail_fm_loss_weight,
            )
            > 0
        )
        self.register_buffer("unit_arch_version_marker", torch.tensor(UNIT_ARCH_VERSION), persistent=True)
        for modality in ("physical", "appearance"):
            self.register_buffer(
                f"unit_{modality}_teacher_memory",
                torch.zeros(self.unit_contrastive_memory_size, self.unit_dim),
                persistent=True,
            )
            self.register_buffer(
                f"unit_{modality}_teacher_memory_pointer",
                torch.zeros((), dtype=torch.long),
                persistent=True,
            )
            self.register_buffer(
                f"unit_{modality}_teacher_memory_count",
                torch.zeros((), dtype=torch.long),
                persistent=True,
            )

        self.latents_info = self._resolve_latents_info(latents_info)
        if len(self.latents_info) != 1:
            raise ValueError(
                "UPCAST uses exactly one selected DFoT level; "
                f"got latents_info={self.latents_info}"
            )
        self.geometry_target_info = tuple(int(value) for value in geometry_feature_info)
        if self.geometry_target_info[0] != 3:
            raise ValueError(f"UPCAST geometry_feature_info must start with 3 channels, got {self.geometry_target_info}")
        self.geometry_teacher = None
        if not self.unit_video_only_inference:
            self.geometry_teacher = build_geometry_teacher(
                name=teacher_name,
                repo_id=str(geometry_teacher_repo_id or "zhouk777/PAGE4D"),
                checkpoint_path=geometry_teacher_checkpoint,
                representation=str(geometry_representation or "point_map"),
                target_size=self.geometry_target_info[1:],
                normalize=bool(geometry_normalize),
                confidence_threshold=None,
            )

        heads = int(unit_num_heads)
        ff_dim = int(unit_ffn_dim)
        page_dim = int(unit_geometry_feature_channels)
        latent_channels = int(self.latents_info[0][0])
        self.dino_extractor = None
        dino_feature_dim = int(unit_dino_feature_dim)
        if dino_feature_dim < 1:
            raise ValueError("unit_dino_feature_dim must be positive")
        if not self.unit_video_only_inference and not self.unit_physical_only_continuation:
            self.dino_extractor = FrozenDINOv2PatchExtractor(
                str(unit_dino_model_name or "dinov2_vits14"),
                self.unit_anchor_grid,
            )
            dino_feature_dim = self.dino_extractor.output_dim
        self.video_pair_projector = VideoPairProjector(self.latents_info, self.unit_dim, self.unit_spatial_grid)
        self.video_transition_encoder = SpatialTransitionEncoder(
            self.unit_dim,
            self.unit_spatial_grid,
            depth=int(unit_mformer_layers),
            heads=heads,
            ff_dim=ff_dim,
            use_noise_embedding=True,
        )
        self.video_appearance_encoder = SpatialTransitionEncoder(
            self.unit_dim,
            self.unit_spatial_grid,
            depth=int(unit_appearance_layers),
            heads=heads,
            ff_dim=ff_dim,
            use_noise_embedding=True,
        )
        self.video_vq_adapter = ResidualTokenAdapter(self.unit_dim)
        self.unit_quantizer = SpatialResidualVQ(
            input_dim=self.unit_dim,
            code_dim=int(unit_vq_dim),
            num_quantizers=int(unit_vq_stages),
            codebook_size=int(unit_vq_codebook_size),
            commitment_beta=float(unit_commitment_beta),
            ema_decay=float(unit_vq_ema_decay),
            dead_code_threshold=float(unit_vq_dead_code_threshold),
            dead_code_restart_interval=int(unit_vq_dead_code_restart_interval),
        )
        if not self.unit_video_only_inference:
            self.geometry_pair_projector = GeometryPairProjector(
                page_dim,
                self.unit_dim,
                self.unit_spatial_grid,
            )
            self.geometry_transition_encoder = SpatialTransitionEncoder(
                self.unit_dim,
                self.unit_spatial_grid,
                depth=int(unit_mformer_layers),
                heads=heads,
                ff_dim=ff_dim,
                use_noise_embedding=False,
            )
            self.fusion_transition_encoder = SpatialFusionEncoder(
                self.unit_dim,
                self.unit_spatial_grid,
                depth=int(unit_fusion_layers),
                heads=heads,
                ff_dim=ff_dim,
            )
            self.geometry_vq_adapter = ResidualTokenAdapter(self.unit_dim)
            self.fused_vq_adapter = ResidualTokenAdapter(self.unit_dim)
            if not self.unit_physical_only_continuation:
                self.world_token_fusion = WorldTokenFusion(self.unit_dim)
                self.semantic_pair_projector = DenseFeaturePairProjector(
                    dino_feature_dim,
                    self.unit_dim,
                    self.unit_spatial_grid,
                )
                self.semantic_transition_encoder = SpatialTransitionEncoder(
                    self.unit_dim,
                    self.unit_spatial_grid,
                    depth=int(unit_appearance_layers),
                    heads=heads,
                    ff_dim=ff_dim,
                    use_noise_embedding=False,
                )
                self.appearance_fusion_encoder = SpatialFusionEncoder(
                    self.unit_dim,
                    self.unit_spatial_grid,
                    depth=int(unit_appearance_fusion_layers),
                    heads=heads,
                    ff_dim=ff_dim,
                )
                self.visual_anchor_encoder = VisualAnchorEncoder(
                    dino_feature_dim,
                    self.unit_dim,
                    self.unit_anchor_grid,
                )
                self.geometry_anchor_encoder = GeometryAnchorEncoder(
                    page_dim,
                    self.unit_dim,
                    self.unit_anchor_grid,
                )
                self.visual_decoder = ConditionalVisualDecoder(
                    self.unit_dim,
                    dino_feature_dim,
                    latent_channels,
                    token_grid=self.unit_spatial_grid,
                    anchor_grid=self.unit_anchor_grid,
                    depth=int(unit_visual_decoder_layers),
                    heads=heads,
                    ff_dim=ff_dim,
                    token_dim=self.unit_dim * 2,
                )
                self.geometry_decoder = ConditionalGeometryDecoder(
                    self.unit_dim,
                    page_dim,
                    token_grid=self.unit_spatial_grid,
                    anchor_grid=self.unit_anchor_grid,
                    point_size=self.geometry_target_info[1:],
                    depth=int(unit_geometry_decoder_layers),
                    heads=heads,
                    ff_dim=ff_dim,
                )
        need_vjepa = any(
            weight > 0
            for weight in (
                self.generation_vjepa_spatial_loss_weight,
                self.generation_vjepa_temporal_loss_weight,
                self.generation_vjepa_global_loss_weight,
            )
        )
        self.generation_video_extractor = (
            FrozenVJEPATokenExtractor(generation_video_encoder_name, generation_video_num_frames)
            if need_vjepa
            and not self.unit_video_only_inference
            and not self.unit_physical_only_continuation
            else None
        )

        object.__setattr__(self, "_video_tail_source_model", None)
        object.__setattr__(self, "_video_tail_model", None)
        self._video_tail_snapshot_failed = False
        self._video_tail_validated = False
        # Keep every UPCAST parameter trainable until DDP and the optimizer have
        # registered the complete model. on_train_batch_start applies the
        # geometry-only phase after distributed setup.
        self.unit_phase = "registration"
        self.unit_is_frozen = False
        self.last_metrics: dict[str, Tensor] = {}
        self.last_generation_metrics: dict[str, Tensor] = {}
        self.last_world_state_observations: dict[str, dict[str, Tensor]] = {}

    @staticmethod
    def _resolve_latents_info(latents_info):
        if len(latents_info) == 3 and all(isinstance(value, int) for value in latents_info):
            return [list(latents_info)]
        return [list(item) for item in latents_info]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.geometry_teacher is not None:
            self.geometry_teacher.eval()
        if self.dino_extractor is not None:
            self.dino_extractor.eval()
        if self.generation_video_extractor is not None:
            self.generation_video_extractor.eval()
        preference_dino = getattr(self, "_preference_dino_extractor", None)
        preference_vjepa = getattr(self, "_preference_vjepa_extractor", None)
        if preference_dino is not None:
            preference_dino.eval()
        if preference_vjepa is not None:
            preference_vjepa.eval()
        self._apply_unit_phase(self.unit_phase)
        return self

    def _unit_modules(self) -> tuple[nn.Module, ...]:
        names = (
            "video_pair_projector",
            "geometry_pair_projector",
            "semantic_pair_projector",
            "video_transition_encoder",
            "geometry_transition_encoder",
            "video_appearance_encoder",
            "semantic_transition_encoder",
            "fusion_transition_encoder",
            "appearance_fusion_encoder",
            "video_vq_adapter",
            "geometry_vq_adapter",
            "fused_vq_adapter",
            "unit_quantizer",
            "world_token_fusion",
            "visual_anchor_encoder",
            "geometry_anchor_encoder",
            "visual_decoder",
            "geometry_decoder",
        )
        return tuple(getattr(self, name) for name in names if hasattr(self, name))

    def _geometry_teacher_modules(self) -> tuple[nn.Module, ...]:
        if getattr(self, "unit_video_only_inference", False):
            return ()
        return (
            self.geometry_pair_projector,
            self.geometry_transition_encoder,
            self.geometry_vq_adapter,
            self.unit_quantizer,
        )

    def _cotrain_modules(self) -> tuple[nn.Module, ...]:
        """Video-side students allowed to track a slowly changing DFoT."""
        return (
            self.video_pair_projector,
            self.video_transition_encoder,
            self.video_appearance_encoder,
            self.video_vq_adapter,
        )

    def _apply_unit_phase(self, phase: str) -> None:
        if phase not in {
            "registration",
            "geometry",
            "alignment",
            "joint",
            "cotrain",
        }:
            raise ValueError(f"Unknown UPCAST training phase: {phase!r}")
        trainable = phase in {"registration", "geometry", "alignment"}
        for module in self._unit_modules():
            module.requires_grad_(trainable)
            module.train(trainable and self.training)
        if phase == "geometry":
            geometry_trainable = set(self._geometry_teacher_modules())
            geometry_trainable.update(
                (self.geometry_anchor_encoder, self.geometry_decoder)
            )
            for module in self._unit_modules():
                if module not in geometry_trainable:
                    module.requires_grad_(False)
                    module.eval()
        elif phase == "alignment":
            for module in self._geometry_teacher_modules():
                module.requires_grad_(False)
                module.eval()
        elif phase == "cotrain":
            # Keep geometry, RVQ, fusion, anchors and decoders as a stationary
            # teacher. Only the low-capacity video students may adapt, so the
            # shared physical space cannot move to follow DFoT shortcuts.
            for module in self._cotrain_modules():
                module.requires_grad_(True)
                module.train(self.training)
        self.unit_phase = phase
        self.unit_is_frozen = phase == "joint"
        if self.geometry_teacher is not None:
            self.geometry_teacher.eval()
        if self.dino_extractor is not None:
            self.dino_extractor.eval()

    def set_unit_phase(self, phase: str) -> None:
        phase = str(phase).lower()
        if phase != "geometry" and int(
            self.unit_quantizer.pending_microbatches.item()
        ) != 0:
            raise RuntimeError(
                "Cannot leave the geometry-tokenizer phase with an uncommitted "
                "RVQ accumulation window."
            )
        self._apply_unit_phase(phase)

    def set_unit_trainable(self, trainable: bool) -> None:
        self.set_unit_phase("alignment" if trainable else "joint")

    def commit_unit_codebook_update(
        self,
        expected_microbatches: int,
    ) -> Tensor:
        if self.unit_phase != "geometry":
            if int(self.unit_quantizer.pending_microbatches.item()) != 0:
                raise RuntimeError(
                    "A fixed UPCAST codebook has pending RVQ EMA statistics."
                )
            return torch.zeros_like(self.unit_quantizer.last_restarted_codes)
        return self.unit_quantizer.commit_codebook_update(
            expected_microbatches=expected_microbatches,
        )

    def discard_unit_codebook_update(self) -> None:
        self.unit_quantizer.discard_pending_codebook_update()

    def set_video_tail_model(self, diffusion_model: nn.Module) -> None:
        object.__setattr__(self, "_video_tail_source_model", diffusion_model)

    def _ensure_video_tail_model(self):
        if max(
            self.unit_tail_fm_loss_weight,
            self.unit_joint_tail_fm_loss_weight,
        ) <= 0:
            return None
        tail_model = self._video_tail_model
        if tail_model is not None:
            return tail_model
        source = self._video_tail_source_model
        if source is None:
            raise RuntimeError("Frozen-tail source model was not attached")
        if self._video_tail_snapshot_failed:
            raise RuntimeError("Frozen-tail model snapshot previously failed")
        try:
            backbone = getattr(source, "model", None)
            if backbone is None or not hasattr(backbone, "build_frozen_tail"):
                raise TypeError(
                    "Configured DFoT backbone does not expose build_frozen_tail"
                )
            tail_model = backbone.build_frozen_tail(
                activation_checkpointing=self.unit_tail_activation_checkpointing,
            )
            if self.unit_video_tail_checkpoint is not None:
                checkpoint_path = Path(self.unit_video_tail_checkpoint).expanduser()
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(
                        f"Frozen-tail checkpoint does not exist: {checkpoint_path}"
                    )
                try:
                    payload = torch.load(
                        checkpoint_path,
                        map_location="cpu",
                        mmap=True,
                        weights_only=False,
                    )
                except TypeError:
                    payload = torch.load(checkpoint_path, map_location="cpu")
                checkpoint_state = payload.get("state_dict", payload)
                backbone_state = {}
                prefixes = (
                    "diffusion_model.model.",
                    "diffusion_model._orig_mod.model.",
                )
                for key, value in checkpoint_state.items():
                    for prefix in prefixes:
                        if key.startswith(prefix):
                            backbone_state[key[len(prefix) :]] = value
                            break
                if not backbone_state:
                    raise RuntimeError(
                        "Frozen-tail checkpoint contains no DFoT backbone weights"
                    )
                tail_model.load_from_backbone_state_dict(backbone_state)
                del payload, checkpoint_state, backbone_state
            tail_model.eval()
            tail_model.requires_grad_(False)
            object.__setattr__(self, "_video_tail_model", tail_model)
            return tail_model
        except Exception as exc:
            self._video_tail_snapshot_failed = True
            raise RuntimeError("Failed to snapshot the frozen DFoT tail") from exc

    @torch.no_grad()
    def _extract_geometry_output(self, images: Tensor) -> GeometryTeacherOutput:
        if self.geometry_teacher is None:
            raise RuntimeError("PAGE teacher is unavailable in video-only inference mode")
        return self.geometry_teacher(images[:, : self.alignment_context_length : self.keyframe_stride])

    @torch.no_grad()
    def _extract_dino(self, images: Tensor) -> Tensor:
        if self.dino_extractor is None:
            raise RuntimeError("DINO teacher is unavailable in video-only inference mode")
        return self.dino_extractor(
            images[:, : self.alignment_context_length : self.keyframe_stride],
            allow_input_grad=False,
        )

    @torch.no_grad()
    def prepare_teacher_context(
        self,
        images: Tensor,
    ) -> PreparedUnitTeacherContext:
        """Materialize frozen PAGE/DINO targets before the DFoT graph exists."""
        aligned_images = images[
            :, : self.alignment_context_length : self.keyframe_stride
        ]
        teacher = self._extract_geometry_output(images)
        if teacher.features is None:
            raise RuntimeError(
                "UPCAST requires PAGE aggregated patch features"
            )
        batch_size, num_frames = teacher.target.shape[:2]
        if num_frames != 16:
            raise RuntimeError(
                f"UPCAST expected 16 aligned frames, got {num_frames}"
            )
        current, future, delta = self.sample_transition_pairs(
            batch_size,
            num_frames,
            teacher.target.device,
            training=self.training,
        )
        current_images = gather_frames(aligned_images, current)
        future_images = gather_frames(aligned_images, future)
        pair_dino = torch.nan_to_num(
            self._extract_dino(
                torch.stack((current_images, future_images), dim=1)
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        current_page, current_points = self._extract_geometry_anchor(
            aligned_images,
            current,
        )
        return PreparedUnitTeacherContext(
            teacher=teacher,
            current=current,
            future=future,
            delta=delta,
            pair_dino=pair_dino,
            current_page=current_page,
            current_points=current_points,
        )

    @torch.no_grad()
    def prepare_physical_teacher_context(
        self,
        images: Tensor,
        *,
        training: bool,
    ) -> PreparedPhysicalTeacherContext:
        """Prepare only PAGE targets needed by the generative physical branch.

        DINO and decoder anchors are intentionally skipped.  The new objective
        uses UPCAST as a physical transition tokenizer, not as another dense
        reconstruction graph during continuation training.
        """
        teacher = self._extract_geometry_output(images)
        if teacher.features is None:
            raise RuntimeError(
                "Physical conditioner requires PAGE aggregated patch features"
            )
        batch_size, num_frames = teacher.target.shape[:2]
        if num_frames != self.alignment_context_length:
            raise RuntimeError(
                "Physical conditioner PAGE context length mismatch: "
                f"expected {self.alignment_context_length}, got {num_frames}"
            )
        current, future, delta = self.sample_transition_pairs(
            batch_size,
            num_frames,
            teacher.target.device,
            training=bool(training),
        )
        return PreparedPhysicalTeacherContext(
            teacher=teacher,
            current=current,
            future=future,
            delta=delta,
        )

    def encode_generative_physical_tokens(
        self,
        latents: list[Tensor] | Tensor,
        denoising_terms: dict[str, Tensor | None],
        prepared_context: PreparedPhysicalTeacherContext,
    ) -> GenerativePhysicalTokens:
        """Encode video-only and oracle-fused compact physical transitions.

        UPCAST parameters and PAGE targets are normally frozen by the caller.
        The video branch remains differentiable with respect to the captured
        DFoT feature, while the fused branch is detached before use as an
        oracle teacher.
        """
        if self.unit_video_only_inference:
            raise RuntimeError(
                "Generative physical encoding requires the complete UPCAST teacher"
            )
        source_latents = self._select_source_latents(latents)
        teacher = prepared_context.teacher
        if teacher.features is None:
            raise RuntimeError("PAGE patch features are missing")
        current = prepared_context.current
        future = prepared_context.future
        delta = prepared_context.delta
        page_features = torch.nan_to_num(
            teacher.features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        points = torch.nan_to_num(
            teacher.target,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        batch_size, num_frames = points.shape[:2]
        logsnr = self._frame_logsnr(
            denoising_terms,
            batch_size,
            num_frames,
            points.device,
        )
        current_logsnr = _gather_bt(logsnr, current)
        future_logsnr = _gather_bt(logsnr, future)
        raw_signal = self._raw_transition_signal(logsnr, current, future)
        sample_mask = raw_signal >= self.unit_noise_signal_threshold

        geometry_current, geometry_future = self.geometry_pair_projector(
            page_features,
            points,
            current,
            future,
        )
        physical_geometry = self.geometry_transition_encoder(
            geometry_current,
            geometry_future,
            delta,
        )
        video_current, video_future = self.video_pair_projector(
            source_latents,
            current,
            future,
        )
        physical_video = self.video_transition_encoder(
            video_current,
            video_future,
            delta,
            current_logsnr,
            future_logsnr,
        )
        physical_fused = self.fusion_transition_encoder(
            physical_video,
            physical_geometry,
        )
        _, _, video_metrics = self.unit_quantizer(
            self.video_vq_adapter(physical_video),
            freeze_codebook=True,
        )
        _, _, fused_metrics = self.unit_quantizer(
            self.fused_vq_adapter(physical_fused),
            freeze_codebook=True,
        )
        return GenerativePhysicalTokens(
            video=video_metrics["native_quantized"],
            fused=fused_metrics["native_quantized"],
            current=current,
            future=future,
            delta=delta,
            future_logsnr=future_logsnr,
            sample_mask=sample_mask,
            video_indices=video_metrics["indices"],
            fused_indices=fused_metrics["indices"],
        )

    @torch.no_grad()
    def prepare_generation_targets(
        self,
        xs_target: Tensor,
    ) -> PreparedGenerationTargets:
        """Extract frozen GT semantics before retaining trainable video activations."""
        target = xs_target[
            :, : self.alignment_context_length : self.keyframe_stride
        ]
        dino = None
        if self.generation_dino_patch_loss_weight > 0:
            dino = self.dino_extractor(
                target,
                allow_input_grad=False,
            )
        vjepa = None
        if self.generation_video_extractor is not None:
            vjepa = self.generation_video_extractor(
                target,
                allow_input_grad=False,
            )
        return PreparedGenerationTargets(
            dino=None if dino is None else dino.detach(),
            vjepa=None if vjepa is None else vjepa.detach(),
            num_frames=int(target.shape[1]),
        )

    def _ensure_preference_scorers(
        self,
        device: torch.device,
    ) -> tuple[FrozenDINOv2PatchExtractor, FrozenVJEPATokenExtractor]:
        if not self.unit_enable_preference_scorers:
            raise RuntimeError(
                "Sampled-preference scorers are disabled; set "
                "alignment.unit_enable_preference_scorers=true"
            )
        dino = getattr(self, "_preference_dino_extractor", None)
        if dino is None:
            dino = FrozenDINOv2PatchExtractor(
                self._preference_dino_model_name,
                self.unit_anchor_grid,
            )
            dino.requires_grad_(False).eval().to(device)
            object.__setattr__(self, "_preference_dino_extractor", dino)
        elif next(dino.parameters()).device != device:
            dino.to(device)
        vjepa = getattr(self, "_preference_vjepa_extractor", None)
        if vjepa is None:
            vjepa = FrozenVJEPATokenExtractor(
                self._preference_video_model_name,
                self._preference_video_num_frames,
            )
            vjepa.requires_grad_(False).eval().to(device)
            object.__setattr__(self, "_preference_vjepa_extractor", vjepa)
        elif next(vjepa.parameters()).device != device:
            vjepa.to(device)
        return dino, vjepa

    @torch.no_grad()
    def prepare_preference_targets(
        self,
        xs_target: Tensor,
    ) -> PreparedPreferenceTargets:
        """Cache frozen target features for sampled state counterfactuals."""
        target = xs_target[
            :, : self.alignment_context_length : self.keyframe_stride
        ].float().clamp(0.0, 1.0)
        dino, vjepa = self._ensure_preference_scorers(target.device)
        return PreparedPreferenceTargets(
            dino=dino(target, allow_input_grad=False).detach(),
            vjepa=vjepa(target, allow_input_grad=False).detach(),
            num_frames=int(target.shape[1]),
        )

    @staticmethod
    def _preference_weighted_per_sample(
        values: Tensor,
        weights: Tensor,
    ) -> Tensor:
        while weights.ndim < values.ndim:
            weights = weights.unsqueeze(-1)
        expanded = weights.expand_as(values).to(values)
        numerator = (values * expanded).flatten(1).sum(dim=1)
        denominator = expanded.ne(0).flatten(1).sum(dim=1).clamp_min(1)
        return numerator / denominator

    @torch.no_grad()
    def compute_preference_components(
        self,
        videos: Tensor,
        targets: PreparedPreferenceTargets,
        *,
        future_start: int = 0,
    ) -> dict[str, Tensor]:
        """Return per-sample lower-is-better visual and temporal scores."""
        pred = videos[
            :, : self.alignment_context_length : self.keyframe_stride
        ].float().clamp(0.0, 1.0)
        if pred.shape[1] != targets.num_frames:
            raise ValueError(
                "Preference target length mismatch: "
                f"{pred.shape[1]} != {targets.num_frames}"
            )
        if not 0 <= int(future_start) < pred.shape[1]:
            raise ValueError(
                f"future_start must be in [0,{pred.shape[1] - 1}]"
            )
        dino, vjepa = self._ensure_preference_scorers(pred.device)
        pred_dino = dino(pred, allow_input_grad=False)
        target_dino = targets.dino.to(pred_dino)
        dino_map = 1.0 - (
            F.normalize(pred_dino.float(), dim=2, eps=1e-6)
            * F.normalize(target_dino.float(), dim=2, eps=1e-6)
        ).sum(dim=2)
        frame_weights = pred.new_zeros(pred.shape[:2], dtype=torch.float32)
        frame_weights[:, int(future_start) :] = 1.0
        dino_score = self._preference_weighted_per_sample(
            dino_map,
            frame_weights,
        )

        pred_tokens = vjepa(pred, allow_input_grad=False)
        target_tokens = targets.vjepa.to(pred_tokens)
        temporal_tokens = pred_tokens.shape[1]
        tubelet_weights = F.adaptive_avg_pool1d(
            frame_weights[:, None], temporal_tokens
        ).squeeze(1)
        spatial_map = 1.0 - (pred_tokens * target_tokens).sum(dim=-1)
        spatial_score = self._preference_weighted_per_sample(
            spatial_map,
            tubelet_weights,
        )

        temporal_score = pred.new_zeros(pred.shape[0], dtype=torch.float32)
        if temporal_tokens > 1:
            pred_delta = pred_tokens[:, 1:].float() - pred_tokens[:, :-1].float()
            target_delta = (
                target_tokens[:, 1:].float() - target_tokens[:, :-1].float()
            )
            direction_map = 1.0 - (
                F.normalize(pred_delta, dim=-1, eps=1e-6)
                * F.normalize(target_delta, dim=-1, eps=1e-6)
            ).sum(dim=-1)
            pred_magnitude = pred_delta.norm(dim=-1)
            target_magnitude = target_delta.norm(dim=-1)
            magnitude_map = (
                torch.log(pred_magnitude + 1e-4)
                - torch.log(target_magnitude + 1e-4)
            ).abs().clamp_max(5.0)
            temporal_weights = torch.minimum(
                tubelet_weights[:, 1:], tubelet_weights[:, :-1]
            )
            motion_mask = (target_magnitude > 1e-3).to(temporal_weights)
            direction_score = self._preference_weighted_per_sample(
                direction_map,
                temporal_weights[:, :, None, None] * motion_mask,
            )
            magnitude_score = self._preference_weighted_per_sample(
                magnitude_map,
                temporal_weights,
            )
            temporal_score = direction_score + 0.1 * magnitude_score
        return {
            "dino": dino_score.to(pred),
            "vjepa_spatial": spatial_score.to(pred),
            "vjepa_temporal": temporal_score.to(pred),
        }

    @staticmethod
    def _build_causal_geometry_prefix(images: Tensor, current_indices: Tensor) -> Tensor:
        """Build a batched PAGE prefix without exposing frames after each t."""
        max_current = int(current_indices.max().item())
        prefix = images[:, : max_current + 1].clone()
        current_images = gather_frames(images, current_indices)
        positions = torch.arange(max_current + 1, device=images.device)[None]
        repeat_current = positions > current_indices[:, None]
        return torch.where(
            repeat_current[:, :, None, None, None],
            current_images[:, None],
            prefix,
        )

    @torch.no_grad()
    def _extract_geometry_anchor(
        self,
        images: Tensor,
        current_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # PAGE world coordinates are sequence-gauge dependent. A single-frame
        # call would not share the first-frame world gauge used by the full-clip
        # target. Repeated current frames pad a common batched prefix without
        # revealing any actual frame after t.
        prefix = self._build_causal_geometry_prefix(images, current_indices)
        output = self.geometry_teacher(prefix)
        if output.features is None:
            raise RuntimeError("PAGE-4D did not return causal-prefix anchor features")
        features = torch.nan_to_num(
            gather_frames(output.features, current_indices),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        points = torch.nan_to_num(
            gather_frames(output.target, current_indices),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        normalized_confidence = self._normalize_teacher_confidence(
            output.confidence,
            output.target,
        )
        confidence = gather_frames(
            normalized_confidence[:, :, None],
            current_indices,
        )[:, 0]
        points = points * confidence[:, None]
        feature_confidence = resize_map(confidence[:, None], features.shape[-2:])
        features = features * feature_confidence
        return features, points

    def _select_source_latents(self, latents: list[Tensor] | Tensor) -> list[Tensor]:
        candidates = [latents] if isinstance(latents, Tensor) else list(latents)
        selected = []
        used: set[int] = set()
        for channels, height, width in self.latents_info:
            hit = None
            for index in range(len(candidates)):
                latent = candidates[index]
                if index not in used and tuple(latent.shape[2:]) == (channels, height, width):
                    hit = index
                    break
            if hit is None:
                shapes = [tuple(candidate.shape) for candidate in candidates]
                raise RuntimeError(
                    f"Configured alignment latent {(channels, height, width)} was not captured; got {shapes}"
                )
            selected.append(candidates[hit][:, : self.alignment_context_length : self.keyframe_stride])
            used.add(hit)
        return selected

    @staticmethod
    def _world_state_transition_spans(num_frames: int) -> tuple[tuple[int, int], ...]:
        """Cover local motion and the complete chunk without future leakage."""
        if num_frames < 2:
            raise ValueError("A world-state observation needs at least two frames")
        spans = []
        for current in range(0, num_frames - 1, 4):
            future = min(current + 3, num_frames - 1)
            if future > current:
                spans.append((current, future))
        global_span = (0, num_frames - 1)
        if global_span not in spans:
            spans.append(global_span)
        return tuple(spans)

    def encode_video_world_observation(
        self,
        latents: list[Tensor] | Tensor,
        logsnr: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Encode one finalized chunk using only the video-side UPCAST path.

        PAGE and DINO are intentionally absent here. Local 4-frame transitions
        preserve motion detail while the 0-to-last span carries camera/layout
        change across a complete autoregressive window.
        """
        source_latents = self._select_source_latents(latents)
        batch_size, num_frames = source_latents[0].shape[:2]
        if logsnr is None:
            logsnr = source_latents[0].new_full((batch_size, num_frames), 20.0)
        else:
            logsnr = logsnr.to(device=source_latents[0].device).float()
            if logsnr.shape != (batch_size, num_frames):
                raise ValueError(
                    "World-state logSNR must match the captured clip, got "
                    f"{tuple(logsnr.shape)} and {(batch_size, num_frames)}"
                )

        physical_parts = []
        appearance_parts = []
        code_indices = []
        token_commitments = []
        transition_spans = self._world_state_transition_spans(num_frames)
        for current_index, future_index in transition_spans:
            current = torch.full(
                (batch_size,),
                current_index,
                device=source_latents[0].device,
                dtype=torch.long,
            )
            future = torch.full_like(current, future_index)
            delta = future - current
            current_map, future_map = self.video_pair_projector(
                source_latents,
                current,
                future,
            )
            physical = self.video_transition_encoder(
                current_map,
                future_map,
                delta,
                _gather_bt(logsnr, current),
                _gather_bt(logsnr, future),
            )
            appearance = self.video_appearance_encoder(
                current_map,
                future_map,
                delta,
                _gather_bt(logsnr, current),
                _gather_bt(logsnr, future),
            )
            quantized, _, metrics = self.unit_quantizer(
                self.video_vq_adapter(physical),
                freeze_codebook=True,
            )
            physical_parts.append(quantized)
            appearance_parts.append(appearance)
            code_indices.append(metrics["indices"])
            token_commitments.append(metrics["commitment_per_token"])

        physical_spans = torch.stack(physical_parts, dim=1)
        appearance_spans = torch.stack(appearance_parts, dim=1)
        physical = physical_spans.mean(dim=1)
        appearance = appearance_spans.mean(dim=1)
        # The state updater owns the learned memory dynamics. Parameter-free
        # normalization preserves compatibility with existing checkpoints.
        physical = F.layer_norm(physical, (physical.shape[-1],))
        appearance = F.layer_norm(appearance, (appearance.shape[-1],))
        indices = torch.stack(code_indices, dim=1)
        commitments = torch.stack(token_commitments, dim=1)
        return {
            "physical": physical,
            "appearance": appearance,
            "physical_spans": physical_spans,
            "appearance_spans": appearance_spans,
            "indices": indices,
            "commitment_per_token": commitments,
            "span_current_indices": torch.tensor(
                [current for current, _ in transition_spans],
                device=physical.device,
                dtype=torch.long,
            ),
            "span_future_indices": torch.tensor(
                [future for _, future in transition_spans],
                device=physical.device,
                dtype=torch.long,
            ),
            "num_spans": physical.new_tensor(float(len(physical_parts))),
            "span_code_agreement": (
                indices == indices[:, :1]
            ).float().mean(),
        }

    @staticmethod
    def sample_transition_pairs(
        batch_size: int,
        num_frames: int,
        device: torch.device,
        training: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if num_frames < 2:
            raise ValueError("Spatial transitions require at least two frames")
        if not training:
            current = torch.zeros(batch_size, device=device, dtype=torch.long)
            future = torch.full_like(current, num_frames - 1)
            return current, future, future - current

        categories = torch.randint(0, 3, (batch_size,), device=device)
        ranges = ((1, 3), (4, 8), (9, 15))
        deltas = torch.empty(batch_size, device=device, dtype=torch.long)
        for category, (lower, upper) in enumerate(ranges):
            mask = categories == category
            count = int(mask.sum())
            if count == 0:
                continue
            low = min(lower, num_frames - 1)
            high = min(upper, num_frames - 1)
            if low > high:
                low = high
            deltas[mask] = torch.randint(low, high + 1, (count,), device=device)
        max_current = num_frames - 1 - deltas
        current = (torch.rand(batch_size, device=device) * (max_current + 1).float()).floor().long()
        future = current + deltas
        return current, future, deltas

    def _frame_logsnr(self, denoising_terms: dict | None, batch_size: int, num_frames: int, device) -> Tensor:
        if denoising_terms is None or denoising_terms.get("logsnr") is None:
            return torch.full((batch_size, num_frames), 20.0, device=device)
        logsnr = denoising_terms["logsnr"].to(device=device).float()
        return logsnr[:, :num_frames]

    @staticmethod
    def _raw_transition_signal(logsnr: Tensor, current: Tensor, future: Tensor) -> Tensor:
        return torch.sqrt(
            torch.sigmoid(_gather_bt(logsnr, current))
            * torch.sigmoid(_gather_bt(logsnr, future))
        )

    def _transition_signal(self, logsnr: Tensor, current: Tensor, future: Tensor) -> Tensor:
        signal = self._raw_transition_signal(logsnr, current, future)
        return torch.where(signal >= self.unit_noise_signal_threshold, signal, torch.zeros_like(signal))

    @staticmethod
    def _normalize_teacher_confidence(confidence: Tensor | None, points: Tensor) -> Tensor:
        if confidence is None:
            confidence = torch.ones(points.shape[0], points.shape[1], *points.shape[-2:], device=points.device)
        confidence = torch.nan_to_num(
            confidence.to(device=points.device, dtype=points.dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if confidence.ndim == 5:
            confidence = confidence.squeeze(2)
        flat = confidence.flatten(2)
        p90 = torch.quantile(flat.float(), 0.9, dim=2, keepdim=True).to(confidence.dtype)
        normalized = (flat / p90.clamp_min(1e-6)).clamp(0.0, 1.0).reshape_as(confidence)
        finite = torch.isfinite(points).all(dim=2)
        return normalized * finite.to(normalized)

    @staticmethod
    def _feature_cosine_per_sample(pred_tokens: Tensor, target_map: Tensor, confidence: Tensor) -> Tensor:
        target_map = resize_map(target_map, (16, 16))
        target_tokens = rearrange(target_map, "b c h w -> b (h w) c")
        pred = F.normalize(pred_tokens.float(), p=2, dim=-1, eps=1e-6)
        target = F.normalize(target_tokens.float(), p=2, dim=-1, eps=1e-6)
        loss = 1.0 - (pred * target).sum(dim=-1)
        weights = resize_map(confidence[:, None].float(), (16, 16)).flatten(1)
        return (loss * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)

    @staticmethod
    def _point_components_per_sample(
        pred: Tensor,
        pred_confidence: Tensor,
        target: Tensor,
        confidence: Tensor,
    ) -> dict[str, Tensor]:
        pred = pred.float()
        target = target.float()
        weights = confidence[:, None].float()
        denominator = weights.flatten(1).sum(dim=1).clamp_min(1e-6)
        robust_map = F.smooth_l1_loss(pred, target, reduction="none", beta=0.05).mean(dim=1)
        point = (robust_map * weights[:, 0]).flatten(1).sum(dim=1) / denominator

        gradient_scales = []
        normal_scales = []
        for stride in (1, 2, 4):
            pred_scale = pred[..., ::stride, ::stride]
            target_scale = target[..., ::stride, ::stride]
            weight_scale = weights[..., ::stride, ::stride]

            pred_dx = pred_scale[..., 1:] - pred_scale[..., :-1]
            pred_dy = pred_scale[..., 1:, :] - pred_scale[..., :-1, :]
            target_dx = target_scale[..., 1:] - target_scale[..., :-1]
            target_dy = target_scale[..., 1:, :] - target_scale[..., :-1, :]
            wx = weight_scale[..., 1:] * weight_scale[..., :-1]
            wy = weight_scale[..., 1:, :] * weight_scale[..., :-1, :]
            grad_x = (
                ((pred_dx - target_dx).abs().mean(dim=1) * wx[:, 0])
                .flatten(1)
                .sum(dim=1)
                / wx[:, 0].flatten(1).sum(dim=1).clamp_min(1e-6)
            )
            grad_y = (
                ((pred_dy - target_dy).abs().mean(dim=1) * wy[:, 0])
                .flatten(1)
                .sum(dim=1)
                / wy[:, 0].flatten(1).sum(dim=1).clamp_min(1e-6)
            )
            gradient_scales.append(0.5 * (grad_x + grad_y))

            pred_normal = F.normalize(
                torch.cross(pred_dx[..., :-1, :], pred_dy[..., :, :-1], dim=1),
                dim=1,
                eps=1e-6,
            )
            target_normal = F.normalize(
                torch.cross(target_dx[..., :-1, :], target_dy[..., :, :-1], dim=1),
                dim=1,
                eps=1e-6,
            )
            normal_map = 1.0 - (pred_normal * target_normal).sum(dim=1)
            wn = (
                weight_scale[..., :-1, :-1]
                * weight_scale[..., :-1, 1:]
                * weight_scale[..., 1:, :-1]
            )
            normal_scales.append(
                (normal_map * wn[:, 0]).flatten(1).sum(dim=1)
                / wn[:, 0].flatten(1).sum(dim=1).clamp_min(1e-6)
            )
        gradient = torch.stack(gradient_scales).mean(dim=0)
        normal = torch.stack(normal_scales).mean(dim=0)

        precision = pred_confidence.float().clamp(1e-4, 1e4)
        euclidean_error = torch.linalg.vector_norm(pred - target, dim=1, keepdim=True)
        # PAGE-4D uses alpha=0.2 for confidence regularization.
        confidence_nll_map = precision * euclidean_error - 0.2 * torch.log(precision)
        confidence_nll = (confidence_nll_map * weights).flatten(1).sum(dim=1) / denominator
        return {
            "point": point,
            "gradient": gradient,
            "normal": normal,
            "confidence_nll": confidence_nll,
        }

    def _geometry_loss(
        self,
        pred_feature: Tensor,
        pred_point: Tensor,
        pred_confidence: Tensor,
        target_feature: Tensor,
        target_point: Tensor,
        confidence: Tensor,
        sample_mask: Tensor | None,
        return_per_sample: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]] | tuple[Tensor, dict[str, Tensor], Tensor]:
        feature = self._feature_cosine_per_sample(pred_feature, target_feature, confidence)
        point_parts = self._point_components_per_sample(pred_point, pred_confidence, target_point, confidence)
        per_sample = (
            feature
            + point_parts["point"]
            + 0.25 * point_parts["gradient"]
            + 0.25 * point_parts["normal"]
            + point_parts["confidence_nll"]
        )
        parts = {"feature": _masked_batch_mean(feature, sample_mask)}
        parts.update({name: _masked_batch_mean(value, sample_mask) for name, value in point_parts.items()})
        reduced = _masked_batch_mean(per_sample, sample_mask)
        if return_per_sample:
            return reduced, parts, per_sample
        return reduced, parts

    @staticmethod
    def _visual_loss(
        pred_tokens: Tensor,
        target_map: Tensor,
        sample_mask: Tensor | None,
        return_per_sample: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        target = rearrange(resize_map(target_map, (16, 16)), "b c h w -> b (h w) c")
        cosine = 1.0 - (
            F.normalize(pred_tokens.float(), dim=-1, eps=1e-6)
            * F.normalize(target.float(), dim=-1, eps=1e-6)
        ).sum(dim=-1)
        per_sample = cosine.mean(dim=1)
        reduced = _masked_batch_mean(per_sample, sample_mask)
        if return_per_sample:
            return reduced, per_sample
        return reduced

    @staticmethod
    def _tail_source_is_static(source: nn.Module | None) -> bool:
        return source is not None and not source.training

    def _run_frozen_unit_checkpoint(self, function, *args):
        """Recompute UPCAST paths during low-memory distillation/cotrain."""
        should_checkpoint = (
            self.unit_phase in {"joint", "cotrain"}
            and torch.is_grad_enabled()
            and any(
                torch.is_tensor(value) and value.requires_grad
                for value in args
            )
        )
        if not should_checkpoint:
            return function(*args)
        return activation_checkpoint(
            function,
            *args,
            use_reentrant=False,
            preserve_rng_state=False,
        )

    def _run_video_tail(
        self,
        tail_model: nn.Module,
        replacement: Tensor,
        tail_state: dict[str, Any],
    ) -> Tensor:
        if (
            self.unit_tail_activation_checkpointing
            and torch.is_grad_enabled()
            and replacement.requires_grad
        ):
            return activation_checkpoint(
                lambda value: tail_model(value, tail_state),
                replacement,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return tail_model(replacement, tail_state)

    def _tail_fm_loss(
        self,
        pred_future_latent: Tensor,
        captured_latent: Tensor,
        future_indices: Tensor,
        sample_mask: Tensor | None,
        denoising_terms: dict | None,
        validate_source: bool = False,
    ) -> tuple[Tensor, Tensor]:
        zero = pred_future_latent.sum() * 0.0
        if denoising_terms is None:
            raise RuntimeError("Tail FM is enabled but denoising_terms were not provided")
        required = ("tail_state", "x_t", "noise", "logsnr")
        if any(denoising_terms.get(key) is None for key in required):
            missing = [key for key in required if denoising_terms.get(key) is None]
            raise RuntimeError(f"Tail FM is missing denoising terms: {missing}")
        tail_model = self._ensure_video_tail_model()

        batch_size, num_frames = captured_latent.shape[:2]
        frame_mask = F.one_hot(future_indices, num_classes=num_frames).to(pred_future_latent)
        frame_mask = frame_mask[:, :, None, None, None]
        replacement = captured_latent.detach() * (1.0 - frame_mask) + pred_future_latent[:, None] * frame_mask
        tail_state = _detach_tree(denoising_terms["tail_state"])
        tail_model.eval()
        source = self._video_tail_source_model
        # In joint training the tail stays at the bootstrap-start weights while
        # the source suffix updates. An explicit execution-phase flag limits
        # this exactness check to the no-grad bootstrap forward.
        if (
            not self._video_tail_validated
            and validate_source
            and source is not None
            and self._tail_source_is_static(source)
            and denoising_terms.get("v_pred") is not None
        ):
            with torch.no_grad():
                reconstructed = tail_model(captured_latent.detach(), tail_state)
                reference = denoising_terms["v_pred"].detach()
                if not torch.allclose(
                    reconstructed.float(),
                    reference.float(),
                    rtol=5e-3,
                    atol=5e-3,
                ):
                    absolute_error = (reconstructed.float() - reference.float()).abs()
                    raise RuntimeError(
                        "Frozen tail does not reproduce the original DFoT suffix: "
                        f"mean_error={absolute_error.mean().item():.6g}, "
                        f"max_error={absolute_error.max().item():.6g}."
                    )
            self._video_tail_validated = True
        # Per-block checkpointing does not release the down/up skip state or the
        # tail's non-block activations. Checkpoint the complete frozen suffix once
        # DFoT gradients persist across accumulated micro-batches.
        tail_v = self._run_video_tail(tail_model, replacement, tail_state)

        x_t = gather_frames(denoising_terms["x_t"], future_indices)
        noise = gather_frames(denoising_terms["noise"], future_indices)
        logsnr = _gather_bt(denoising_terms["logsnr"], future_indices)
        tail_v = gather_frames(tail_v, future_indices)
        alpha = torch.sigmoid(logsnr).sqrt()[:, None, None, None]
        sigma = torch.sigmoid(-logsnr).sqrt()[:, None, None, None]
        noise_pred = alpha * tail_v + sigma * x_t
        loss = F.mse_loss(noise_pred, noise, reduction="none").flatten(1).mean(dim=1)
        sigmoid_bias = denoising_terms.get("sigmoid_bias")
        if sigmoid_bias is not None:
            loss = loss * torch.sigmoid(sigmoid_bias.to(logsnr) - logsnr)
        return _masked_batch_mean(loss, sample_mask), zero.detach().new_ones(())

    @staticmethod
    def _shuffle_transition_tokens(tokens: Tensor, grid: tuple[int, int]) -> Tensor:
        if tokens.shape[0] > 1:
            return tokens.roll(1, dims=0)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            # Per-rank batch size is normally one. A transition from the next
            # rank is a much stronger counterfactual than a spatial roll of the
            # same room, especially under large camera motion.
            gathered = [torch.zeros_like(tokens) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, tokens.detach())
            return gathered[(dist.get_rank() + 1) % dist.get_world_size()]
        token_map = rearrange(tokens, "b (h w) c -> b h w c", h=grid[0], w=grid[1])
        return rearrange(token_map.roll(1, dims=2), "b h w c -> b (h w) c")

    @staticmethod
    def _variance_loss(tokens: Tensor, sample_mask: Tensor | None = None) -> Tensor:
        values = tokens.float()
        if sample_mask is not None:
            values = values[sample_mask.bool()]
        values = values.reshape(-1, values.shape[-1])
        if values.shape[0] < 2:
            return tokens.sum() * 0.0
        # The training configuration intentionally uses one clip per rank.
        # Computing variance after temporal-token pooling therefore returned
        # exactly zero forever. Spatial transition tokens are the population
        # being quantized, so their variance is also the relevant anti-collapse
        # statistic for the 8x8 shared codebook.
        standard_deviation = torch.sqrt(values.var(dim=0, unbiased=False) + 1e-4)
        return F.relu(1.0 - standard_deviation).mean()

    def _contrastive_memory(self, modality: str) -> Tensor:
        memory = getattr(self, f"unit_{modality}_teacher_memory")
        count = int(
            getattr(self, f"unit_{modality}_teacher_memory_count").item()
        )
        return memory[:count].detach()

    @torch.no_grad()
    def _enqueue_contrastive_memory(
        self,
        modality: str,
        teacher: Tensor,
        sample_mask: Tensor | None = None,
    ) -> None:
        memory = getattr(self, f"unit_{modality}_teacher_memory")
        if memory.shape[0] == 0:
            return
        pooled = teacher.detach().float().mean(dim=1)
        if sample_mask is not None:
            pooled = pooled[sample_mask.bool()]
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            count = torch.tensor([pooled.shape[0]], device=teacher.device, dtype=torch.long)
            gathered_counts = [torch.zeros_like(count) for _ in range(world_size)]
            dist.all_gather(gathered_counts, count)
            counts = [int(value.item()) for value in gathered_counts]
            max_count = max(counts)
            if max_count > 0:
                padded = F.pad(pooled, (0, 0, 0, max_count - pooled.shape[0]))
                gathered = [torch.zeros_like(padded) for _ in range(world_size)]
                dist.all_gather(gathered, padded)
                pooled = torch.cat(
                    [
                        value[:count_value]
                        for value, count_value in zip(gathered, counts)
                    ],
                    dim=0,
                )
            else:
                pooled = pooled[:0]
        if pooled.shape[0] == 0:
            return

        pooled = pooled.to(device=memory.device, dtype=memory.dtype)
        capacity = memory.shape[0]
        if pooled.shape[0] >= capacity:
            memory.copy_(pooled[-capacity:])
            getattr(
                self,
                f"unit_{modality}_teacher_memory_pointer",
            ).zero_()
            getattr(
                self,
                f"unit_{modality}_teacher_memory_count",
            ).fill_(capacity)
            return

        pointer_buffer = getattr(
            self,
            f"unit_{modality}_teacher_memory_pointer",
        )
        count_buffer = getattr(
            self,
            f"unit_{modality}_teacher_memory_count",
        )
        pointer = int(pointer_buffer.item())
        rows = pooled.shape[0]
        first = min(rows, capacity - pointer)
        memory[pointer : pointer + first].copy_(pooled[:first])
        if first < rows:
            memory[: rows - first].copy_(pooled[first:])
        pointer_buffer.fill_((pointer + rows) % capacity)
        count_buffer.fill_(min(capacity, int(count_buffer.item()) + rows))

    def _contrastive_loss(
        self,
        query: Tensor,
        teacher: Tensor,
        sample_mask: Tensor | None = None,
        memory: Tensor | None = None,
    ) -> Tensor:
        query_pooled = F.normalize(query.float().mean(dim=1), dim=-1, eps=1e-6)
        teacher_pooled = F.normalize(
            teacher.detach().float().mean(dim=1), dim=-1, eps=1e-6
        )
        if sample_mask is not None:
            valid = sample_mask.bool()
            query_pooled = query_pooled[valid]
            teacher_pooled = teacher_pooled[valid]
        local_count = query_pooled.shape[0]
        target_offset = 0
        all_teachers = teacher_pooled
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            count = torch.tensor([local_count], device=query.device, dtype=torch.long)
            gathered_counts = [torch.zeros_like(count) for _ in range(world_size)]
            dist.all_gather(gathered_counts, count)
            counts = [int(value.item()) for value in gathered_counts]
            max_count = max(counts)
            if max_count > 0:
                padded = F.pad(
                    teacher_pooled,
                    (0, 0, 0, max_count - local_count),
                )
                gathered = [torch.zeros_like(padded) for _ in range(world_size)]
                dist.all_gather(gathered, padded.detach())
                all_teachers = torch.cat(
                    [
                        value[:count_value]
                        for value, count_value in zip(gathered, counts)
                    ],
                    dim=0,
                )
            else:
                all_teachers = teacher_pooled
            target_offset = sum(counts[:rank])

        if memory is not None and memory.shape[0] > 0:
            all_teachers = torch.cat(
                (
                    all_teachers,
                    F.normalize(memory.detach().float(), dim=-1, eps=1e-6),
                ),
                dim=0,
            )
        if local_count == 0:
            return query.sum() * 0.0
        if all_teachers.shape[0] < 2:
            return (1.0 - (query_pooled * teacher_pooled).sum(dim=-1)).mean()
        logits = query_pooled @ all_teachers.t()
        logits = logits / max(self.unit_contrastive_temperature, 1e-4)
        targets = torch.arange(local_count, device=query.device) + target_offset
        return F.cross_entropy(logits, targets)

    @staticmethod
    @torch.no_grad()
    def _retrieval_accuracy(
        query: Tensor,
        teacher: Tensor,
        memory: Tensor | None = None,
    ) -> Tensor:
        query = F.normalize(query.float().mean(dim=1), dim=-1, eps=1e-6)
        teacher = F.normalize(teacher.float().mean(dim=1), dim=-1, eps=1e-6)
        candidates = teacher
        if memory is not None and memory.shape[0] > 0:
            candidates = torch.cat(
                (
                    teacher,
                    F.normalize(memory.detach().float(), dim=-1, eps=1e-6),
                ),
                dim=0,
            )
        predictions = (query @ candidates.t()).argmax(dim=1)
        targets = torch.arange(query.shape[0], device=query.device)
        return (predictions == targets).float().mean()

    def _assignment_loss(
        self,
        student: Tensor,
        teacher_indices: Tensor,
        sample_mask: Tensor | None,
    ) -> Tensor:
        if sample_mask is not None:
            valid = sample_mask.bool()
            student = student[valid]
            teacher_indices = teacher_indices[valid]
        if student.shape[0] == 0:
            return student.sum() * 0.0
        return self.unit_quantizer.assignment_loss(
            student,
            teacher_indices.detach(),
            temperature=self.unit_code_temperature,
            level_weights=self.unit_code_level_weights,
        )

    def _soft_assignment_loss(
        self,
        student: Tensor,
        teacher: Tensor,
        sample_mask: Tensor | None,
    ) -> Tensor:
        if sample_mask is not None:
            valid = sample_mask.bool()
            student = student[valid]
            teacher = teacher[valid]
        if student.shape[0] == 0:
            return student.sum() * 0.0
        return self.unit_quantizer.soft_assignment_distillation(
            student,
            teacher.detach(),
            temperature=self.unit_code_temperature,
            level_weights=self.unit_code_level_weights,
        )

    def _pre_vq_alignment_loss(
        self,
        student: Tensor,
        teacher: Tensor,
        sample_mask: Tensor | None,
    ) -> Tensor:
        if sample_mask is not None:
            valid = sample_mask.bool()
            student = student[valid]
            teacher = teacher[valid]
        if student.shape[0] == 0:
            return student.sum() * 0.0
        return self.unit_quantizer.projected_alignment_loss(
            student,
            teacher.detach(),
        )

    def _quantize_physical_transitions(
        self,
        physical_video: Tensor | None,
        physical_geometry: Tensor,
        physical_fused: Tensor | None,
        sample_mask: Tensor,
    ) -> tuple[
        dict[str, Tensor],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        dict[str, dict[str, Tensor]],
    ]:
        phase = self.unit_phase
        geometry_warmup = phase == "geometry"
        update_codebook = bool(geometry_warmup and self.training)
        adapted_geometry = self.geometry_vq_adapter(physical_geometry)
        q_geometry, _, geometry_metrics = self.unit_quantizer(
            adapted_geometry,
            freeze_codebook=True,
        )
        branches = {"g": q_geometry}
        metrics = {"g": geometry_metrics}

        zero = adapted_geometry.sum() * 0.0
        if geometry_warmup:
            commitments = [geometry_metrics["commitment_per_sample"].mean()]
            assignments: list[Tensor] = []
            soft_assignments: list[Tensor] = []
            pre_vq_alignments: list[Tensor] = []
            code_usage_loss, usage_metrics = self.unit_quantizer.codebook_usage_loss(
                adapted_geometry,
                temperature=self.unit_code_usage_temperature,
            )
            student_spreads: list[Tensor] = []
        else:
            if physical_video is None:
                raise RuntimeError(
                    "Video transitions are required after geometry warmup"
                )
            adapted_video = self.video_vq_adapter(physical_video)
            q_video, _, video_metrics = self.unit_quantizer(
                adapted_video,
                freeze_codebook=True,
            )
            branches["v"] = q_video
            metrics["v"] = video_metrics
            commitments = [
                _masked_batch_mean(
                    video_metrics["commitment_per_sample"],
                    sample_mask,
                )
            ]
            assignments = [
                self._assignment_loss(
                    adapted_video,
                    geometry_metrics["indices"],
                    sample_mask,
                )
            ]
            soft_assignments = [
                self._soft_assignment_loss(
                    adapted_video,
                    adapted_geometry,
                    sample_mask,
                )
            ]
            pre_vq_alignments = [
                self._pre_vq_alignment_loss(
                    adapted_video,
                    adapted_geometry,
                    sample_mask,
                )
            ]
            code_usage_loss = zero
            usage_metrics = {
                "soft_perplexity": torch.zeros(
                    self.unit_quantizer.num_quantizers,
                    device=adapted_geometry.device,
                ),
                "projected_std": adapted_geometry.new_zeros(()),
                "projected_spread_loss": adapted_geometry.new_zeros(()),
            }
            valid_video = adapted_video[sample_mask.bool()]
            video_spread, video_spread_metrics = (
                self.unit_quantizer.projected_spread_loss(valid_video)
            )
            video_metrics.update(video_spread_metrics)
            student_spreads = [video_spread]
        geometry_metrics.update(usage_metrics)
        if physical_fused is not None:
            adapted_fused = self.fused_vq_adapter(physical_fused)
            q_fused, _, fused_metrics = self.unit_quantizer(
                adapted_fused,
                freeze_codebook=True,
            )
            branches["f"] = q_fused
            metrics["f"] = fused_metrics
            commitments.append(
                _masked_batch_mean(fused_metrics["commitment_per_sample"], sample_mask)
            )
            assignments.append(
                self._assignment_loss(
                    adapted_fused,
                    geometry_metrics["indices"],
                    sample_mask,
                )
            )
            soft_assignments.append(
                self._soft_assignment_loss(
                    adapted_fused,
                    adapted_geometry,
                    sample_mask,
                )
            )
            pre_vq_alignments.append(
                self._pre_vq_alignment_loss(
                    adapted_fused,
                    adapted_geometry,
                    sample_mask,
                )
            )
            valid_fused = adapted_fused[sample_mask.bool()]
            fused_spread, fused_spread_metrics = (
                self.unit_quantizer.projected_spread_loss(valid_fused)
            )
            fused_metrics.update(fused_spread_metrics)
            student_spreads.append(fused_spread)
        student_spread_loss = (
            torch.stack(student_spreads).mean()
            if student_spreads
            else zero
        )
        geometry_metrics["student_spread_loss"] = student_spread_loss.detach()
        # Quantize and supervise every modality against one immutable snapshot.
        # Geometry statistics are queued without changing the snapshot. The
        # Lightning optimizer hook commits them once after all accumulated
        # backwards, so every gradient in one optimizer step sees one codebook.
        if update_codebook:
            self.unit_quantizer.queue_codebook_update(
                adapted_geometry.detach(),
                geometry_metrics["indices"].detach(),
            )
        geometry_metrics["restarted_codes"] = (
            self.unit_quantizer.last_restarted_codes.detach().clone()
        )
        return (
            branches,
            torch.stack(commitments).mean(),
            torch.stack(assignments).mean() if assignments else zero,
            torch.stack(soft_assignments).mean() if soft_assignments else zero,
            (
                torch.stack(pre_vq_alignments).mean()
                if pre_vq_alignments
                else zero
            ),
            code_usage_loss,
            student_spread_loss,
            metrics,
        )

    def _branch_code_metrics(self, indices: Tensor) -> dict[str, Tensor]:
        output = {}
        for level in range(indices.shape[-1]):
            counts = torch.bincount(
                indices[..., level].reshape(-1),
                minlength=self.unit_quantizer.codebook_size,
            ).float()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(counts)
            probabilities = counts / counts.sum().clamp_min(1.0)
            output[f"active_q{level}"] = (counts > 0).float().mean()
            output[f"perplexity_q{level}"] = torch.exp(
                -(probabilities * (probabilities + 1e-12).log()).sum()
            )
        return output

    def forward(
        self,
        latents: list[Tensor] | Tensor,
        images: Tensor,
        frozen_diffusion: bool = False,
        denoising_terms: dict | None = None,
        return_unweighted: bool = False,
        batch_idx: int = 0,
        prepared_teacher_context: PreparedUnitTeacherContext | None = None,
        **_: object,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if self.unit_video_only_inference:
            raise RuntimeError(
                "The video-only UPCAST instance only supports "
                "encode_video_world_observation(), not training losses."
            )
        if self.unit_physical_only_continuation:
            raise RuntimeError(
                "The physical-only UPCAST instance only supports "
                "encode_generative_physical_tokens(), not reconstruction losses."
            )
        phase = "alignment" if self.unit_phase == "registration" else self.unit_phase
        geometry_warmup = phase == "geometry"
        alignment_phase = phase == "alignment"
        pre_joint = phase not in {"joint", "cotrain"}
        source_latents = (
            None if geometry_warmup else self._select_source_latents(latents)
        )
        teacher = (
            prepared_teacher_context.teacher
            if prepared_teacher_context is not None
            else self._extract_geometry_output(images)
        )
        if teacher.features is None:
            raise RuntimeError("UPCAST requires PAGE aggregated patch features")
        raw_points = teacher.target
        normalized_confidence = self._normalize_teacher_confidence(
            teacher.confidence,
            raw_points,
        )
        points = torch.nan_to_num(raw_points, nan=0.0, posinf=0.0, neginf=0.0)
        page_features = torch.nan_to_num(
            teacher.features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        batch_size, num_frames = points.shape[:2]
        if num_frames != 16:
            raise RuntimeError(f"UPCAST expected 16 aligned frames, got {num_frames}")
        if prepared_teacher_context is None:
            current, future, delta = self.sample_transition_pairs(
                batch_size,
                num_frames,
                points.device,
                training=self.training,
            )
        else:
            current = prepared_teacher_context.current
            future = prepared_teacher_context.future
            delta = prepared_teacher_context.delta
            expected_pair_shape = (batch_size,)
            if (
                current.shape != expected_pair_shape
                or future.shape != expected_pair_shape
                or delta.shape != expected_pair_shape
            ):
                raise ValueError(
                    "Prepared teacher frame-pair shapes do not match batch size: "
                    f"current={tuple(current.shape)}, future={tuple(future.shape)}, "
                    f"delta={tuple(delta.shape)}, batch={batch_size}"
                )
        logsnr = self._frame_logsnr(denoising_terms, batch_size, num_frames, points.device)
        current_logsnr = _gather_bt(logsnr, current)
        future_logsnr = _gather_bt(logsnr, future)
        raw_signal = self._raw_transition_signal(logsnr, current, future)
        sample_mask = (raw_signal >= self.unit_noise_signal_threshold).to(raw_signal)

        geometry_current, geometry_future = self.geometry_pair_projector(page_features, points, current, future)
        physical_geometry = self.geometry_transition_encoder(
            geometry_current,
            geometry_future,
            delta,
        )
        physical_video = None
        appearance_video = None
        appearance_semantic = None
        physical_fused = None
        appearance_fused = None
        current_images = None
        current_dino = None
        target_dino = None
        if not geometry_warmup:
            assert source_latents is not None
            video_current, video_future = self._run_frozen_unit_checkpoint(
                lambda *selected_latents: self.video_pair_projector(
                    list(selected_latents),
                    current,
                    future,
                ),
                *source_latents,
            )
            physical_video = self._run_frozen_unit_checkpoint(
                self.video_transition_encoder,
                video_current,
                video_future,
                delta,
                current_logsnr,
                future_logsnr,
            )
            current_images = gather_frames(
                images[:, : self.alignment_context_length : self.keyframe_stride],
                current,
            )
            future_images = gather_frames(
                images[:, : self.alignment_context_length : self.keyframe_stride],
                future,
            )
            # DINOv2 is frame-independent, while UPCAST only consumes the sampled
            # current/future pair.
            pair_dino = (
                prepared_teacher_context.pair_dino
                if prepared_teacher_context is not None
                else torch.nan_to_num(
                    self._extract_dino(
                        torch.stack((current_images, future_images), dim=1)
                    ),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )
            current_dino, target_dino = pair_dino[:, 0], pair_dino[:, 1]
            pair_current = torch.zeros(
                batch_size,
                device=points.device,
                dtype=torch.long,
            )
            pair_future = torch.ones_like(pair_current)
            semantic_current, semantic_future = self.semantic_pair_projector(
                pair_dino,
                pair_current,
                pair_future,
            )
            appearance_video = self._run_frozen_unit_checkpoint(
                self.video_appearance_encoder,
                video_current,
                video_future,
                delta,
                current_logsnr,
                future_logsnr,
            )
            appearance_semantic = self.semantic_transition_encoder(
                semantic_current,
                semantic_future,
                delta,
            )
            if alignment_phase:
                physical_fused = self.fusion_transition_encoder(
                    physical_video,
                    physical_geometry,
                )
                appearance_fused = self.appearance_fusion_encoder(
                    appearance_video,
                    appearance_semantic,
                )

        (
            physical_branches,
            rvq_loss,
            code_assignment_loss,
            soft_code_loss,
            pre_vq_alignment_loss,
            code_usage_loss,
            student_spread_loss,
            rvq_metrics,
        ) = (
            self._quantize_physical_transitions(
                physical_video,
                physical_geometry,
                physical_fused,
                sample_mask,
            )
        )
        q_geometry = physical_branches["g"]
        q_video = physical_branches.get("v")
        state_teacher_physical = (
            physical_branches.get("f") if alignment_phase else None
        )
        state_teacher_appearance = appearance_fused if alignment_phase else None
        if (
            not geometry_warmup
            and not alignment_phase
            and physical_video is not None
            and appearance_video is not None
            and appearance_semantic is not None
        ):
            # q_f remains a stationary PAGE/DINO teacher after bootstrap. It is
            # recomputed without a graph, so joint training cannot move the
            # teacher or pay for decoder/tail branches that are not used.
            with torch.no_grad():
                teacher_physical_pre_vq = self.fusion_transition_encoder(
                    physical_video.detach(),
                    physical_geometry.detach(),
                )
                state_teacher_appearance = self.appearance_fusion_encoder(
                    appearance_video.detach(),
                    appearance_semantic.detach(),
                )
                state_teacher_physical, _, _ = self.unit_quantizer(
                    self.fused_vq_adapter(teacher_physical_pre_vq),
                    freeze_codebook=True,
                )
        world_video = (
            self.world_token_fusion(q_video, appearance_video)
            if q_video is not None and appearance_video is not None
            else None
        )
        world_geometry_semantic = (
            self.world_token_fusion(q_geometry, appearance_semantic)
            if appearance_semantic is not None
            else None
        )
        world_fused = (
            self.world_token_fusion(physical_branches["f"], appearance_fused)
            if alignment_phase and appearance_fused is not None
            else None
        )
        world_state_distill = points.new_zeros(())
        if (
            q_video is not None
            and appearance_video is not None
            and state_teacher_physical is not None
            and state_teacher_appearance is not None
        ):
            physical_per_sample = 1.0 - (
                F.normalize(q_video.float(), dim=-1, eps=1e-6)
                * F.normalize(
                    state_teacher_physical.detach().float(),
                    dim=-1,
                    eps=1e-6,
                )
            ).sum(dim=-1).mean(dim=1)
            appearance_per_sample = 1.0 - (
                F.normalize(appearance_video.float(), dim=-1, eps=1e-6)
                * F.normalize(
                    state_teacher_appearance.detach().float(),
                    dim=-1,
                    eps=1e-6,
                )
            ).sum(dim=-1).mean(dim=1)
            world_state_distill = _masked_batch_mean(
                physical_per_sample + 0.25 * appearance_per_sample,
                sample_mask,
            )
            self.last_world_state_observations = {
                "video": {
                    "physical": q_video.detach(),
                    "appearance": appearance_video.detach(),
                },
                "fused": {
                    "physical": state_teacher_physical.detach(),
                    "appearance": state_teacher_appearance.detach(),
                },
            }
        else:
            self.last_world_state_observations = {}

        aligned_images = images[:, : self.alignment_context_length : self.keyframe_stride]
        if prepared_teacher_context is None:
            current_page, current_points = self._extract_geometry_anchor(
                aligned_images,
                current,
            )
        else:
            current_page = prepared_teacher_context.current_page
            current_points = prepared_teacher_context.current_points
        visual_anchor = (
            self.visual_anchor_encoder(current_images, current_dino)
            if current_images is not None and current_dino is not None
            else None
        )
        geometry_anchor = self.geometry_anchor_encoder(current_page, current_points)
        if visual_anchor is not None:
            (
                visual_anchor,
                visual_anchor_drop,
                visual_anchor_sample_drop,
            ) = self.visual_anchor_encoder.condition_dropout(
                visual_anchor,
                self.unit_anchor_dropout,
                self.unit_anchor_sample_dropout,
                enabled=bool(pre_joint and self.training),
            )
        else:
            visual_anchor_drop = points.new_zeros(())
            visual_anchor_sample_drop = points.new_zeros(())
        (
            geometry_anchor,
            geometry_anchor_drop,
            geometry_anchor_sample_drop,
        ) = self.geometry_anchor_encoder.condition_dropout(
            geometry_anchor,
            self.unit_anchor_dropout,
            self.unit_anchor_sample_dropout,
            enabled=bool(pre_joint and self.training),
        )
        target_page = gather_frames(page_features, future)
        target_points = gather_frames(points, future)
        target_confidence = gather_frames(normalized_confidence[:, :, None], future)[:, 0]

        visual_total = points.new_zeros(())
        geometry_total = points.new_zeros(())
        tail_total = points.new_zeros(())
        visual_denominator = geometry_denominator = tail_denominator = 0.0
        tail_valid = points.new_zeros(())
        branch_metrics: dict[str, Tensor] = {}
        captured_latent = None if source_latents is None else source_latents[0]

        visual_inputs: dict[str, tuple[Tensor, Tensor | None]] = {}
        geometry_inputs: dict[str, tuple[Tensor, Tensor | None]] = {}
        tail_names: set[str] = set()
        tail_enabled = bool(
            self.unit_tail_fm_loss_weight > 0
            if alignment_phase
            else self.unit_joint_tail_fm_loss_weight > 0
        )
        if geometry_warmup:
            geometry_inputs["g"] = (q_geometry, None)
        else:
            assert (
                world_video is not None
                and q_video is not None
                and captured_latent is not None
                and visual_anchor is not None
                and target_dino is not None
            )
            visual_inputs["v"] = (world_video, sample_mask)
            geometry_inputs["v"] = (q_video, sample_mask)
            if tail_enabled:
                tail_names.add("v")
        if alignment_phase:
            assert world_geometry_semantic is not None and world_fused is not None
            visual_inputs["gs"] = (world_geometry_semantic, None)
            visual_inputs["f"] = (world_fused, sample_mask)
            geometry_inputs["g"] = (q_geometry, None)
            geometry_inputs["f"] = (physical_branches["f"], sample_mask)
            if tail_enabled:
                tail_names.add("f")

        visual_per_sample: dict[str, Tensor] = {}
        geometry_per_sample: dict[str, Tensor] = {}
        for name, (world_tokens, branch_mask) in visual_inputs.items():
            pred_dino, pred_latent = self._run_frozen_unit_checkpoint(
                self.visual_decoder,
                world_tokens,
                visual_anchor,
                delta,
            )
            visual_loss, per_sample = self._visual_loss(
                pred_dino,
                target_dino,
                branch_mask,
                return_per_sample=True,
            )
            visual_per_sample[name] = per_sample
            visual_total = visual_total + visual_loss
            visual_denominator += 1.0
            branch_metrics[f"alignment_unit_visual_{name}"] = visual_loss.detach()
            if name in tail_names:
                tail_loss, valid = self._tail_fm_loss(
                    pred_latent,
                    captured_latent,
                    future,
                    branch_mask,
                    denoising_terms,
                    validate_source=bool(frozen_diffusion),
                )
                tail_total = tail_total + tail_loss
                tail_denominator += 1.0
                tail_valid = torch.maximum(tail_valid, valid)
                branch_metrics[f"alignment_unit_tail_fm_{name}"] = tail_loss.detach()

        for name, (physical_tokens, branch_mask) in geometry_inputs.items():
            pred_page, pred_points, pred_confidence = self._run_frozen_unit_checkpoint(
                self.geometry_decoder,
                physical_tokens,
                geometry_anchor,
                delta,
            )
            geometry_loss, components, per_sample = self._geometry_loss(
                pred_page,
                pred_points,
                pred_confidence,
                target_page,
                target_points,
                target_confidence,
                branch_mask,
                return_per_sample=True,
            )
            geometry_per_sample[name] = per_sample
            geometry_total = geometry_total + geometry_loss
            geometry_denominator += 1.0
            branch_metrics[f"alignment_unit_geometry_{name}"] = geometry_loss.detach()
            for component_name, component in components.items():
                branch_metrics[
                    f"alignment_unit_geometry_{name}_{component_name}"
                ] = component.detach()

        visual_total = visual_total / max(visual_denominator, 1e-8)
        geometry_total = geometry_total / max(geometry_denominator, 1e-8)
        tail_total = tail_total / max(tail_denominator, 1e-8)

        q_only_loss = points.new_zeros(())
        q_only_visual = points.new_zeros(())
        q_only_geometry = points.new_zeros(())
        q_only_computed = points.new_zeros(())
        should_compute_q_only = bool(
            pre_joint
            and self.training
            and int(batch_idx) % self.unit_q_only_loss_freq
            == self.unit_q_only_loss_phase
        )
        if should_compute_q_only:
            zero_geometry_anchor = torch.zeros_like(geometry_anchor)
            q_only_visual_parts = []
            q_only_geometry_parts = []
            if alignment_phase:
                assert (
                    visual_anchor is not None
                    and world_video is not None
                    and world_fused is not None
                    and target_dino is not None
                    and q_video is not None
                )
                zero_visual_anchor = torch.zeros_like(visual_anchor)
                pred_dino, _ = self.visual_decoder(
                    world_video,
                    zero_visual_anchor,
                    delta,
                )
                q_only_visual_parts.append(
                    self._visual_loss(
                        pred_dino,
                        target_dino,
                        sample_mask,
                    )
                )
                # q_v is the branch ultimately distilled into Video DiT. q_f
                # already receives full-sample anchor dropout and an explicit
                # geometry residual; duplicating both q-only decoder graphs
                # leaves too little headroom on 40 GB devices.
                geometry_tokens = (q_video,)
            else:
                geometry_tokens = (q_geometry,)
            for physical_tokens in geometry_tokens:
                pred_page, pred_points, pred_confidence = self.geometry_decoder(
                    physical_tokens,
                    zero_geometry_anchor,
                    delta,
                )
                geometry_loss, _ = self._geometry_loss(
                    pred_page,
                    pred_points,
                    pred_confidence,
                    target_page,
                    target_points,
                    target_confidence,
                    sample_mask,
                )
                q_only_geometry_parts.append(geometry_loss)
            if q_only_visual_parts:
                q_only_visual = torch.stack(q_only_visual_parts).mean()
            q_only_geometry = torch.stack(q_only_geometry_parts).mean()
            q_only_parts = [q_only_geometry]
            if q_only_visual_parts:
                q_only_parts.append(q_only_visual)
            q_only_loss = torch.stack(q_only_parts).mean()
            q_only_loss = q_only_loss * self.unit_q_only_loss_freq
            q_only_computed = points.new_ones(())

        physical_memory = self._contrastive_memory("physical")
        appearance_memory = self._contrastive_memory("appearance")
        if geometry_warmup:
            physical_contrast = points.new_zeros(())
            appearance_contrast = points.new_zeros(())
        else:
            assert (
                physical_video is not None
                and appearance_video is not None
                and appearance_semantic is not None
            )
            physical_contrast_parts = [
                self._contrastive_loss(
                    physical_video,
                    physical_geometry,
                    sample_mask,
                    physical_memory,
                )
            ]
            appearance_contrast_parts = [
                self._contrastive_loss(
                    appearance_video,
                    appearance_semantic,
                    sample_mask,
                    appearance_memory,
                )
            ]
            if alignment_phase:
                assert physical_fused is not None and appearance_fused is not None
                physical_contrast_parts.append(
                    self._contrastive_loss(
                        physical_fused,
                        physical_geometry,
                        sample_mask,
                        physical_memory,
                    )
                )
                appearance_contrast_parts.append(
                    self._contrastive_loss(
                        appearance_fused,
                        appearance_semantic,
                        sample_mask,
                        appearance_memory,
                    )
                )
            physical_contrast = torch.stack(physical_contrast_parts).mean()
            appearance_contrast = torch.stack(appearance_contrast_parts).mean()

        variance_loss = points.new_zeros(())
        if geometry_warmup:
            variance_terms = [self._variance_loss(physical_geometry)]
            variance_loss = torch.stack(variance_terms).mean()
        elif alignment_phase:
            variance_terms = [
                self._variance_loss(physical_video, sample_mask),
                self._variance_loss(appearance_video, sample_mask),
                self._variance_loss(appearance_semantic),
                self._variance_loss(physical_fused, sample_mask),
                self._variance_loss(appearance_fused, sample_mask),
            ]
            variance_loss = torch.stack(variance_terms).mean()

        shuffle_loss = points.new_zeros(())
        shuffle_visual_gap = points.new_zeros(())
        shuffle_visual_physical_gap = points.new_zeros(())
        shuffle_visual_appearance_gap = points.new_zeros(())
        shuffle_geometry_gap = points.new_zeros(())
        shuffle_computed = points.new_zeros(())
        should_shuffle = bool(
            alignment_phase
            and self.training
            and int(batch_idx) % self.unit_shuffle_loss_freq
            == self.unit_shuffle_loss_phase
        )
        if should_shuffle:
            shuffled_q_video = self._shuffle_transition_tokens(
                q_video,
                self.unit_spatial_grid,
            )
            shuffled_appearance_video = self._shuffle_transition_tokens(
                appearance_video,
                self.unit_spatial_grid,
            )
            physical_wrong_world = self.world_token_fusion(
                shuffled_q_video,
                appearance_video,
            )
            appearance_wrong_world = self.world_token_fusion(
                q_video,
                shuffled_appearance_video,
            )
            physical_wrong_dino, _ = self.visual_decoder(
                physical_wrong_world,
                visual_anchor,
                delta,
            )
            appearance_wrong_dino, _ = self.visual_decoder(
                appearance_wrong_world,
                visual_anchor,
                delta,
            )
            _, wrong_visual_physical = self._visual_loss(
                physical_wrong_dino,
                target_dino,
                sample_mask,
                return_per_sample=True,
            )
            _, wrong_visual_appearance = self._visual_loss(
                appearance_wrong_dino,
                target_dino,
                sample_mask,
                return_per_sample=True,
            )
            wrong_page, wrong_points, wrong_confidence = self.geometry_decoder(
                shuffled_q_video,
                geometry_anchor,
                delta,
            )
            _, _, wrong_geometry = self._geometry_loss(
                wrong_page,
                wrong_points,
                wrong_confidence,
                target_page,
                target_points,
                target_confidence,
                sample_mask,
                return_per_sample=True,
            )
            correct_visual = visual_per_sample["v"]
            correct_geometry = geometry_per_sample["v"]
            visual_physical_hinge = _masked_batch_mean(
                F.relu(
                    self.unit_shuffle_margin
                    + correct_visual
                    - wrong_visual_physical
                ),
                sample_mask,
            )
            visual_appearance_hinge = _masked_batch_mean(
                F.relu(
                    self.unit_shuffle_margin
                    + correct_visual
                    - wrong_visual_appearance
                ),
                sample_mask,
            )
            geometry_hinge = _masked_batch_mean(
                F.relu(self.unit_shuffle_margin + correct_geometry - wrong_geometry),
                sample_mask,
            )
            shuffle_visual_physical_gap = _masked_batch_mean(
                wrong_visual_physical - correct_visual,
                sample_mask,
            )
            shuffle_visual_appearance_gap = _masked_batch_mean(
                wrong_visual_appearance - correct_visual,
                sample_mask,
            )
            shuffle_visual_gap = 0.5 * (
                shuffle_visual_physical_gap
                + shuffle_visual_appearance_gap
            )
            shuffle_geometry_gap = _masked_batch_mean(
                wrong_geometry - correct_geometry,
                sample_mask,
            )
            # Preserve the configured expected weight under sparse evaluation.
            shuffle_loss = (
                visual_physical_hinge
                + visual_appearance_hinge
                + geometry_hinge
            ) / 3.0
            shuffle_loss = shuffle_loss * self.unit_shuffle_loss_freq
            shuffle_computed = points.new_ones(())

        if geometry_warmup:
            active_weights = {
                "visual": 0.0,
                "geometry": self.unit_geometry_loss_weight,
                "tail": 0.0,
                "rvq": self.unit_rvq_loss_weight,
                "assignment": 0.0,
                "soft_code": 0.0,
                "pre_vq": 0.0,
                "code_usage": self.unit_code_usage_loss_weight,
                "student_spread": 0.0,
                "physical_contrast": 0.0,
                "appearance_contrast": 0.0,
                "variance": self.unit_variance_loss_weight,
                "shuffle": 0.0,
                "q_only": self.unit_q_only_loss_weight,
                "world_state": 0.0,
            }
        elif alignment_phase:
            active_weights = {
                "visual": self.unit_visual_loss_weight,
                "geometry": self.unit_geometry_loss_weight,
                "tail": self.unit_tail_fm_loss_weight,
                "rvq": self.unit_rvq_loss_weight,
                "assignment": self.unit_code_assignment_loss_weight,
                "soft_code": self.unit_soft_code_loss_weight,
                "pre_vq": self.unit_pre_vq_alignment_loss_weight,
                "code_usage": 0.0,
                "student_spread": self.unit_student_spread_loss_weight,
                "physical_contrast": self.unit_physical_contrast_loss_weight,
                "appearance_contrast": self.unit_appearance_contrast_loss_weight,
                "variance": self.unit_variance_loss_weight,
                "shuffle": self.unit_shuffle_loss_weight,
                "q_only": self.unit_q_only_loss_weight,
                "world_state": self.unit_world_state_distill_loss_weight,
            }
        else:
            active_weights = {
                "visual": self.unit_joint_visual_loss_weight,
                "geometry": self.unit_joint_geometry_loss_weight,
                "tail": self.unit_joint_tail_fm_loss_weight,
                "rvq": self.unit_joint_rvq_loss_weight,
                "assignment": self.unit_joint_code_assignment_loss_weight,
                "soft_code": self.unit_joint_soft_code_loss_weight,
                "pre_vq": self.unit_joint_pre_vq_alignment_loss_weight,
                "code_usage": 0.0,
                "student_spread": 0.0,
                "physical_contrast": self.unit_joint_physical_contrast_loss_weight,
                "appearance_contrast": self.unit_joint_appearance_contrast_loss_weight,
                "variance": 0.0,
                "shuffle": 0.0,
                "q_only": 0.0,
                "world_state": self.unit_joint_world_state_distill_loss_weight,
            }
        weighted_components = {
            "visual": active_weights["visual"] * visual_total,
            "geometry": active_weights["geometry"] * geometry_total,
            "tail": active_weights["tail"] * tail_total,
            "rvq": active_weights["rvq"] * rvq_loss,
            "assignment": active_weights["assignment"] * code_assignment_loss,
            "soft_code": active_weights["soft_code"] * soft_code_loss,
            "pre_vq": active_weights["pre_vq"] * pre_vq_alignment_loss,
            "code_usage": active_weights["code_usage"] * code_usage_loss,
            "student_spread": active_weights["student_spread"] * student_spread_loss,
            "physical_contrast": active_weights["physical_contrast"] * physical_contrast,
            "appearance_contrast": active_weights["appearance_contrast"] * appearance_contrast,
            "variance": active_weights["variance"] * variance_loss,
            "shuffle": active_weights["shuffle"] * shuffle_loss,
            "q_only": active_weights["q_only"] * q_only_loss,
            "world_state": active_weights["world_state"] * world_state_distill,
        }
        loss = torch.stack(tuple(weighted_components.values())).sum()
        geometry_code_indices = rvq_metrics["g"]["indices"]
        video_code_indices = (
            rvq_metrics["v"]["indices"] if "v" in rvq_metrics else None
        )
        video_geometry_code_agreement = (
            (video_code_indices == geometry_code_indices).float().mean()
            if video_code_indices is not None
            else points.new_zeros(())
        )
        physical_retrieval = (
            self._retrieval_accuracy(
                physical_video,
                physical_geometry,
                physical_memory,
            )
            if physical_video is not None
            else points.new_zeros(())
        )
        appearance_retrieval = (
            self._retrieval_accuracy(
                appearance_video,
                appearance_semantic,
                appearance_memory,
            )
            if appearance_video is not None and appearance_semantic is not None
            else points.new_zeros(())
        )
        self.last_metrics = {
            "alignment_loss": loss.detach(),
            "alignment_loss_unweighted": loss.detach(),
            "alignment_unit_visual_cross_loss": visual_total.detach(),
            "alignment_unit_geometry_cross_loss": geometry_total.detach(),
            "alignment_unit_tail_fm_loss": tail_total.detach(),
            "alignment_unit_tail_fm_valid": tail_valid.detach(),
            "alignment_unit_rvq_commitment_loss": rvq_loss.detach(),
            "alignment_unit_code_assignment_loss": code_assignment_loss.detach(),
            "alignment_unit_soft_code_distillation_loss": soft_code_loss.detach(),
            "alignment_unit_pre_vq_alignment_loss": pre_vq_alignment_loss.detach(),
            "alignment_unit_code_usage_loss": code_usage_loss.detach(),
            "alignment_unit_student_spread_loss": student_spread_loss.detach(),
            "alignment_unit_physical_contrast_loss": physical_contrast.detach(),
            "alignment_unit_appearance_contrast_loss": appearance_contrast.detach(),
            "alignment_unit_variance_loss": variance_loss.detach(),
            "alignment_unit_shuffle_ranking_loss": shuffle_loss.detach(),
            "alignment_unit_q_only_loss": q_only_loss.detach(),
            "alignment_unit_q_only_visual_loss": q_only_visual.detach(),
            "alignment_unit_q_only_geometry_loss": q_only_geometry.detach(),
            "alignment_unit_world_state_distill_loss": world_state_distill.detach(),
            "alignment_unit_q_only_computed": q_only_computed.detach(),
            "alignment_unit_shuffle_visual_gap": shuffle_visual_gap.detach(),
            "alignment_unit_shuffle_visual_physical_gap": shuffle_visual_physical_gap.detach(),
            "alignment_unit_shuffle_visual_appearance_gap": shuffle_visual_appearance_gap.detach(),
            "alignment_unit_shuffle_geometry_gap": shuffle_geometry_gap.detach(),
            "alignment_unit_shuffle_computed": shuffle_computed.detach(),
            "alignment_unit_visual_anchor_drop_fraction": visual_anchor_drop.detach(),
            "alignment_unit_visual_anchor_sample_drop_fraction": (
                visual_anchor_sample_drop.detach()
            ),
            "alignment_unit_geometry_anchor_drop_fraction": geometry_anchor_drop.detach(),
            "alignment_unit_geometry_anchor_sample_drop_fraction": (
                geometry_anchor_sample_drop.detach()
            ),
            "alignment_unit_signal_raw_mean": raw_signal.mean().detach(),
            "alignment_unit_signal_mean": (raw_signal * sample_mask).mean().detach(),
            "alignment_unit_signal_valid_fraction": sample_mask.mean().detach(),
            "alignment_unit_signal_retained_mass": (
                (raw_signal * sample_mask).sum() / raw_signal.sum().clamp_min(1e-8)
            ).detach(),
            "alignment_unit_pair_delta_mean": delta.float().mean().detach(),
            "alignment_unit_pair_short_fraction": ((delta >= 1) & (delta <= 3)).float().mean().detach(),
            "alignment_unit_pair_medium_fraction": ((delta >= 4) & (delta <= 8)).float().mean().detach(),
            "alignment_unit_pair_long_fraction": ((delta >= 9) & (delta <= 15)).float().mean().detach(),
            "alignment_geometry_confidence_mean": target_confidence.mean().detach(),
            "alignment_unit_active_code_ratio": rvq_metrics["g"]["active_code_ratio"].mean().detach(),
            "alignment_unit_perplexity": rvq_metrics["g"]["perplexity"].mean().detach(),
            "alignment_unit_soft_perplexity": rvq_metrics["g"]["soft_perplexity"].mean().detach(),
            "alignment_unit_projected_std": rvq_metrics["g"]["projected_std"].detach(),
            "alignment_unit_projected_spread_loss": rvq_metrics["g"]["projected_spread_loss"].detach(),
            "alignment_unit_geometry_active_code_ratio": rvq_metrics["g"]["active_code_ratio"].mean().detach(),
            "alignment_unit_geometry_perplexity": rvq_metrics["g"]["perplexity"].mean().detach(),
            "alignment_unit_geometry_soft_perplexity": rvq_metrics["g"]["soft_perplexity"].mean().detach(),
            "alignment_unit_geometry_projected_std": rvq_metrics["g"]["projected_std"].detach(),
            "alignment_unit_restarted_codes": rvq_metrics["g"]["restarted_codes"].sum().float().detach(),
            "alignment_unit_codebook_update_count": self.unit_quantizer.ema_update_count.float().detach().clone(),
            "alignment_unit_codebook_pending_microbatches": self.unit_quantizer.pending_microbatches.float().detach().clone(),
            "alignment_unit_codebook_window_microbatches": self.unit_quantizer.last_committed_microbatches.float().detach().clone(),
            "alignment_unit_codebook_window_global_tokens": self.unit_quantizer.last_committed_global_tokens.float().detach().clone(),
            "alignment_unit_video_geometry_code_agreement": (
                video_geometry_code_agreement.detach()
            ),
            "alignment_unit_physical_retrieval_top1": physical_retrieval.detach(),
            "alignment_unit_appearance_retrieval_top1": appearance_retrieval.detach(),
            "alignment_unit_physical_memory_fill_ratio": points.new_tensor(
                physical_memory.shape[0]
                / max(self.unit_contrastive_memory_size, 1)
            ),
            "alignment_unit_appearance_memory_fill_ratio": points.new_tensor(
                appearance_memory.shape[0]
                / max(self.unit_contrastive_memory_size, 1)
            ),
            "alignment_unit_codebook_frozen": points.new_tensor(
                float(not (geometry_warmup and self.training))
            ),
            "alignment_unit_geometry_phase": points.new_tensor(float(geometry_warmup)),
            "alignment_unit_alignment_phase": points.new_tensor(float(alignment_phase)),
            "alignment_unit_joint_phase": points.new_tensor(
                float(phase in {"joint", "cotrain"})
            ),
            "alignment_unit_cotrain_phase": points.new_tensor(
                float(phase == "cotrain")
            ),
            "alignment_unit_bootstrap_phase": points.new_tensor(
                float(geometry_warmup or alignment_phase)
            ),
            "alignment_unit_diffusion_frozen": points.new_tensor(float(frozen_diffusion)),
            "alignment_unit_arch_version": points.new_tensor(float(UNIT_ARCH_VERSION)),
        }
        for component_name, component_value in weighted_components.items():
            self.last_metrics[
                f"alignment_unit_{component_name}_weighted"
            ] = component_value.detach()
        if "f" in rvq_metrics:
            self.last_metrics["alignment_unit_fused_geometry_code_agreement"] = (
                rvq_metrics["f"]["indices"] == geometry_code_indices
            ).float().mean().detach()
        for level, value in enumerate(rvq_metrics["g"]["active_code_ratio"]):
            self.last_metrics[f"alignment_unit_active_code_ratio_q{level}"] = value.detach()
        for level, value in enumerate(rvq_metrics["g"]["perplexity"]):
            self.last_metrics[f"alignment_unit_perplexity_q{level}"] = value.detach()
        for level, value in enumerate(rvq_metrics["g"]["soft_perplexity"]):
            self.last_metrics[f"alignment_unit_soft_perplexity_q{level}"] = value.detach()
        for level, value in enumerate(rvq_metrics["g"]["restarted_codes"]):
            self.last_metrics[f"alignment_unit_restarted_codes_q{level}"] = value.float().detach()
        for branch_name, branch_values in rvq_metrics.items():
            for metric_name, value in self._branch_code_metrics(
                branch_values["indices"]
            ).items():
                self.last_metrics[
                    f"alignment_unit_{branch_name}_{metric_name}"
                ] = value.detach()
            for metric_name in ("projected_std", "projected_spread_loss"):
                if metric_name in branch_values:
                    self.last_metrics[
                        f"alignment_unit_{branch_name}_{metric_name}"
                    ] = branch_values[metric_name].detach()
        student_branches = [
            rvq_metrics[name]
            for name in ("v", "f")
            if name in rvq_metrics
        ]
        if student_branches:
            student_active = torch.stack(
                [value["active_code_ratio"].mean() for value in student_branches]
            )
            student_perplexity = torch.stack(
                [value["perplexity"].mean() for value in student_branches]
            )
            self.last_metrics["alignment_unit_student_active_code_ratio"] = (
                student_active.mean().detach()
            )
            self.last_metrics["alignment_unit_student_min_perplexity"] = (
                student_perplexity.min().detach()
            )
        self.last_metrics.update(branch_metrics)
        if self.training and not geometry_warmup:
            assert appearance_semantic is not None
            self._enqueue_contrastive_memory(
                "physical",
                physical_geometry,
                sample_mask,
            )
            self._enqueue_contrastive_memory(
                "appearance",
                appearance_semantic,
                sample_mask,
            )
        return (loss, loss.detach()) if return_unweighted else loss

    def _noise_frame_weights(
        self,
        logsnr: Tensor | None,
        batch_size: int,
        num_frames: int,
        device,
    ) -> Tensor:
        if logsnr is None:
            return torch.ones(batch_size, num_frames, device=device)
        weights = torch.sigmoid(logsnr[:, :num_frames].to(device=device).float())
        return torch.where(
            weights >= self.generation_noise_signal_threshold,
            weights,
            torch.zeros_like(weights),
        )

    @staticmethod
    def _weighted_map_loss(values: Tensor, weights: Tensor) -> Tensor:
        while weights.ndim < values.ndim:
            weights = weights.unsqueeze(-1)
        expanded = weights.expand_as(values)
        return (values * expanded).sum() / expanded.ne(0).sum().clamp_min(1)

    def compute_alignment_generation_loss(
        self,
        xs_pred: Tensor,
        xs_target: Tensor | None = None,
        logsnr: Tensor | None = None,
        prepared_targets: PreparedGenerationTargets | None = None,
    ) -> tuple[Tensor, Tensor]:
        zero = xs_pred.sum() * 0.0
        metric_names = (
            "alignment_generation_dino_patch_loss_component",
            "alignment_generation_vjepa_spatial_loss_component",
            "alignment_generation_vjepa_temporal_loss_component",
            "alignment_generation_vjepa_temporal_direction_component",
            "alignment_generation_vjepa_temporal_magnitude_component",
            "alignment_generation_vjepa_global_loss_component",
            "alignment_generation_color_moment_loss_component",
            "alignment_generation_rgb_temporal_loss_component",
            "alignment_generation_rgb_temporal_direction_component",
            "alignment_generation_rgb_temporal_magnitude_component",
            "alignment_generation_dino_patch_loss_weighted",
            "alignment_generation_vjepa_spatial_loss_weighted",
            "alignment_generation_vjepa_temporal_loss_weighted",
            "alignment_generation_vjepa_global_loss_weighted",
            "alignment_generation_color_moment_loss_weighted",
            "alignment_generation_rgb_temporal_loss_weighted",
        )
        self.last_generation_metrics = {name: zero.detach() for name in metric_names}
        if xs_target is None:
            return zero, zero.detach()
        pred = xs_pred[:, : self.alignment_context_length : self.keyframe_stride]
        target = xs_target[:, : self.alignment_context_length : self.keyframe_stride].to(pred)
        num_frames = min(pred.shape[1], target.shape[1])
        pred, target = pred[:, :num_frames], target[:, :num_frames]
        if (
            prepared_targets is not None
            and prepared_targets.num_frames != num_frames
        ):
            raise ValueError(
                "Prepared generation target length does not match prediction: "
                f"{prepared_targets.num_frames} != {num_frames}"
            )
        frame_weights = self._noise_frame_weights(logsnr, pred.shape[0], num_frames, pred.device)

        pred_float = pred.float()
        target_float = target.detach().float()
        color_moment = zero
        if self.generation_color_moment_loss_weight > 0:
            pred_mean = pred_float.mean(dim=(-2, -1))
            target_mean = target_float.mean(dim=(-2, -1))
            pred_std = pred_float.std(dim=(-2, -1), unbiased=False)
            target_std = target_float.std(dim=(-2, -1), unbiased=False)
            color_map = F.smooth_l1_loss(
                pred_mean,
                target_mean,
                beta=0.05,
                reduction="none",
            ).mean(dim=-1) + F.smooth_l1_loss(
                pred_std,
                target_std,
                beta=0.05,
                reduction="none",
            ).mean(dim=-1)
            color_moment = self._weighted_map_loss(color_map, frame_weights)

        rgb_temporal = zero
        rgb_temporal_direction = zero
        rgb_temporal_magnitude = zero
        if self.generation_rgb_temporal_loss_weight > 0 and num_frames > 1:
            pred_delta = (pred_float[:, 1:] - pred_float[:, :-1]).flatten(2)
            target_delta = (target_float[:, 1:] - target_float[:, :-1]).flatten(2)
            pred_magnitude = pred_delta.norm(dim=-1)
            target_magnitude = target_delta.norm(dim=-1)
            direction_map = 1.0 - F.cosine_similarity(
                pred_delta,
                target_delta,
                dim=-1,
                eps=1e-6,
            )
            magnitude_map = (
                torch.log(pred_magnitude + 1e-4)
                - torch.log(target_magnitude + 1e-4)
            ).abs().clamp_max(5.0)
            temporal_weights = torch.minimum(
                frame_weights[:, 1:],
                frame_weights[:, :-1],
            )
            motion_mask = (target_magnitude > 1e-3).to(temporal_weights)
            rgb_temporal_direction = self._weighted_map_loss(
                direction_map,
                temporal_weights * motion_mask,
            )
            rgb_temporal_magnitude = self._weighted_map_loss(
                magnitude_map,
                temporal_weights,
            )
            rgb_temporal = (
                rgb_temporal_direction + 0.1 * rgb_temporal_magnitude
            )

        dino_loss = zero
        if self.generation_dino_patch_loss_weight > 0:
            if prepared_targets is not None:
                target_dino = prepared_targets.dino
                if target_dino is None:
                    raise RuntimeError(
                        "Prepared generation targets are missing DINO features"
                    )
            else:
                with torch.no_grad():
                    target_dino = self.dino_extractor(
                        target.detach(),
                        allow_input_grad=False,
                    )
            pred_dino = self.dino_extractor(pred, allow_input_grad=True)
            target_dino = target_dino.to(pred_dino)
            dino_map = 1.0 - (
                F.normalize(pred_dino.float(), dim=2, eps=1e-6)
                * F.normalize(target_dino.float(), dim=2, eps=1e-6)
            ).sum(dim=2)
            dino_loss = self._weighted_map_loss(dino_map, frame_weights)

        vjepa_spatial = vjepa_temporal = vjepa_global = zero
        vjepa_temporal_direction = vjepa_temporal_magnitude = zero
        if self.generation_video_extractor is not None:
            if prepared_targets is not None:
                target_tokens = prepared_targets.vjepa
                if target_tokens is None:
                    raise RuntimeError(
                        "Prepared generation targets are missing V-JEPA features"
                    )
            else:
                with torch.no_grad():
                    target_tokens = self.generation_video_extractor(
                        target.detach(),
                        allow_input_grad=False,
                    )
            pred_tokens = self.generation_video_extractor(pred, allow_input_grad=True)
            target_tokens = target_tokens.to(pred_tokens)
            temporal_tokens = pred_tokens.shape[1]
            tubelet_weights = F.adaptive_avg_pool1d(frame_weights[:, None], temporal_tokens).squeeze(1)

            spatial_map = 1.0 - (pred_tokens * target_tokens).sum(dim=-1)
            vjepa_spatial = self._weighted_map_loss(spatial_map, tubelet_weights)
            if temporal_tokens > 1:
                pred_delta = pred_tokens[:, 1:] - pred_tokens[:, :-1]
                target_delta = target_tokens[:, 1:] - target_tokens[:, :-1]
                pred_magnitude = pred_delta.float().norm(dim=-1)
                target_magnitude = target_delta.float().norm(dim=-1)
                direction_map = 1.0 - (
                    F.normalize(pred_delta.float(), dim=-1, eps=1e-6)
                    * F.normalize(target_delta.float(), dim=-1, eps=1e-6)
                ).sum(dim=-1)
                magnitude_map = (
                    torch.log(pred_magnitude + 1e-4)
                    - torch.log(target_magnitude + 1e-4)
                ).abs().clamp_max(5.0)
                temporal_weights = torch.minimum(
                    tubelet_weights[:, 1:],
                    tubelet_weights[:, :-1],
                )[:, :, None, None]
                # Static target locations do not define a reliable motion
                # direction. Preserve magnitude supervision but gate direction.
                motion_mask = (target_magnitude > 1e-3).to(temporal_weights)
                direction_weights = temporal_weights * motion_mask
                vjepa_temporal_direction = self._weighted_map_loss(
                    direction_map,
                    direction_weights,
                )
                vjepa_temporal_magnitude = self._weighted_map_loss(
                    magnitude_map,
                    temporal_weights,
                )
                vjepa_temporal = (
                    vjepa_temporal_direction
                    + 0.1 * vjepa_temporal_magnitude
                )
            pred_global = F.normalize(pred_tokens.mean(dim=(1, 2, 3)), dim=-1, eps=1e-6)
            target_global = F.normalize(target_tokens.mean(dim=(1, 2, 3)), dim=-1, eps=1e-6)
            global_map = 1.0 - (pred_global * target_global).sum(dim=-1)
            vjepa_global = _weighted_batch_mean(global_map, tubelet_weights.mean(dim=1))

        loss = (
            self.generation_dino_patch_loss_weight * dino_loss
            + self.generation_vjepa_spatial_loss_weight * vjepa_spatial
            + self.generation_vjepa_temporal_loss_weight * vjepa_temporal
            + self.generation_vjepa_global_loss_weight * vjepa_global
            + self.generation_color_moment_loss_weight * color_moment
            + self.generation_rgb_temporal_loss_weight * rgb_temporal
        )
        unweighted = (
            dino_loss
            + vjepa_spatial
            + vjepa_temporal
            + vjepa_global
            + color_moment
            + rgb_temporal
        )
        self.last_generation_metrics = {
            "alignment_generation_dino_patch_loss_component": dino_loss.detach(),
            "alignment_generation_vjepa_spatial_loss_component": vjepa_spatial.detach(),
            "alignment_generation_vjepa_temporal_loss_component": vjepa_temporal.detach(),
            "alignment_generation_vjepa_temporal_direction_component": vjepa_temporal_direction.detach(),
            "alignment_generation_vjepa_temporal_magnitude_component": vjepa_temporal_magnitude.detach(),
            "alignment_generation_vjepa_global_loss_component": vjepa_global.detach(),
            "alignment_generation_color_moment_loss_component": color_moment.detach(),
            "alignment_generation_rgb_temporal_loss_component": rgb_temporal.detach(),
            "alignment_generation_rgb_temporal_direction_component": rgb_temporal_direction.detach(),
            "alignment_generation_rgb_temporal_magnitude_component": rgb_temporal_magnitude.detach(),
            "alignment_generation_dino_patch_loss_weighted": (self.generation_dino_patch_loss_weight * dino_loss).detach(),
            "alignment_generation_vjepa_spatial_loss_weighted": (self.generation_vjepa_spatial_loss_weight * vjepa_spatial).detach(),
            "alignment_generation_vjepa_temporal_loss_weighted": (self.generation_vjepa_temporal_loss_weight * vjepa_temporal).detach(),
            "alignment_generation_vjepa_global_loss_weighted": (self.generation_vjepa_global_loss_weight * vjepa_global).detach(),
            "alignment_generation_color_moment_loss_weighted": (self.generation_color_moment_loss_weight * color_moment).detach(),
            "alignment_generation_rgb_temporal_loss_weighted": (self.generation_rgb_temporal_loss_weight * rgb_temporal).detach(),
            "alignment_generation_signal_mean": frame_weights.mean().detach(),
            "alignment_generation_signal_valid_fraction": (frame_weights > 0).float().mean().detach(),
        }
        return loss, unweighted.detach()
