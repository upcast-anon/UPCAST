"""Spatial factorized transition components for UPCAST.

The transition encoder and conditional visual decoder follow the query-based
cross-reconstruction design of UniT, adapted to retain an 8x8 spatial layout.
The rotation estimator used by :class:`SpatialResidualVQ` follows the rotation
trick implementation from vector-quantize-pytorch.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def resize_map(x: Tensor, size: tuple[int, int]) -> Tensor:
    if x.shape[-2:] == size:
        return x
    mode = "area" if size[0] <= x.shape[-2] and size[1] <= x.shape[-1] else "bilinear"
    if mode == "area":
        return F.interpolate(x, size=size, mode=mode)
    return F.interpolate(x, size=size, mode=mode, align_corners=False)


def gather_frames(x: Tensor, indices: Tensor) -> Tensor:
    if x.ndim != 5:
        raise ValueError(f"Expected [B,T,C,H,W], got {tuple(x.shape)}")
    if indices.shape != (x.shape[0],):
        raise ValueError(f"Expected frame indices [{x.shape[0]}], got {tuple(indices.shape)}")
    gather_index = indices[:, None, None, None, None].expand(-1, 1, *x.shape[2:])
    return x.gather(1, gather_index).squeeze(1)


def sinusoidal_embedding(values: Tensor, dim: int) -> Tensor:
    if dim <= 0:
        raise ValueError("Embedding dimension must be positive")
    half = dim // 2
    if half == 0:
        return values[:, None]
    scale = -math.log(10000.0) / max(half - 1, 1)
    frequencies = torch.exp(torch.arange(half, device=values.device, dtype=values.dtype) * scale)
    embedding = torch.cat(
        [torch.sin(values[:, None] * frequencies[None]), torch.cos(values[:, None] * frequencies[None])],
        dim=-1,
    )
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


def transformer_stack(dim: int, depth: int, heads: int, ff_dim: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=heads,
        dim_feedforward=ff_dim,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=depth, norm=nn.LayerNorm(dim))


class VideoPairProjector(nn.Module):
    def __init__(self, latents_info: list[list[int]], dim: int, grid: tuple[int, int]):
        super().__init__()
        self.grid = grid
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm2d(int(channels)),
                    nn.Conv2d(int(channels), dim, kernel_size=1),
                    nn.SiLU(),
                    nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                )
                for channels, _, _ in latents_info
            ]
        )
        self.layer_embeddings = nn.Parameter(torch.zeros(len(self.projections), dim, 1, 1))
        self.fusion = nn.Sequential(
            LayerNorm2d(dim * len(self.projections)),
            nn.Conv2d(dim * len(self.projections), dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )
        nn.init.normal_(self.layer_embeddings, std=0.02)

    def _project_frame(self, latents: list[Tensor], indices: Tensor) -> Tensor:
        maps = []
        for layer_index, (latent, projection) in enumerate(zip(latents, self.projections)):
            frame = gather_frames(latent, indices)
            projected = resize_map(projection(frame), self.grid)
            maps.append(projected + self.layer_embeddings[layer_index].to(projected))
        return self.fusion(torch.cat(maps, dim=1))

    def forward(self, latents: list[Tensor], current_indices: Tensor, future_indices: Tensor) -> tuple[Tensor, Tensor]:
        return self._project_frame(latents, current_indices), self._project_frame(latents, future_indices)


class GeometryPairProjector(nn.Module):
    def __init__(self, feature_channels: int, dim: int, grid: tuple[int, int]):
        super().__init__()
        self.grid = grid
        feature_dim = dim * 2 // 3
        point_dim = dim - feature_dim
        self.feature_projection = nn.Sequential(
            LayerNorm2d(feature_channels),
            nn.Conv2d(feature_channels, feature_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
        )
        self.point_projection = nn.Sequential(
            LayerNorm2d(3),
            nn.Conv2d(3, point_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(point_dim, point_dim, kernel_size=3, padding=1),
        )
        self.fusion = nn.Sequential(
            LayerNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )

    def _project_frame(self, features: Tensor, points: Tensor, indices: Tensor) -> Tensor:
        feature = resize_map(self.feature_projection(gather_frames(features, indices)), self.grid)
        point = resize_map(self.point_projection(gather_frames(points, indices)), self.grid)
        return self.fusion(torch.cat([feature, point], dim=1))

    def forward(
        self,
        features: Tensor,
        points: Tensor,
        current_indices: Tensor,
        future_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return (
            self._project_frame(features, points, current_indices),
            self._project_frame(features, points, future_indices),
        )


class DenseFeaturePairProjector(nn.Module):
    """Project a dense frozen-teacher feature pair onto the UPCAST spatial grid."""

    def __init__(self, feature_channels: int, dim: int, grid: tuple[int, int]):
        super().__init__()
        self.grid = grid
        self.projection = nn.Sequential(
            LayerNorm2d(feature_channels),
            nn.Conv2d(feature_channels, dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )

    def _project_frame(self, features: Tensor, indices: Tensor) -> Tensor:
        return resize_map(self.projection(gather_frames(features, indices)), self.grid)

    def forward(
        self,
        features: Tensor,
        current_indices: Tensor,
        future_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return (
            self._project_frame(features, current_indices),
            self._project_frame(features, future_indices),
        )


class ResidualTokenAdapter(nn.Module):
    """Map one modality into the shared code space without compressing it."""

    def __init__(self, dim: int, expansion: int = 2):
        super().__init__()
        hidden = dim * int(expansion)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.output_norm(tokens + self.mlp(self.norm(tokens)))


class SpatialTransitionEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        grid: tuple[int, int],
        depth: int = 4,
        heads: int = 12,
        ff_dim: int = 3072,
        use_noise_embedding: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.grid = grid
        self.num_tokens = grid[0] * grid[1]
        self.use_noise_embedding = use_noise_embedding
        self.query_tokens = nn.Parameter(torch.zeros(1, self.num_tokens, dim))
        self.separator = nn.Parameter(torch.zeros(1, 1, dim))
        self.query_position = nn.Parameter(torch.zeros(1, self.num_tokens, dim))
        self.current_position = nn.Parameter(torch.zeros(1, self.num_tokens, dim))
        self.future_position = nn.Parameter(torch.zeros(1, self.num_tokens, dim))
        self.type_embeddings = nn.Parameter(torch.zeros(3, dim))
        self.delta_projection = nn.Sequential(nn.Linear(128, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.noise_projection = (
            nn.Sequential(nn.Linear(128, dim), nn.SiLU(), nn.Linear(dim, dim))
            if use_noise_embedding
            else None
        )
        self.local_transition = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        self.transformer = transformer_stack(dim, depth, heads, ff_dim)
        self.output_norm = nn.LayerNorm(dim)
        for parameter in (
            self.query_tokens,
            self.separator,
            self.query_position,
            self.current_position,
            self.future_position,
            self.type_embeddings,
        ):
            nn.init.normal_(parameter, std=0.02)

    def _map_tokens(self, feature_map: Tensor) -> Tensor:
        feature_map = resize_map(feature_map, self.grid)
        return rearrange(feature_map, "b c h w -> b (h w) c")

    def forward(
        self,
        current_map: Tensor,
        future_map: Tensor,
        delta: Tensor,
        current_logsnr: Tensor | None = None,
        future_logsnr: Tensor | None = None,
    ) -> Tensor:
        batch_size = current_map.shape[0]
        current_content = self._map_tokens(current_map)
        future_content = self._map_tokens(future_map)
        local_transition = self.local_transition(
            torch.cat(
                (
                    current_content,
                    future_content,
                    future_content - current_content,
                ),
                dim=-1,
            )
        )
        current = current_content + self.current_position + self.type_embeddings[1][None, None]
        future = future_content + self.future_position + self.type_embeddings[2][None, None]
        query = self.query_tokens.expand(batch_size, -1, -1) + self.query_position + self.type_embeddings[0][None, None]

        delta_embedding = self.delta_projection(sinusoidal_embedding(delta.float(), 128)).to(query.dtype)
        query = query + delta_embedding[:, None]
        if self.noise_projection is not None:
            if current_logsnr is None or future_logsnr is None:
                raise ValueError("Video transition encoding requires current and future logSNR")
            current_noise = self.noise_projection(sinusoidal_embedding(current_logsnr.float(), 128)).to(current.dtype)
            future_noise = self.noise_projection(sinusoidal_embedding(future_logsnr.float(), 128)).to(future.dtype)
            current = current + current_noise[:, None]
            future = future + future_noise[:, None]

        separator = self.separator.expand(batch_size, -1, -1)
        encoded = self.transformer(torch.cat([query, current, separator, future], dim=1))
        # The fixed residual route prevents learned queries from solving the
        # objective while ignoring the actual local frame transition.
        return self.output_norm(encoded[:, : self.num_tokens] + local_transition)


class SpatialFusionEncoder(nn.Module):
    def __init__(self, dim: int, grid: tuple[int, int], depth: int = 4, heads: int = 12, ff_dim: int = 3072):
        super().__init__()
        self.num_tokens = grid[0] * grid[1]
        self.query_tokens = nn.Parameter(torch.zeros(1, self.num_tokens, dim))
        self.separator = nn.Parameter(torch.zeros(1, 1, dim))
        self.type_embeddings = nn.Parameter(torch.zeros(3, dim))
        self.local_fusion = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        self.transformer = transformer_stack(dim, depth, heads, ff_dim)
        self.output_norm = nn.LayerNorm(dim)
        nn.init.normal_(self.query_tokens, std=0.02)
        nn.init.normal_(self.separator, std=0.02)
        nn.init.normal_(self.type_embeddings, std=0.02)

    def forward(self, video_tokens: Tensor, geometry_tokens: Tensor) -> Tensor:
        batch_size = video_tokens.shape[0]
        query = self.query_tokens.expand(batch_size, -1, -1) + self.type_embeddings[0][None, None]
        video = video_tokens + self.type_embeddings[1][None, None]
        geometry = geometry_tokens + self.type_embeddings[2][None, None]
        separator = self.separator.expand(batch_size, -1, -1)
        encoded = self.transformer(
            torch.cat([query, video, separator, geometry], dim=1)
        )[:, : self.num_tokens]
        local = self.local_fusion(
            torch.cat(
                (video_tokens, geometry_tokens, video_tokens - geometry_tokens),
                dim=-1,
            )
        )
        # Geometry is an explicit residual teacher. Fusion can enrich it with
        # video evidence but cannot silently discard the physical structure.
        return self.output_norm(encoded + geometry_tokens + local)


class WorldTokenFusion(nn.Module):
    """Build an invertible direct-sum world token from physical and appearance parts.

    The additive coupling keeps both ``dim``-wide inputs in a ``2 * dim`` output.
    This matters here: compressing the concatenation before cross-reconstruction
    would recreate the information bottleneck that the factorized latent is
    intended to avoid.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.physical_norm = nn.LayerNorm(dim)
        self.appearance_norm = nn.LayerNorm(dim)
        self.appearance_to_physical = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        self.physical_to_appearance = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, physical: Tensor, appearance: Tensor) -> Tensor:
        if physical.shape != appearance.shape:
            raise ValueError(
                "Physical and appearance tokens must have the same shape, got "
                f"{tuple(physical.shape)} and {tuple(appearance.shape)}"
            )
        mixed_physical = physical + self.appearance_to_physical(
            self.appearance_norm(appearance)
        )
        mixed_appearance = appearance + self.physical_to_appearance(
            self.physical_norm(mixed_physical)
        )
        return torch.cat((mixed_physical, mixed_appearance), dim=-1)

    def inverse(self, world: Tensor) -> tuple[Tensor, Tensor]:
        if world.shape[-1] % 2 != 0:
            raise ValueError(
                f"World token width must be even, got {world.shape[-1]}"
            )
        mixed_physical, mixed_appearance = world.chunk(2, dim=-1)
        appearance = mixed_appearance - self.physical_to_appearance(
            self.physical_norm(mixed_physical)
        )
        physical = mixed_physical - self.appearance_to_physical(
            self.appearance_norm(appearance)
        )
        return physical, appearance


def _l2norm(x: Tensor) -> Tensor:
    return F.normalize(x, p=2, dim=-1, eps=1e-6)


def _rotate_to(source: Tensor, target: Tensor) -> Tensor:
    """Preserve target values while transporting gradients along source-to-target rotation."""
    source_norm = source.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    source_direction = _l2norm(source)
    target_direction = _l2norm(target)
    reflection = _l2norm(source_direction + target_direction).detach()
    rotated = source_norm * (
        2.0 * (source_direction * reflection).sum(dim=-1, keepdim=True) * reflection - source_direction
    )
    scale = (target.norm(dim=-1, keepdim=True) / source_norm).detach()
    rotated = rotated * scale
    return rotated + (target - rotated).detach()


class SpatialResidualVQ(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        code_dim: int = 64,
        num_quantizers: int = 2,
        codebook_size: int = 256,
        commitment_beta: float = 0.25,
        ema_decay: float = 0.8,
        dead_code_threshold: float = 2.0,
        dead_code_restart_interval: int = 100,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.commitment_beta = commitment_beta
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold
        self.dead_code_restart_interval = max(1, int(dead_code_restart_interval))
        self.project_down = nn.Linear(input_dim, code_dim)
        self.project_up = nn.Linear(code_dim, input_dim)
        codebooks = torch.randn(num_quantizers, codebook_size, code_dim)
        codebooks = F.normalize(codebooks, dim=-1)
        self.register_buffer("codebooks", codebooks)
        self.register_buffer("ema_counts", torch.zeros(num_quantizers, codebook_size))
        self.register_buffer("ema_sums", codebooks.clone())
        self.register_buffer(
            "ema_update_count",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "last_restarted_codes",
            torch.zeros(num_quantizers, dtype=torch.long),
            persistent=False,
        )
        # EMA statistics are accumulated while gradients are accumulated.  They
        # are intentionally non-persistent: checkpoints are written on optimizer
        # boundaries, where this window must be empty.
        # Pending values are deliberately plain attributes, not registered
        # buffers. DDP may broadcast module buffers before every forward, which
        # would overwrite each rank's distinct accumulation statistics.
        self._pending_ema_counts: Tensor | None = None
        self._pending_ema_sums: Tensor | None = None
        self._pending_microbatches = 0
        self._pending_token_count = 0
        self.register_buffer(
            "last_committed_microbatches",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "last_committed_global_tokens",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self._pending_restart_samples: list[list[Tensor]] = [
            [] for _ in range(num_quantizers)
        ]

    @property
    def pending_microbatches(self) -> Tensor:
        return torch.tensor(
            self._pending_microbatches,
            device=self.codebooks.device,
            dtype=torch.long,
        )

    @property
    def pending_token_count(self) -> Tensor:
        return torch.tensor(
            self._pending_token_count,
            device=self.codebooks.device,
            dtype=torch.long,
        )

    def _ensure_pending_statistics(self) -> tuple[Tensor, Tensor]:
        if self._pending_ema_counts is None:
            self._pending_ema_counts = torch.zeros_like(self.ema_counts)
        if self._pending_ema_sums is None:
            self._pending_ema_sums = torch.zeros_like(self.ema_sums)
        return self._pending_ema_counts, self._pending_ema_sums

    def _nearest(self, residual: Tensor, codebook: Tensor) -> Tensor:
        residual_sq = residual.square().sum(dim=-1, keepdim=True)
        codebook_sq = codebook.square().sum(dim=-1)[None]
        distance = residual_sq + codebook_sq - 2.0 * residual @ codebook.t()
        return distance.argmin(dim=-1)

    def assignment_loss(
        self,
        tokens: Tensor,
        target_indices: Tensor,
        temperature: float = 0.1,
        level_weights: tuple[float, ...] | list[float] | None = None,
    ) -> Tensor:
        """Map a student branch to code indices emitted by a fixed teacher branch."""
        if target_indices.shape != (*tokens.shape[:-1], self.num_quantizers):
            raise ValueError(
                f"Target indices {tuple(target_indices.shape)} do not match "
                f"tokens {tuple(tokens.shape)} and {self.num_quantizers} quantizers"
            )
        projected = self.project_down(tokens).float()
        residual = projected.reshape(-1, self.code_dim)
        losses = []
        if level_weights is None:
            weights = [1.0] * self.num_quantizers
        else:
            weights = [float(value) for value in level_weights]
            if len(weights) != self.num_quantizers or any(
                value < 0 for value in weights
            ):
                raise ValueError(
                    "level_weights must contain one non-negative value per "
                    f"quantizer, got {weights}"
                )
            if sum(weights) <= 0:
                raise ValueError("At least one assignment level weight must be positive")
        temperature = max(float(temperature), 1e-4)
        for level in range(self.num_quantizers):
            # The snapshot remains fixed until the whole optimizer accumulation
            # window has completed its backwards.
            codebook = self.codebooks[level].float().detach().clone()
            residual_sq = residual.square().sum(dim=-1, keepdim=True)
            codebook_sq = codebook.square().sum(dim=-1)[None]
            distance = residual_sq + codebook_sq - 2.0 * residual @ codebook.t()
            # Mean squared distance keeps the temperature independent of VQ
            # width. The previous summed distance made a 64-D code assignment
            # loss dominate every reconstruction objective.
            logits = -distance / (temperature * self.code_dim)
            targets = target_indices[..., level].reshape(-1).long()
            losses.append(F.cross_entropy(logits, targets))
            teacher_code = F.embedding(targets, codebook)
            residual = residual - teacher_code.detach()
        weight_tensor = projected.new_tensor(weights)
        return (torch.stack(losses) * weight_tensor).sum() / weight_tensor.sum()

    def soft_assignment_distillation(
        self,
        student_tokens: Tensor,
        teacher_tokens: Tensor,
        temperature: float = 0.1,
        level_weights: tuple[float, ...] | list[float] | None = None,
    ) -> Tensor:
        """Distill the teacher's full residual-code distribution per token."""
        if student_tokens.shape != teacher_tokens.shape:
            raise ValueError(
                "Student and teacher tokens must have the same shape, got "
                f"{tuple(student_tokens.shape)} and {tuple(teacher_tokens.shape)}"
            )
        if level_weights is None:
            weights = [1.0] * self.num_quantizers
        else:
            weights = [float(value) for value in level_weights]
            if len(weights) != self.num_quantizers or any(
                value < 0 for value in weights
            ):
                raise ValueError(
                    "level_weights must contain one non-negative value per "
                    f"quantizer, got {weights}"
                )
            if sum(weights) <= 0:
                raise ValueError("At least one soft-assignment weight must be positive")

        student = self.project_down(student_tokens).float().reshape(-1, self.code_dim)
        teacher = (
            self.project_down(teacher_tokens).float().detach().reshape(-1, self.code_dim)
        )
        temperature = max(float(temperature), 1e-4)
        losses = []
        codebook_snapshot = self.codebooks.detach().float().clone()
        for level in range(self.num_quantizers):
            codebook = codebook_snapshot[level]
            student_distance = (
                student.square().sum(dim=-1, keepdim=True)
                + codebook.square().sum(dim=-1)[None]
                - 2.0 * student @ codebook.t()
            )
            teacher_distance = (
                teacher.square().sum(dim=-1, keepdim=True)
                + codebook.square().sum(dim=-1)[None]
                - 2.0 * teacher @ codebook.t()
            )
            scale = temperature * self.code_dim
            teacher_probability = (-teacher_distance / scale).softmax(dim=-1)
            student_log_probability = (-student_distance / scale).log_softmax(
                dim=-1
            )
            losses.append(
                F.kl_div(
                    student_log_probability,
                    teacher_probability,
                    reduction="batchmean",
                )
            )
            teacher_indices = teacher_distance.argmin(dim=-1)
            teacher_code = F.embedding(teacher_indices, codebook)
            student = student - teacher_code
            teacher = teacher - teacher_code

        weight_tensor = student_tokens.new_tensor(weights, dtype=torch.float32)
        return (torch.stack(losses) * weight_tensor).sum() / weight_tensor.sum()

    def projected_alignment_loss(
        self,
        student_tokens: Tensor,
        teacher_tokens: Tensor,
    ) -> Tensor:
        """Align physical content before discretization while preserving scale."""
        if student_tokens.shape != teacher_tokens.shape:
            raise ValueError(
                "Student and teacher tokens must have the same shape, got "
                f"{tuple(student_tokens.shape)} and {tuple(teacher_tokens.shape)}"
            )
        student = self.project_down(student_tokens).float()
        teacher = self.project_down(teacher_tokens).float().detach()
        direction = 1.0 - (
            F.normalize(student, dim=-1, eps=1e-6)
            * F.normalize(teacher, dim=-1, eps=1e-6)
        ).sum(dim=-1)
        magnitude = F.smooth_l1_loss(
            student.norm(dim=-1),
            teacher.norm(dim=-1),
            reduction="none",
            beta=0.1,
        )
        return (direction + 0.1 * magnitude).mean()

    def projected_spread_loss(
        self,
        tokens: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Keep a modality's VQ projection at the scale of unit-norm codes."""
        if tokens.ndim < 2 or tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected [...,{self.input_dim}] tokens, got {tuple(tokens.shape)}"
            )
        if tokens.numel() == 0:
            zero = self.project_down.weight.sum() * 0.0
            return zero, {
                "projected_std": zero.detach(),
                "projected_spread_loss": zero.detach(),
            }
        projected = self.project_down(tokens).float().reshape(-1, self.code_dim)
        projected_std_per_dim = torch.sqrt(
            projected.var(dim=0, unbiased=False) + 1e-4
        )
        projected_std = projected_std_per_dim.mean()
        target_std = 1.0 / math.sqrt(self.code_dim)
        projected_spread_loss = (
            F.relu(target_std - projected_std_per_dim).mean() / target_std
        )
        return projected_spread_loss, {
            "projected_std": projected_std.detach().to(tokens),
            "projected_spread_loss": projected_spread_loss.detach().to(tokens),
        }

    def codebook_usage_loss(
        self,
        tokens: Tensor,
        temperature: float = 0.01,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Encourage the geometry teacher to use the shared code space.

        Hard nearest-neighbour usage is not differentiable. This objective uses
        the marginal soft assignment entropy at each residual level, while the
        actual forward path remains hard RVQ. Under DDP every rank observes the
        same global marginal and contributes gradients for its local tokens.
        """
        projected = self.project_down(tokens).float()
        residual = projected.reshape(-1, self.code_dim)
        temperature = max(float(temperature), 1e-4)
        normalizer = math.log(max(self.codebook_size, 2))
        losses = []
        soft_perplexities = []
        codebook_snapshot = self.codebooks.detach().float().clone()
        for level in range(self.num_quantizers):
            codebook = codebook_snapshot[level]
            residual_sq = residual.square().sum(dim=-1, keepdim=True)
            codebook_sq = codebook.square().sum(dim=-1)[None]
            distance = residual_sq + codebook_sq - 2.0 * residual @ codebook.t()
            logits = -distance / (temperature * self.code_dim)
            probabilities = logits.softmax(dim=-1)
            local_marginal = probabilities.mean(dim=0)
            marginal = local_marginal
            if dist.is_available() and dist.is_initialized():
                global_marginal = local_marginal.detach().clone()
                dist.all_reduce(global_marginal)
                global_marginal.div_(dist.get_world_size())
                # DDP averages parameter gradients across ranks, supplying the
                # missing 1/world_size factor for this local differentiable term.
                marginal = global_marginal + (
                    local_marginal - local_marginal.detach()
                )
            entropy = -(
                marginal * marginal.clamp_min(1e-12).log()
            ).sum()
            losses.append((normalizer - entropy) / normalizer)
            soft_perplexities.append(entropy.detach().exp().to(tokens))

            indices = distance.detach().argmin(dim=-1)
            residual = residual - F.embedding(indices, codebook).detach()

        projected_spread_loss, spread_metrics = self.projected_spread_loss(tokens)
        return torch.stack(losses).mean() + projected_spread_loss, {
            "soft_perplexity": torch.stack(soft_perplexities),
            **spread_metrics,
        }

    @torch.no_grad()
    def _gather_restart_samples(self, samples: Tensor) -> Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return samples
        world_size = dist.get_world_size()
        local_count = torch.tensor(
            [samples.shape[0]],
            device=samples.device,
            dtype=torch.long,
        )
        gathered_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
        dist.all_gather(gathered_counts, local_count)
        counts = [int(value.item()) for value in gathered_counts]
        max_count = max(counts)
        if max_count == 0:
            return samples[:0]
        padded = F.pad(samples, (0, 0, 0, max_count - samples.shape[0]))
        gathered = [torch.zeros_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded)
        return torch.cat(
            [value[:count] for value, count in zip(gathered, counts)],
            dim=0,
        )

    @torch.no_grad()
    def _restart_dead_codes(self, level: int, samples: Tensor) -> int:
        dead = self.ema_counts[level] < self.dead_code_threshold
        num_dead = int(dead.sum().item())
        if num_dead == 0:
            return 0

        # Every rank must enter the gather even if its local shard is empty.
        # Otherwise an uneven final DDP batch can deadlock during a restart.
        candidates = self._gather_restart_samples(samples)
        if candidates.shape[0] == 0:
            return 0
        replacements = torch.empty(
            num_dead,
            self.code_dim,
            device=samples.device,
            dtype=samples.dtype,
        )
        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            # Prefer unique global transition tokens. Repetition is only used
            # when the number of dead codes exceeds all tokens in the batch.
            order = torch.randperm(candidates.shape[0], device=samples.device)
            selected = order.repeat(
                math.ceil(num_dead / max(candidates.shape[0], 1))
            )[:num_dead]
            replacements.copy_(candidates[selected])
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(replacements, src=0)
        self.codebooks[level, dead] = replacements
        self.ema_sums[level, dead] = replacements * self.dead_code_threshold
        self.ema_counts[level, dead] = self.dead_code_threshold
        return num_dead

    @torch.no_grad()
    def _ema_update_from_statistics(
        self,
        level: int,
        counts: Tensor,
        sums: Tensor,
        restart_samples: Tensor,
        restart_dead_codes: bool,
    ) -> int:
        self.ema_counts[level].mul_(self.ema_decay).add_(counts, alpha=1.0 - self.ema_decay)
        self.ema_sums[level].mul_(self.ema_decay).add_(sums, alpha=1.0 - self.ema_decay)
        denominator = self.ema_counts[level].clamp_min(1e-5)[:, None]
        self.codebooks[level].copy_(self.ema_sums[level] / denominator)

        if restart_dead_codes:
            return self._restart_dead_codes(level, restart_samples)
        return 0

    @torch.no_grad()
    def queue_codebook_update(self, tokens: Tensor, indices: Tensor) -> None:
        """Accumulate one micro-batch without mutating the shared codebook.

        All residuals in an optimizer accumulation window are built from the
        same immutable codebook.  This is essential when assignment gradients
        from several micro-batches are combined into one optimizer update.
        """
        expected = (*tokens.shape[:-1], self.num_quantizers)
        if indices.shape != expected:
            raise ValueError(
                f"Indices {tuple(indices.shape)} do not match tokens "
                f"{tuple(tokens.shape)} and {self.num_quantizers} quantizers"
            )
        projected = self.project_down(tokens).float().reshape(-1, self.code_dim)
        residual = projected
        codebook_snapshot = self.codebooks.detach().float().clone()
        pending_counts, pending_sums = self._ensure_pending_statistics()
        for level in range(self.num_quantizers):
            level_indices = indices[..., level].reshape(-1).long()
            counts = torch.bincount(
                level_indices,
                minlength=self.codebook_size,
            ).to(pending_counts)
            sums = torch.zeros_like(pending_sums[level])
            sums.index_add_(0, level_indices, residual.to(sums))
            pending_counts[level].add_(counts)
            pending_sums[level].add_(sums)
            self._pending_restart_samples[level].append(residual.detach().clone())
            teacher_code = F.embedding(
                level_indices,
                codebook_snapshot[level],
            )
            residual = residual - teacher_code
        self._pending_microbatches += 1
        self._pending_token_count += projected.shape[0]

    @torch.no_grad()
    def discard_pending_codebook_update(self) -> None:
        self._pending_ema_counts = None
        self._pending_ema_sums = None
        self._pending_microbatches = 0
        self._pending_token_count = 0
        self._pending_restart_samples = [
            [] for _ in range(self.num_quantizers)
        ]

    @torch.no_grad()
    def commit_codebook_update(
        self,
        expected_microbatches: int | None = None,
    ) -> Tensor:
        """Commit one optimizer window with one synchronized EMA update."""
        local_microbatches = self._pending_microbatches
        if dist.is_available() and dist.is_initialized():
            window_bounds = torch.tensor(
                [local_microbatches, local_microbatches],
                device=self.codebooks.device,
                dtype=torch.long,
            )
            dist.all_reduce(window_bounds[:1], op=dist.ReduceOp.MIN)
            dist.all_reduce(window_bounds[1:], op=dist.ReduceOp.MAX)
            if int(window_bounds[0].item()) != int(window_bounds[1].item()):
                raise RuntimeError(
                    "DDP ranks queued different RVQ accumulation windows: "
                    f"min={int(window_bounds[0].item())}, "
                    f"max={int(window_bounds[1].item())}."
                )
        if local_microbatches == 0:
            return torch.zeros_like(self.last_restarted_codes)
        if expected_microbatches is not None and local_microbatches != int(
            expected_microbatches
        ):
            raise RuntimeError(
                "RVQ accumulation window is incomplete: "
                f"queued={local_microbatches}, expected={expected_microbatches}."
            )

        if self._pending_ema_counts is None or self._pending_ema_sums is None:
            raise RuntimeError("RVQ pending window has no accumulated EMA statistics.")
        counts = self._pending_ema_counts.detach().clone()
        sums = self._pending_ema_sums.detach().clone()
        global_tokens = torch.tensor(
            self._pending_token_count,
            device=self.codebooks.device,
            dtype=torch.long,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts)
            dist.all_reduce(sums)
            dist.all_reduce(global_tokens)

        self.ema_update_count.add_(1)
        update_number = int(self.ema_update_count.item())
        restart = (
            update_number == 1
            or update_number % self.dead_code_restart_interval == 0
        )
        restarted = []
        for level in range(self.num_quantizers):
            samples = (
                torch.cat(self._pending_restart_samples[level], dim=0)
                if self._pending_restart_samples[level]
                else sums.new_zeros((0, self.code_dim))
            )
            restarted.append(
                self._ema_update_from_statistics(
                    level,
                    counts[level],
                    sums[level],
                    samples,
                    restart_dead_codes=restart,
                )
            )
        restarted_tensor = torch.tensor(
            restarted,
            device=self.last_restarted_codes.device,
            dtype=self.last_restarted_codes.dtype,
        )
        self.last_restarted_codes.copy_(restarted_tensor)
        self.last_committed_microbatches.fill_(local_microbatches)
        self.last_committed_global_tokens.copy_(global_tokens)
        self.discard_pending_codebook_update()
        return restarted_tensor

    @torch.no_grad()
    def update_codebook(self, tokens: Tensor, indices: Tensor) -> Tensor:
        """Immediately update one standalone batch.

        Training code must use :meth:`queue_codebook_update` followed by one
        :meth:`commit_codebook_update` at the optimizer boundary.
        """
        if int(self.pending_microbatches.item()) != 0:
            raise RuntimeError(
                "Immediate RVQ update cannot be mixed with a queued optimizer window."
            )
        self.queue_codebook_update(tokens, indices)
        return self.commit_codebook_update(expected_microbatches=1)

    def forward(self, tokens: Tensor, freeze_codebook: bool = False) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        projected = self.project_down(tokens)
        # Distances, EMA accumulation, and dead-code replacement stay in FP32.
        # This avoids half-precision nearest-neighbour ties and dtype-dependent
        # index_add failures under mixed precision training.
        flat = projected.float().reshape(-1, self.code_dim)
        residual = flat
        quantized_sum = torch.zeros_like(flat)
        losses = []
        sample_losses = []
        token_losses = []
        active_ratios = []
        perplexities = []
        all_indices = []

        # Every level in this forward uses one immutable codebook snapshot.
        # This also makes it safe to defer EMA updates until all modalities in
        # a shared-codebook micro-batch have been quantized.
        codebook_snapshot = self.codebooks.detach().clone()
        for level in range(self.num_quantizers):
            level_input = residual
            indices = self._nearest(level_input.detach(), codebook_snapshot[level])
            quantized = F.embedding(indices, codebook_snapshot[level])
            quantized_st = _rotate_to(level_input, quantized)
            quantized_sum = quantized_sum + quantized_st
            level_token_loss = self.commitment_beta * (
                level_input - quantized.detach()
            ).square().mean(dim=-1).reshape(*tokens.shape[:-1])
            level_sample_loss = level_token_loss.reshape(
                tokens.shape[0], -1
            ).mean(dim=1)
            token_losses.append(level_token_loss)
            sample_losses.append(level_sample_loss)
            losses.append(level_sample_loss.mean())
            residual = residual - quantized.detach()

            with torch.no_grad():
                counts = torch.bincount(indices, minlength=self.codebook_size).float()
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(counts)
                probabilities = counts / counts.sum().clamp_min(1.0)
                active_ratios.append((counts > 0).float().mean().to(tokens))
                perplexities.append(torch.exp(-(probabilities * (probabilities + 1e-12).log()).sum()).to(tokens))
                all_indices.append(indices.reshape(*tokens.shape[:-1]))

        stacked_indices = torch.stack(all_indices, dim=-1)
        if self.training and not freeze_codebook:
            self.update_codebook(tokens.detach(), stacked_indices.detach())

        quantized_sum = quantized_sum.reshape(*tokens.shape[:-1], self.code_dim).to(projected.dtype)
        quantized_tokens = self.project_up(quantized_sum)
        return quantized_tokens, torch.stack(losses).mean(), {
            "active_code_ratio": torch.stack(active_ratios),
            "perplexity": torch.stack(perplexities),
            "indices": stacked_indices,
            "commitment_per_sample": torch.stack(sample_losses, dim=0).mean(dim=0),
            "commitment_per_token": torch.stack(token_losses, dim=0).mean(dim=0),
            # Keep the compact shared physical state available to generative
            # objectives. This tensor preserves the rotation-trick gradient;
            # callers decide where the oracle branch must be detached.
            "native_quantized": quantized_sum,
        }


class VisualAnchorEncoder(nn.Module):
    def __init__(self, dino_dim: int, dim: int = 768, output_grid: tuple[int, int] = (16, 16)):
        super().__init__()
        self.output_grid = output_grid
        semantic_dim = dim // 2
        appearance_dim = dim - semantic_dim
        self.dino_projection = nn.Sequential(
            LayerNorm2d(dino_dim),
            nn.Conv2d(dino_dim, semantic_dim, kernel_size=1),
        )
        self.pixel_stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(192, appearance_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(LayerNorm2d(dim), nn.Conv2d(dim, dim, kernel_size=3, padding=1))

    def forward(self, image: Tensor, dino_features: Tensor) -> Tensor:
        semantic = resize_map(self.dino_projection(dino_features), self.output_grid)
        appearance = resize_map(self.pixel_stem(image), self.output_grid)
        anchor = self.fusion(torch.cat([semantic, appearance], dim=1))
        return rearrange(anchor, "b c h w -> b (h w) c")

    @staticmethod
    def token_dropout(
        anchor: Tensor,
        probability: float,
        enabled: bool,
    ) -> tuple[Tensor, Tensor]:
        if not enabled or probability <= 0:
            return anchor, anchor.new_zeros(())
        dropped = torch.rand(
            anchor.shape[0],
            anchor.shape[1],
            1,
            device=anchor.device,
        ) < probability
        return anchor.masked_fill(dropped, 0.0), dropped.float().mean().to(anchor)

    @staticmethod
    def condition_dropout(
        anchor: Tensor,
        token_probability: float,
        sample_probability: float,
        enabled: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Drop full anchors as well as local tokens.

        Local dropout alone leaves enough current-frame context for a decoder to
        ignore transition tokens. Full-sample dropout creates an explicit
        transition-only reconstruction path without changing inference.
        """
        zero = anchor.new_zeros(())
        if not enabled:
            return anchor, zero, zero
        sample_mask = (
            torch.rand(anchor.shape[0], 1, 1, device=anchor.device)
            < float(sample_probability)
        )
        token_mask = (
            torch.rand(anchor.shape[0], anchor.shape[1], 1, device=anchor.device)
            < float(token_probability)
        )
        dropped = sample_mask | token_mask
        return (
            anchor.masked_fill(dropped, 0.0),
            token_mask.float().mean().to(anchor),
            sample_mask.float().mean().to(anchor),
        )


class GeometryAnchorEncoder(nn.Module):
    def __init__(self, page_dim: int, dim: int = 768, output_grid: tuple[int, int] = (16, 16)):
        super().__init__()
        self.output_grid = output_grid
        feature_dim = dim * 2 // 3
        point_dim = dim - feature_dim
        self.feature_projection = nn.Sequential(LayerNorm2d(page_dim), nn.Conv2d(page_dim, feature_dim, 1))
        self.point_projection = nn.Sequential(LayerNorm2d(3), nn.Conv2d(3, point_dim, 1))
        self.fusion = nn.Sequential(LayerNorm2d(dim), nn.Conv2d(dim, dim, 3, padding=1))

    def forward(self, page_features: Tensor, points: Tensor) -> Tensor:
        features = resize_map(self.feature_projection(page_features), self.output_grid)
        points = resize_map(self.point_projection(points), self.output_grid)
        anchor = self.fusion(torch.cat([features, points], dim=1))
        return rearrange(anchor, "b c h w -> b (h w) c")

    token_dropout = staticmethod(VisualAnchorEncoder.token_dropout)
    condition_dropout = staticmethod(VisualAnchorEncoder.condition_dropout)


class ConditionalVisualDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        dino_dim: int,
        latent_channels: int,
        token_grid: tuple[int, int] = (8, 8),
        anchor_grid: tuple[int, int] = (16, 16),
        depth: int = 12,
        heads: int = 12,
        ff_dim: int = 3072,
        token_dim: int | None = None,
    ):
        super().__init__()
        self.token_grid = token_grid
        self.anchor_grid = anchor_grid
        self.model_dim = int(token_dim or dim)
        self.token_projection = nn.Identity()
        self.anchor_projection = (
            nn.Identity()
            if self.model_dim == dim
            else nn.Linear(dim, self.model_dim)
        )
        self.delta_projection = nn.Sequential(
            nn.Linear(128, self.model_dim),
            nn.SiLU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.condition_mixer = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.model_dim * 2),
            nn.SiLU(),
            nn.Linear(self.model_dim * 2, self.model_dim),
        )
        self.position = nn.Parameter(
            torch.zeros(1, anchor_grid[0] * anchor_grid[1], self.model_dim)
        )
        self.transformer = transformer_stack(
            self.model_dim,
            depth,
            heads,
            ff_dim,
        )
        self.dino_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, dino_dim),
        )
        self.latent_head = nn.Sequential(
            nn.ConvTranspose2d(
                self.model_dim,
                dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(32, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim, latent_channels, kernel_size=1),
        )
        nn.init.normal_(self.position, std=0.02)

    def forward(self, quantized_tokens: Tensor, anchor_tokens: Tensor, delta: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = quantized_tokens.shape[0]
        if quantized_tokens.shape[-1] != self.model_dim:
            raise ValueError(
                f"Visual decoder expected {self.model_dim}-wide world tokens, "
                f"got {quantized_tokens.shape[-1]}"
            )
        quantized_tokens = self.token_projection(quantized_tokens)
        q_map = rearrange(
            quantized_tokens,
            "b (h w) c -> b c h w",
            h=self.token_grid[0],
            w=self.token_grid[1],
        )
        q_map = resize_map(q_map, self.anchor_grid)
        q_tokens = rearrange(q_map, "b c h w -> b (h w) c")
        delta_embedding = self.delta_projection(sinusoidal_embedding(delta.float(), 128)).to(q_tokens.dtype)
        anchor_tokens = self.anchor_projection(anchor_tokens)
        hidden = anchor_tokens + q_tokens
        hidden = hidden + self.condition_mixer(hidden)
        hidden = self.transformer(hidden + self.position + delta_embedding[:, None])
        dino_features = self.dino_head(hidden)
        hidden_map = rearrange(
            hidden,
            "b (h w) c -> b c h w",
            h=self.anchor_grid[0],
            w=self.anchor_grid[1],
        )
        return dino_features, self.latent_head(hidden_map)


class ConditionalGeometryDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        page_dim: int,
        token_grid: tuple[int, int] = (8, 8),
        anchor_grid: tuple[int, int] = (16, 16),
        point_size: tuple[int, int] = (128, 128),
        depth: int = 8,
        heads: int = 12,
        ff_dim: int = 3072,
    ):
        super().__init__()
        self.token_grid = token_grid
        self.anchor_grid = anchor_grid
        self.point_size = point_size
        self.delta_projection = nn.Sequential(nn.Linear(128, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.condition_fusion = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.position = nn.Parameter(torch.zeros(1, anchor_grid[0] * anchor_grid[1], dim))
        self.transformer = transformer_stack(dim, depth, heads, ff_dim)
        self.page_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, page_dim))
        self.point_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(dim // 2, dim // 4, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(dim // 4, dim // 8, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim // 8, 4, kernel_size=1),
        )
        nn.init.normal_(self.position, std=0.02)

    def forward(self, quantized_tokens: Tensor, anchor_tokens: Tensor, delta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        q_map = rearrange(
            quantized_tokens,
            "b (h w) c -> b c h w",
            h=self.token_grid[0],
            w=self.token_grid[1],
        )
        q_map = resize_map(q_map, self.anchor_grid)
        q_tokens = rearrange(q_map, "b c h w -> b (h w) c")
        delta_embedding = self.delta_projection(sinusoidal_embedding(delta.float(), 128)).to(q_tokens.dtype)
        hidden = self.condition_fusion(torch.cat([anchor_tokens, q_tokens], dim=-1))
        hidden = self.transformer(hidden + self.position + delta_embedding[:, None])
        page_features = self.page_head(hidden)
        hidden_map = rearrange(
            hidden,
            "b (h w) c -> b c h w",
            h=self.anchor_grid[0],
            w=self.anchor_grid[1],
        )
        point_output = resize_map(self.point_head(hidden_map), self.point_size)
        return page_features, point_output[:, :3], F.softplus(point_output[:, 3:4]) + 1e-6
