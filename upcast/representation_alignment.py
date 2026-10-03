"""Same-backbone REPA and VideoREPA objectives for DFoT.

The teachers are loaded lazily and intentionally kept outside the registered
module tree.  They are frozen training assets, so persisting them in every
Lightning checkpoint would add hundreds of megabytes without changing the
inference model.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from typing import Any

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class TokenProjector(nn.Module):
    """The three-layer SiLU projector used by the official VideoREPA code."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        return self.net(tokens)


def repa_cosine_loss(student: Tensor, teacher: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    """Patch-wise REPA loss for matching token tensors of identical shape."""
    if student.shape != teacher.shape:
        raise ValueError(
            "REPA student and teacher tokens must match, got "
            f"{tuple(student.shape)} and {tuple(teacher.shape)}"
        )
    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.detach().float(), dim=-1)
    cosine = (student * teacher).sum(dim=-1)
    loss = 1.0 - cosine.mean()
    return loss, {
        "cosine": cosine.mean().detach(),
        "student_std": student.std().detach(),
        "teacher_std": teacher.std().detach(),
    }


def token_relation_distillation_loss(
    student: Tensor,
    teacher: Tensor,
    *,
    margin: float,
    query_frame_chunk: int = 1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Exact VideoREPA TRD loss with memory-bounded query-frame chunks.

    Args:
        student: Projected student tokens shaped ``[B,T,S,D]``.
        teacher: Frozen video-teacher tokens shaped ``[B,T,S,D]``.
        margin: The official hinge margin applied to relation errors.
        query_frame_chunk: Number of query frames materialized at once.  This
            changes peak memory only; the summed objective is unchanged.
    """
    if student.shape != teacher.shape or student.ndim != 4:
        raise ValueError(
            "TRD expects matching [B,T,S,D] tensors, got "
            f"{tuple(student.shape)} and {tuple(teacher.shape)}"
        )
    if margin < 0:
        raise ValueError(f"TRD margin must be non-negative, got {margin}")
    query_frame_chunk = max(int(query_frame_chunk), 1)
    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.detach().float(), dim=-1)
    student_all = student.flatten(1, 2)
    teacher_all = teacher.flatten(1, 2)

    hinge_sum = student.new_zeros(())
    absolute_sum = student.new_zeros(())
    active_sum = student.new_zeros(())
    element_count = 0
    for start in range(0, student.shape[1], query_frame_chunk):
        stop = min(start + query_frame_chunk, student.shape[1])
        student_relation = torch.einsum(
            "bqsd,bnd->bqsn", student[:, start:stop], student_all
        )
        teacher_relation = torch.einsum(
            "bqsd,bnd->bqsn", teacher[:, start:stop], teacher_all
        )
        absolute_error = (student_relation - teacher_relation).abs()
        hinge = F.relu(absolute_error - float(margin))
        hinge_sum = hinge_sum + hinge.sum()
        absolute_sum = absolute_sum + absolute_error.sum()
        active_sum = active_sum + (absolute_error > float(margin)).float().sum()
        element_count += hinge.numel()

    denominator = float(max(element_count, 1))
    return hinge_sum / denominator, {
        "relation_mae": (absolute_sum / denominator).detach(),
        "active_relation_fraction": (active_sum / denominator).detach(),
        "student_std": student.std().detach(),
        "teacher_std": teacher.std().detach(),
    }


class RepresentationAlignmentLoss(nn.Module):
    """Frozen-foundation representation alignment for one DFoT feature map."""

    def __init__(
        self,
        *,
        method: str,
        student_channels: int = 576,
        projector_hidden_dim: int = 2048,
        teacher_dim: int = 768,
        teacher_input_size: int = 224,
        projector_checkpointing: bool = True,
        projector_frame_chunk: int = 2,
        dino_frame_chunk: int = 8,
        videomae_batch_chunk: int = 1,
        dino_model_name: str = "dinov2_vitb14",
        dino_source_dir: str = "./metric_cache/torch/hub/facebookresearch_dinov2_main",
        dino_checkpoint: str = "./model_cache/dinov2/dinov2_vitb14_pretrain.pth",
        videorepa_source_file: str = "./references/VideoREPA/finetune/models/cogvideox_t2v_align/models/ssl/VideoMAEv2.py",
        videomae_checkpoint: str = "./model_cache/VideoMAEv2/vit_b_k710_dl_from_giant.pth",
        trd_margin: float = 0.1,
        trd_query_frame_chunk: int = 1,
    ) -> None:
        super().__init__()
        method = str(method).lower().replace("-", "_")
        if method not in {"repa", "video_repa", "videorepa"}:
            raise ValueError(f"Unsupported representation alignment method: {method}")
        self.method = "video_repa" if method in {"video_repa", "videorepa"} else "repa"
        self.student_channels = int(student_channels)
        self.teacher_dim = int(teacher_dim)
        self.teacher_input_size = int(teacher_input_size)
        self.projector_checkpointing = bool(projector_checkpointing)
        self.projector_frame_chunk = max(int(projector_frame_chunk), 1)
        self.dino_frame_chunk = max(int(dino_frame_chunk), 1)
        self.videomae_batch_chunk = max(int(videomae_batch_chunk), 1)
        self.dino_model_name = str(dino_model_name)
        self.dino_source_dir = str(dino_source_dir)
        self.dino_checkpoint = str(dino_checkpoint)
        self.videorepa_source_file = str(videorepa_source_file)
        self.videomae_checkpoint = str(videomae_checkpoint)
        self.trd_margin = float(trd_margin)
        self.trd_query_frame_chunk = max(int(trd_query_frame_chunk), 1)
        self.projector = TokenProjector(
            self.student_channels,
            int(projector_hidden_dim),
            self.teacher_dim,
        )
        object.__setattr__(self, "_teacher", None)

    @property
    def teacher(self) -> nn.Module | None:
        return object.__getattribute__(self, "_teacher")

    def _check_asset(self, path: str, label: str) -> Path:
        asset = Path(path).expanduser().resolve()
        if not asset.is_file() or asset.stat().st_size == 0:
            raise FileNotFoundError(
                f"Missing {label}: {asset}. Run "
                "scripts/download_representation_baseline_assets.sh first."
            )
        return asset

    def _load_dino_teacher(self) -> nn.Module:
        source = Path(self.dino_source_dir).expanduser().resolve()
        checkpoint = self._check_asset(self.dino_checkpoint, "DINOv2 checkpoint")
        if not (source / "hubconf.py").is_file():
            raise FileNotFoundError(
                f"Missing local DINOv2 source repository: {source}. Run "
                "scripts/download_representation_baseline_assets.sh first."
            )
        model = torch.hub.load(
            str(source),
            self.dino_model_name,
            source="local",
            pretrained=False,
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state, strict=True)
        return model

    def _load_videomae_teacher(self) -> nn.Module:
        source_file = self._check_asset(
            self.videorepa_source_file, "official VideoREPA VideoMAEv2 source"
        )
        checkpoint = self._check_asset(
            self.videomae_checkpoint, "VideoMAEv2 checkpoint"
        )
        module_name = "upcast_official_videomaev2"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, source_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to import VideoMAEv2 from {source_file}")
            module = importlib.util.module_from_spec(spec)
            # timm's register_model decorator resolves the defining module
            # while the file executes, so it must already be discoverable.
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
        model = module.vit_base_patch16_224(
            all_frames=16,
            align_video_resolution=(self.teacher_input_size, self.teacher_input_size),
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "module" in payload:
            state = payload["module"]
            prefix = "module."
        elif "model" in payload:
            state = payload["model"]
            prefix = "encoder."
        else:
            state = payload
            prefix = ""
        state = {
            (key[len(prefix) :] if prefix and key.startswith(prefix) else key): value
            for key, value in state.items()
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed_unexpected = {
            key for key in unexpected if key.startswith(("head.", "fc_norm."))
        }
        unexpected = [key for key in unexpected if key not in allowed_unexpected]
        if missing or unexpected:
            raise RuntimeError(
                "VideoMAEv2 checkpoint is incompatible with the official base encoder: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return model

    def _ensure_teacher(self, device: torch.device) -> nn.Module:
        teacher = self.teacher
        if teacher is None:
            teacher = (
                self._load_dino_teacher()
                if self.method == "repa"
                else self._load_videomae_teacher()
            )
            teacher.requires_grad_(False)
            teacher.eval()
            teacher.to(device)
            object.__setattr__(self, "_teacher", teacher)
        else:
            parameter = next(teacher.parameters())
            if parameter.device != device:
                teacher.to(device)
        teacher.eval()
        return teacher

    @staticmethod
    def _normalize_rgb(video: Tensor) -> Tensor:
        mean = video.new_tensor(_IMAGENET_MEAN).view(1, 1, 3, 1, 1)
        std = video.new_tensor(_IMAGENET_STD).view(1, 1, 3, 1, 1)
        return (video.clamp(0.0, 1.0) - mean) / std

    def _project_student(self, latent: Tensor) -> Tensor:
        if latent.ndim != 5 or latent.shape[2] != self.student_channels:
            raise ValueError(
                "Expected DFoT latent [B,T,C,H,W] with C="
                f"{self.student_channels}, got {tuple(latent.shape)}"
            )
        projected_frames = []
        for start in range(0, latent.shape[1], self.projector_frame_chunk):
            frame_chunk = latent[:, start : start + self.projector_frame_chunk]
            tokens = rearrange(frame_chunk, "b t c h w -> b t h w c")
            if self.projector_checkpointing and torch.is_grad_enabled():
                projected = activation_checkpoint(
                    self.projector,
                    tokens,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                projected = self.projector(tokens)
            projected_frames.append(projected)
        return torch.cat(projected_frames, dim=1)

    @torch.no_grad()
    def _dino_tokens(self, video: Tensor) -> Tensor:
        teacher = self._ensure_teacher(video.device)
        batch, frames = video.shape[:2]
        images = rearrange(self._normalize_rgb(video), "b t c h w -> (b t) c h w")
        images = F.interpolate(
            images,
            size=(self.teacher_input_size, self.teacher_input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        token_chunks = []
        for images_chunk in images.split(self.dino_frame_chunk, dim=0):
            features: Any = teacher.forward_features(images_chunk)
            if not isinstance(features, dict) or "x_norm_patchtokens" not in features:
                raise RuntimeError("DINOv2 did not return x_norm_patchtokens")
            token_chunks.append(features["x_norm_patchtokens"])
        tokens = torch.cat(token_chunks, dim=0)
        grid = math.isqrt(tokens.shape[1])
        if grid * grid != tokens.shape[1]:
            raise RuntimeError(f"DINOv2 patch count is not square: {tokens.shape[1]}")
        return rearrange(tokens, "(b t) (h w) d -> b t h w d", b=batch, t=frames, h=grid)

    @torch.no_grad()
    def _videomae_tokens(self, video: Tensor) -> Tensor:
        teacher = self._ensure_teacher(video.device)
        normalized = self._normalize_rgb(video)
        normalized = rearrange(normalized, "b t c h w -> (b t) c h w")
        normalized = F.interpolate(
            normalized,
            size=(self.teacher_input_size, self.teacher_input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        normalized = rearrange(
            normalized,
            "(b t) c h w -> b c t h w",
            b=video.shape[0],
            t=video.shape[1],
        )
        token_chunks = []
        for video_chunk in normalized.split(self.videomae_batch_chunk, dim=0):
            token_chunks.append(teacher(video_chunk))
        tokens = torch.cat(token_chunks, dim=0)
        temporal = video.shape[1] // 2
        spatial = self.teacher_input_size // 16
        expected = temporal * spatial * spatial
        if tokens.shape[1] != expected:
            raise RuntimeError(
                "VideoMAEv2 token count mismatch: "
                f"expected {expected}, got {tokens.shape[1]}"
            )
        return rearrange(
            tokens,
            "b (t h w) d -> b t h w d",
            t=temporal,
            h=spatial,
            w=spatial,
        )

    def forward(self, latent: Tensor, clean_video: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if clean_video.ndim != 5 or clean_video.shape[2] != 3:
            raise ValueError(
                f"Expected clean RGB video [B,T,3,H,W], got {tuple(clean_video.shape)}"
            )
        if latent.shape[:2] != clean_video.shape[:2]:
            raise ValueError(
                "Student and teacher videos must share batch/time dimensions: "
                f"{tuple(latent.shape[:2])} vs {tuple(clean_video.shape[:2])}"
            )
        student = self._project_student(latent)
        student = rearrange(student, "b t h w d -> b d t h w")

        if self.method == "repa":
            teacher = self._dino_tokens(clean_video)
            target_shape = teacher.shape[1:4]
            student = F.interpolate(
                student.float(),
                size=target_shape,
                mode="trilinear",
                align_corners=False,
            )
            student = rearrange(student, "b d t h w -> b t h w d")
            loss, metrics = repa_cosine_loss(student, teacher)
        else:
            teacher = self._videomae_tokens(clean_video)
            target_shape = teacher.shape[1:4]
            student = F.interpolate(
                student.float(),
                size=target_shape,
                mode="trilinear",
                align_corners=False,
            )
            student = rearrange(student, "b d t h w -> b t (h w) d")
            teacher = rearrange(teacher, "b t h w d -> b t (h w) d")
            loss, metrics = token_relation_distillation_loss(
                student,
                teacher,
                margin=self.trd_margin,
                query_frame_chunk=self.trd_query_frame_chunk,
            )

        metrics = {f"representation_{name}": value for name, value in metrics.items()}
        metrics["representation_loss"] = loss.detach()
        return loss, metrics

    def extra_repr(self) -> str:
        return (
            f"method={self.method}, student_channels={self.student_channels}, "
            f"teacher_dim={self.teacher_dim}"
        )
