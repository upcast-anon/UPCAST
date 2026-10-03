from dataclasses import dataclass
from pathlib import Path
import sys
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from utils.huggingface_utils import local_hfd_repo_dir


@dataclass
class GeometryTeacherOutput:
    target: Tensor
    valid_mask: Tensor | None = None
    confidence: Tensor | None = None
    features: Tensor | None = None
    patch_start_idx: int | None = None


def _as_hw(size: int | list[int] | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        return int(size), int(size)
    if len(size) != 2:
        raise ValueError(f"Expected int or (height, width), got {size!r}")
    return int(size[0]), int(size[1])


def _resize_btchw(x: Tensor, size: tuple[int, int]) -> Tensor:
    if x.shape[-2:] == size:
        return x
    b, t, c, h, w = x.shape
    x_2d = rearrange(x, "b t c h w -> (b t) c h w")
    mode = "area" if size[0] <= h and size[1] <= w else "bilinear"
    if mode == "area":
        y = F.interpolate(x_2d, size=size, mode=mode)
    else:
        y = F.interpolate(x_2d, size=size, mode=mode, align_corners=False)
    return rearrange(y, "(b t) c h w -> b t c h w", b=b)


def _resize_bthw(x: Tensor, size: tuple[int, int]) -> Tensor:
    if x.shape[-2:] == size:
        return x
    b, t, h, w = x.shape
    x_2d = rearrange(x, "b t h w -> (b t) 1 h w")
    mode = "area" if size[0] <= h and size[1] <= w else "bilinear"
    if mode == "area":
        y = F.interpolate(x_2d, size=size, mode=mode)
    else:
        y = F.interpolate(x_2d, size=size, mode=mode, align_corners=False)
    return rearrange(y, "(b t) 1 h w -> b t h w", b=b)


class BaseGeometryTeacher(nn.Module):
    output_channels: int

    def forward(self, images: Tensor) -> GeometryTeacherOutput:
        raise NotImplementedError


class WorldPointGeometryTeacher(BaseGeometryTeacher):
    def _preprocess(self, images: Tensor) -> Tensor:
        b = images.shape[0]
        images = rearrange(images, "b t c h w -> (b t) c h w")
        images = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)
        images = rearrange(images, "(b t) c h w -> b t c h w", b=b)
        return torch.clamp(images, 0.0, 1.0)

    def _normalize_points(self, points: Tensor) -> Tensor:
        points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.normalize:
            return points
        center = points[:, :1].mean(dim=(3, 4), keepdim=True)
        points = points - center
        scale = points[:, :1].norm(dim=2, keepdim=True).mean(dim=(3, 4), keepdim=True)
        return points / scale.clamp(min=1e-6)

class Page4DGeometryTeacher(WorldPointGeometryTeacher):
    """PAGE-4D adapter exposing patch features, world points, and confidence."""

    def __init__(
        self,
        repo_id: str = "zhouk777/PAGE4D",
        checkpoint_path: str | None = "huggingface/zhouk777--PAGE4D/checkpoint_nomask.pt",
        representation: str = "point_map",
        target_size: int | list[int] | tuple[int, int] = (128, 128),
        normalize: bool = True,
        confidence_threshold: float | None = None,
    ):
        nn.Module.__init__(self)
        self.representation = str(representation or "point_map").lower()
        self.target_size = _as_hw(target_size)
        self.normalize = bool(normalize)
        self.confidence_threshold = (
            None if confidence_threshold is None else float(confidence_threshold)
        )

        if self.representation not in {"point", "points", "point_map", "world_points"}:
            raise ValueError(
                "UPCAST requires geometry_representation=point_map; "
                f"got {self.representation!r}. Framewise point differences are not scene flow."
            )
        self.output_channels = 3

        repo_root = Path(__file__).resolve().parents[1] / "external" / "page4d"
        model_root = repo_root / "model"
        if not model_root.exists():
            raise ImportError(
                f"PAGE-4D code not found at {model_root}. "
                "Clone https://github.com/kaichen-z/PAGE4D into external/page4d."
            )
        if str(model_root) not in sys.path:
            sys.path.insert(0, str(model_root))

        try:
            from page.models.vggt import VGGT as Page4DVGGT
        except Exception as exc:
            raise ImportError(
                "Failed to import PAGE-4D. Install its runtime dependencies from "
                "external/page4d/requirements.txt if needed."
            ) from exc

        checkpoint = checkpoint_path or local_hfd_repo_dir(repo_id)
        if checkpoint is None:
            checkpoint = "huggingface/zhouk777--PAGE4D/checkpoint_nomask.pt"
        checkpoint = Path(str(checkpoint)).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = Path.cwd() / checkpoint
        if checkpoint.is_dir():
            checkpoint = checkpoint / "checkpoint_nomask.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"PAGE-4D checkpoint not found: {checkpoint}. "
                "Run: hf download zhouk777/PAGE4D --include checkpoint_nomask.pt "
                "--local-dir huggingface/zhouk777--PAGE4D"
            )
        if not zipfile.is_zipfile(checkpoint):
            raise RuntimeError(
                f"PAGE-4D checkpoint is not a complete PyTorch zip archive: {checkpoint}. "
                "The previous download may have been truncated or over-appended; re-download it."
            )

        self.model = Page4DVGGT(
            enable_camera=False,
            enable_point=True,
            enable_depth=False,
            enable_track=False,
        )
        try:
            payload = torch.load(checkpoint, map_location="cpu", mmap=True)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        state_dict = payload.get("model", payload) if isinstance(payload, dict) else payload
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        critical_missing = [
            key
            for key in incompatible.missing_keys
            if key.startswith(("aggregator.", "point_head."))
        ]
        if critical_missing:
            raise RuntimeError(
                "PAGE-4D checkpoint is missing geometry-teacher weights: "
                + ", ".join(critical_missing[:10])
            )
        del payload, state_dict
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, images: Tensor) -> GeometryTeacherOutput:
        images = self._preprocess(images)
        aggregated_tokens_list, patch_start_idx = self.model.aggregator(images=images)
        predictions = {}
        with torch.amp.autocast(device_type=images.device.type, enabled=False):
            pts3d, pts3d_conf = self.model.point_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )
            predictions["world_points"] = pts3d
            predictions["world_points_conf"] = pts3d_conf

        points = predictions["world_points"]
        points = points.permute(0, 1, 4, 2, 3).contiguous()
        points = self._normalize_points(points)
        target = _resize_btchw(points, self.target_size)
        confidence = predictions.get("world_points_conf", None)
        valid_mask = None
        if confidence is not None:
            if confidence.ndim == 5 and confidence.shape[-1] == 1:
                confidence = confidence[..., 0]
            confidence = _resize_bthw(confidence, self.target_size)
            if self.confidence_threshold is not None:
                valid_mask = confidence >= self.confidence_threshold

        features = self._tokens_to_feature_map(
            aggregated_tokens_list[-1],
            patch_start_idx=patch_start_idx,
            image_hw=images.shape[-2:],
        )

        return GeometryTeacherOutput(
            target=target,
            valid_mask=valid_mask,
            confidence=confidence,
            features=features,
            patch_start_idx=patch_start_idx,
        )

    def _tokens_to_feature_map(
        self,
        tokens: Tensor,
        patch_start_idx: int,
        image_hw: tuple[int, int],
    ) -> Tensor:
        patch_tokens = tokens[:, :, patch_start_idx:]
        b, t, p, c = patch_tokens.shape
        patch_h = max(1, int(image_hw[0]) // int(self.model.aggregator.patch_size))
        patch_w = max(1, int(image_hw[1]) // int(self.model.aggregator.patch_size))
        if patch_h * patch_w != p:
            side = int(p ** 0.5)
            if side * side != p:
                raise RuntimeError(
                    f"Cannot reshape PAGE-4D patch tokens: tokens={p}, "
                    f"image_hw={image_hw}, patch_start_idx={patch_start_idx}."
                )
            patch_h = patch_w = side
        features = patch_tokens.reshape(b, t, patch_h, patch_w, c)
        return features.permute(0, 1, 4, 2, 3).contiguous()


def build_geometry_teacher(
    name: str,
    repo_id: str,
    checkpoint_path: str | None,
    representation: str,
    target_size: int | list[int] | tuple[int, int],
    normalize: bool,
    confidence_threshold: float | None,
) -> BaseGeometryTeacher:
    key = str(name or "vggt").lower()
    if key in {"page4d", "page-4d", "page_4d"}:
        return Page4DGeometryTeacher(
            repo_id=repo_id,
            checkpoint_path=checkpoint_path,
            representation=representation,
            target_size=target_size,
            normalize=normalize,
            confidence_threshold=confidence_threshold,
        )
    raise ValueError(f"Unknown geometry_teacher={name!r}. UPCAST requires page4d.")
