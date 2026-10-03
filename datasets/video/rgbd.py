"""Prepared RGB-D clips for zero-shot camera-conditioned evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from .base_video import BaseAdvancedVideoDataset, SPLIT


class RGBDSensorVideoDataset(BaseAdvancedVideoDataset):
    """Read synchronized lossless clips produced by a RGB-D preprocessor.

    This dataset is evaluation-only. Each archive stores synchronized RGB,
    camera vectors, sensor depth, and metric camera poses. Keeping those arrays
    together makes it impossible to silently pair generated videos with the
    wrong geometric ground truth.
    """

    _ALL_SPLITS = ["test"]

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "test",
        current_epoch: Optional[int] = None,
    ):
        if split == "validation":
            split = "test"
        if split == "training":
            raise ValueError("RGB-D data is evaluation-only")
        split_dir = Path(cfg.save_dir) / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Prepared RGB-D split not found: {split_dir}. Run "
                "evaluation/prepare_arkitscenes.py first."
            )
        super().__init__(cfg, split, current_epoch)

    def _should_download(self) -> bool:
        return False

    def download_dataset(self) -> None:
        raise RuntimeError(
            "Download the official RGB-D release and run its preprocessor."
        )

    def build_metadata(self, split: SPLIT) -> None:
        manifest_path = self.save_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            paths = []
            for clip in manifest.get("clips", []):
                archive = Path(str(clip["archive"]))
                if not archive.is_file():
                    archive = self.save_dir / split / archive.name
                if not archive.is_file():
                    raise FileNotFoundError(
                        f"Manifest archive is missing for {clip.get('clip_id')}: {archive}"
                    )
                paths.append(archive)
        else:
            paths = sorted((self.save_dir / split).glob("*.npz"), key=str)
        if not paths:
            raise FileNotFoundError(f"No prepared RGB-D clips in {self.save_dir / split}")

        video_pts: List[torch.Tensor] = []
        video_fps: List[float] = []
        scene_ids: List[str] = []
        clip_ids: List[str] = []
        for path in paths:
            with np.load(path, allow_pickle=False) as payload:
                video = payload["video"]
                camera = payload["camera"]
                if video.ndim != 4 or video.shape[1] != 3:
                    raise ValueError(f"{path}: video must be [T,3,H,W], got {video.shape}")
                if camera.shape != (video.shape[0], 16):
                    raise ValueError(
                        f"{path}: camera must be [T,16], got {camera.shape} for {len(video)} frames"
                    )
                video_pts.append(torch.arange(video.shape[0], dtype=torch.long))
                video_fps.append(float(payload["fps"].item()))
                scene_ids.append(str(payload["scene_id"].item()))
                clip_ids.append(str(payload["clip_id"].item()))

        metadata: Dict[str, Any] = {
            "video_paths": paths,
            "video_pts": video_pts,
            "video_fps": video_fps,
            "scene_ids": scene_ids,
            "clip_ids": clip_ids,
        }
        torch.save(metadata, self.metadata_dir / f"{split}.pt")

    def load_video(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        with np.load(video_metadata["video_paths"], allow_pickle=False) as payload:
            video = payload["video"][start_frame:end_frame].copy()
        return torch.from_numpy(video).float().div_(255.0)

    def load_cond(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: int,
    ) -> torch.Tensor:
        with np.load(video_metadata["video_paths"], allow_pickle=False) as payload:
            camera = payload["camera"][start_frame:end_frame].copy()
        return torch.from_numpy(camera).float()

    def _process_external_cond(
        self,
        external_cond: torch.Tensor,
        frame_skip: Optional[int] = None,
    ) -> torch.Tensor:
        return external_cond[:: frame_skip or self.frame_skip].to(torch.float32)

    def _build_idx_remap(self) -> List[int]:
        # Prepared RGB-D archives each represent one pre-registered trajectory.
        # Preserve manifest order so output index maps directly back to GT.
        starts = [0] + self.cumulative_sizes[:-1]
        if self.use_evaluation_subdataset:
            offset = int(getattr(self.cfg, "eval_video_offset", 0) or 0)
            return starts[offset : offset + int(self.cfg.num_eval_videos)]
        return starts
