"""Prepared ARKitScenes clips for zero-shot camera-conditioned evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from omegaconf import DictConfig

from .base_video import BaseAdvancedVideoDataset, SPLIT
from .rgbd import RGBDSensorVideoDataset


class ARKitScenesAdvancedVideoDataset(RGBDSensorVideoDataset):
    """Read lossless archives produced by ``prepare_arkitscenes.py``."""

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "test",
        current_epoch: Optional[int] = None,
    ):
        if split == "validation":
            split = "test"
        if split == "training":
            raise ValueError("ARKitScenes data is evaluation-only")
        split_dir = Path(cfg.save_dir) / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Prepared ARKitScenes split not found: {split_dir}. Run "
                "evaluation/prepare_arkitscenes.py first."
            )
        BaseAdvancedVideoDataset.__init__(self, cfg, split, current_epoch)

    def download_dataset(self) -> None:
        raise RuntimeError(
            "Download the public ARKitScenes raw RGB-D assets with "
            "scripts/download_arkitscenes.sh, then run the preprocessor."
        )
