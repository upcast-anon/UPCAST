#!/usr/bin/env python3
"""Shared I/O helpers for generated-video evaluation artifacts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be NAME=INPUT_DIR")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("--run name cannot be empty")
    return name, Path(path).expanduser()


def parse_indices(value: str | None) -> set[int] | None:
    if not value:
        return None
    indices: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indices.update(range(int(start), int(end) + 1))
        else:
            indices.add(int(part))
    return indices


def video_index(path: Path) -> int | None:
    match = re.match(r"video_(\d+)(?:_|\.|$)", path.name)
    return int(match.group(1)) if match else None


def _latest_gifs(files: list[Path]) -> list[Path]:
    latest: dict[int, tuple[int, Path]] = {}
    pattern = re.compile(r"video_(\d+)_(\d+)_([a-f0-9]+)\.gif")
    for path in files:
        match = pattern.match(path.name)
        if not match:
            continue
        index = int(match.group(1))
        log_index = int(match.group(2))
        if index not in latest or log_index > latest[index][0]:
            latest[index] = (log_index, path)
    if latest:
        return [entry[1] for _, entry in sorted(latest.items())]
    return sorted(files)


def iter_saved_pairs(
    input_dir: Path,
    pattern: str | None,
    indices: set[int] | None,
    max_files: int | None,
    input_format: str = "auto",
) -> list[Path]:
    input_dir = Path(input_dir)
    if input_format not in {"auto", "gif", "npz"}:
        raise ValueError(f"Unsupported input format: {input_format}")
    if input_format == "auto":
        input_format = "npz" if list(input_dir.glob("video_*.npz")) else "gif"
    glob_pattern = pattern or ("video_*.npz" if input_format == "npz" else "*.gif")
    files = sorted(input_dir.glob(glob_pattern))
    if input_format == "gif":
        files = _latest_gifs(files)
    if indices is not None:
        files = [path for path in files if video_index(path) in indices]
    if max_files is not None:
        files = files[:max_files]
    return files


def _to_thwc_uint8(array: np.ndarray, key: str, path: Path) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 4:
        raise RuntimeError(f"{key} in {path} must be rank four, got {value.shape}")
    if value.shape[-1] not in {1, 3, 4} and value.shape[1] in {1, 3, 4}:
        value = np.transpose(value, (0, 2, 3, 1))
    if value.shape[-1] == 1:
        value = np.repeat(value, 3, axis=-1)
    elif value.shape[-1] == 4:
        value = value[..., :3]
    if value.shape[-1] != 3:
        raise RuntimeError(f"Cannot infer channels for {key} in {path}: {value.shape}")
    if value.dtype != np.uint8:
        value = value.astype(np.float32)
        if value.size and float(np.nanmax(value)) <= 1.0001:
            value = value * 255.0
        value = np.clip(value, 0.0, 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(value)


def load_saved_pair(path: Path, max_frames: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as payload:
            pred_key = "gen" if "gen" in payload else "pred"
            if pred_key not in payload or "gt" not in payload:
                raise KeyError(f"{path} must contain gen/pred and gt arrays")
            pred = _to_thwc_uint8(payload[pred_key], pred_key, path)
            gt = _to_thwc_uint8(payload["gt"], "gt", path)
    else:
        frames = iio.imread(path)
        if frames.ndim == 3:
            frames = frames[None, ...]
        frames = _to_thwc_uint8(frames, "gif", path)
        midpoint = frames.shape[2] // 2
        pred, gt = frames[:, :, :midpoint].copy(), frames[:, :, midpoint:].copy()
    frames = min(pred.shape[0], gt.shape[0])
    if max_frames is not None and max_frames > 0:
        frames = min(frames, int(max_frames))
    if frames <= 0:
        raise RuntimeError(f"No frames found in {path}")
    return pred[:frames], gt[:frames]
