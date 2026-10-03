"""Video sampling and paired GIF loading for the official JEDi evaluator."""

from pathlib import Path

import numpy as np
import torch

from evaluation.metrics.saved_video_io import (
    iter_saved_pairs,
    load_saved_pair,
    parse_indices,
    parse_run,
    video_index,
)


def iter_gifs(gif_dir, pattern, indices, max_files):
    return iter_saved_pairs(gif_dir, pattern, indices, max_files, input_format="gif")


def load_side_by_side_pair(path: Path, max_frames=None):
    pred, gt = load_saved_pair(path, max_frames=max_frames)
    if pred.ndim != 4 or gt.ndim != 4:
        raise RuntimeError(f"Unexpected decoded GIF shape: {path}")
    return pred.astype(np.uint8), gt.astype(np.uint8)


def sample_or_pad_video(video: torch.Tensor, num_frames: int) -> torch.Tensor:
    length = int(video.shape[0])
    if length <= 0:
        raise ValueError("Cannot sample an empty video.")
    indices = torch.linspace(0, length - 1, num_frames, device=video.device).round().long()
    return video.index_select(0, indices)
