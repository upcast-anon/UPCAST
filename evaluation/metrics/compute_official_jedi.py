#!/usr/bin/env python3
"""Compute official JEDi MMD on saved short-video W&B GIFs.

The GIFs must use the repository's ``prediction | ground truth`` layout.  This
entry point keeps the official JEDi V-JEPA attentive-probe feature extractor
and polynomial MMD, while allowing a ViT-L/16 + SSv2 probe variant and explicit
removal of copied context frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics.video_helpers import (
    iter_gifs,
    load_side_by_side_pair,
    parse_indices,
    parse_run,
    sample_or_pad_video,
    video_index,
)


class GifHalfDataset(Dataset):
    def __init__(
        self,
        gif_files: list[Path],
        source: str,
        context_frames: int,
        num_frames: int,
        input_size: int,
        max_frames: int | None = None,
    ) -> None:
        if source not in {"pred", "gt"}:
            raise ValueError(f"Unsupported source: {source}")
        self.gif_files = gif_files
        self.source = source
        self.context_frames = max(0, int(context_frames))
        self.num_frames = int(num_frames)
        self.input_size = int(input_size)
        self.max_frames = None if max_frames is None else int(max_frames)

    def __len__(self) -> int:
        return len(self.gif_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        pred, gt = load_side_by_side_pair(self.gif_files[index], max_frames=self.max_frames)
        video = pred if self.source == "pred" else gt
        video = video[self.context_frames :]
        if video.shape[0] == 0:
            raise RuntimeError(
                f"context_frames={self.context_frames} leaves no frames in {self.gif_files[index]}"
            )
        tensor = torch.from_numpy(video).float().permute(0, 3, 1, 2) / 255.0
        tensor = sample_or_pad_video(tensor, self.num_frames)
        tensor = F.interpolate(
            tensor,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        return tensor.clamp_(0.0, 1.0), index


@torch.inference_mode()
def extract_features(model: Any, loader: DataLoader, count: int) -> np.ndarray:
    features: list[np.ndarray] = []
    seen = 0
    for videos, _ in loader:
        videos = videos.cuda(non_blocking=True)
        batch_features = np.asarray(model.get_feats(videos), dtype=np.float32)
        features.append(batch_features)
        seen += batch_features.shape[0]
        if seen >= count:
            break
    if not features:
        raise RuntimeError("No JEDi features were extracted.")
    return np.concatenate(features, axis=0)[:count]


def paired_cosine_distance(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred_norm = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    gt_norm = gt / np.maximum(np.linalg.norm(gt, axis=1, keepdims=True), 1e-12)
    return 1.0 - np.sum(pred_norm * gt_norm, axis=1)


def write_rows(path: Path, gif_files: list[Path], distances: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "gif_file", "jedi_pair_cosine_distance"],
        )
        writer.writeheader()
        for gif_path, distance in zip(gif_files, distances.tolist()):
            writer.writerow(
                {
                    "index": video_index(gif_path),
                    "gif_file": gif_path.name,
                    "jedi_pair_cosine_distance": float(distance),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Official JEDi evaluation for short side-by-side GIFs.")
    parser.add_argument("--run", action="append", type=parse_run, required=True, metavar="NAME=GIF_DIR")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*.gif")
    parser.add_argument("--indices", type=str, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--context_frames", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--max_frames", type=int, default=None, help="Sample the probe clip from this temporal prefix.")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./metric_assets/jedi_official_vitl",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(PROJECT_ROOT / "external/jepa/configs/evals/vitl16_ssv2_16x2x3.yaml"),
    )
    parser.add_argument(
        "--package_path",
        type=str,
        default="./conda_envs/jedi",
    )
    args = parser.parse_args()

    package_path = Path(args.package_path).expanduser().resolve()
    jedi_source = PROJECT_ROOT / "external/JEDi"
    for path in [package_path, jedi_source]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from videojedi.V_JEPA import VJEPA
    from videojedi.mmd_polynomial import mmd_poly
    from videojedi.utils import model_cleanup

    if not torch.cuda.is_available():
        raise RuntimeError("Official JEDi feature extraction requires CUDA in the upstream implementation.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_indices = parse_indices(args.indices)
    model = VJEPA(
        model_dir=str(Path(args.model_dir).expanduser()),
        config_fname=str(Path(args.config_path).expanduser()),
        finetuned=True,
    )

    results: dict[str, Any] = {}
    try:
        for name, gif_dir in args.run:
            gif_files = iter_gifs(gif_dir, args.pattern, selected_indices, args.max_files)
            if not gif_files:
                raise FileNotFoundError(f"No GIFs selected for {name}: {gif_dir}")
            run_dir = output_dir / name
            run_dir.mkdir(parents=True, exist_ok=True)
            common = {
                "gif_files": gif_files,
                "context_frames": args.context_frames,
                "num_frames": args.num_frames,
                "input_size": args.input_size,
                "max_frames": args.max_frames,
            }
            pred_loader = DataLoader(
                GifHalfDataset(source="pred", **common),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            gt_loader = DataLoader(
                GifHalfDataset(source="gt", **common),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            print(f"Extracting official JEDi features for {name}: {len(gif_files)} videos")
            pred_features = extract_features(model, pred_loader, len(gif_files))
            gt_features = extract_features(model, gt_loader, len(gif_files))
            np.save(run_dir / "pred_features.npy", pred_features)
            np.save(run_dir / "gt_features.npy", gt_features)
            pair_distance = paired_cosine_distance(pred_features, gt_features)
            write_rows(run_dir / "per_video.csv", gif_files, pair_distance)
            results[name] = {
                "gif_dir": str(gif_dir),
                "num_videos": len(gif_files),
                "indices": [video_index(path) for path in gif_files],
                "jedi_official_mmd2": float(mmd_poly(pred_features, gt_features, degree=2, coef0=0) * 100.0),
                "jedi_pair_cosine_distance_mean": float(pair_distance.mean()),
                "jedi_pair_cosine_distance_std": float(pair_distance.std()),
                "feature_dim": int(pred_features.shape[1]),
            }
    finally:
        model_cleanup(model)

    report = {
        "metric": "Official JEDi polynomial MMD with V-JEPA-L/16 + SSv2 attentive probe",
        "config": {
            "context_frames": args.context_frames,
            "num_frames": args.num_frames,
            "max_frames": args.max_frames,
            "input_size": args.input_size,
            "model_dir": args.model_dir,
            "config_path": args.config_path,
        },
        "results": results,
    }
    with (output_dir / "official_jedi_report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    with (output_dir / "official_jedi_summary.csv").open("w", newline="") as handle:
        fieldnames = [
            "name",
            "num_videos",
            "jedi_official_mmd2",
            "jedi_pair_cosine_distance_mean",
            "jedi_pair_cosine_distance_std",
            "feature_dim",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, metrics in results.items():
            writer.writerow({"name": name, **{key: metrics[key] for key in fieldnames if key != "name"}})
    print(json.dumps(results, indent=2))
    print(f"Saved official JEDi report to {output_dir}")


if __name__ == "__main__":
    main()
