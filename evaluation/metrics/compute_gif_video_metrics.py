#!/usr/bin/env python3
"""Compute video-generation metrics from saved side-by-side W&B GIFs.

The input GIF layout is assumed to be ``prediction | gt``.  This is intended as
a fair post-hoc comparison for already-saved long-video samples; it is not a
replacement for metrics computed from raw validation tensors because GIF
encoding can change pixel-level scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.functional.image import structural_similarity_index_measure

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.common.metrics.video import SharedVideoMetricModelRegistry, VideoMetric
from algorithms.common.metrics.video.types import VideoMetricModelType, VideoMetricType
from evaluation.metrics.saved_video_io import (
    iter_saved_pairs,
    load_saved_pair,
    parse_indices,
    parse_run,
    video_index,
)


def iter_gifs(gif_dir: Path, pattern: str, indices: set[int] | None, max_files: int | None) -> list[Path]:
    return iter_saved_pairs(gif_dir, pattern, indices, max_files, input_format="gif")


def load_pair(
    path: Path,
    device: torch.device,
    size: int | None,
    max_frames: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen_np, gt_np = load_saved_pair(path, max_frames=max_frames)
    gen = torch.from_numpy(gen_np).to(device=device, dtype=torch.float32) / 255.0
    gt = torch.from_numpy(gt_np).to(device=device, dtype=torch.float32) / 255.0
    gen = rearrange(gen, "t h w c -> 1 t c h w")
    gt = rearrange(gt, "t h w c -> 1 t c h w")
    if size is not None and size > 0 and gen.shape[-1] != size:
        b, t, c, h, w = gen.shape
        gen_2d = rearrange(gen, "b t c h w -> (b t) c h w")
        gt_2d = rearrange(gt, "b t c h w -> (b t) c h w")
        gen_2d = F.interpolate(gen_2d, size=(size, size), mode="bilinear", align_corners=False)
        gt_2d = F.interpolate(gt_2d, size=(size, size), mode="bilinear", align_corners=False)
        gen = rearrange(gen_2d, "(b t) c h w -> b t c h w", b=b, t=t)
        gt = rearrange(gt_2d, "(b t) c h w -> b t c h w", b=b, t=t)
    return gen.clamp(0, 1), gt.clamp(0, 1)


def update_frame_metrics(
    pred_frames: torch.Tensor,
    gt_frames: torch.Tensor,
    registry: SharedVideoMetricModelRegistry,
    state: dict[str, Any],
    chunk_size: int,
) -> None:
    for start in range(0, pred_frames.shape[0], chunk_size):
        pred = pred_frames[start : start + chunk_size].contiguous()
        gt = gt_frames[start : start + chunk_size].contiguous()
        state["ssim"].update(pred, gt)
        state["psnr"].update(pred, gt)
        sq = (pred - gt).pow(2)
        state["mse_sum"] += float(sq.sum().item())
        state["pixel_count"] += int(sq.numel())
        with torch.no_grad():
            lpips = registry(VideoMetricModelType.LPIPS, pred, gt, normalize=True).reshape(-1)
        state["lpips_sum"] += float(lpips.sum().item())
        state["lpips_count"] += int(lpips.numel())


def compute_one_run(
    name: str,
    gif_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    saved_files = iter_saved_pairs(
        gif_dir,
        args.pattern,
        args.indices,
        args.max_files,
        input_format=args.input_format,
    )
    if not saved_files:
        raise FileNotFoundError(f"No saved video pairs selected for {name}: {gif_dir}")

    registry = SharedVideoMetricModelRegistry().to(device)
    video_metric = None
    if args.compute_distribution_metrics:
        metric_types = [VideoMetricType.FVD]
        if args.include_fid:
            metric_types.append(VideoMetricType.FID)
        video_metric = VideoMetric(
            registry=registry,
            metric_types=metric_types,
            split_batch_size=1,
        ).to(device).eval()
    registry.register_for_metric(VideoMetricType.LPIPS)
    registry[VideoMetricModelType.LPIPS].to(device)
    frame_state: dict[str, Any] = {
        "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(device),
        "psnr": PeakSignalNoiseRatio(data_range=1.0).to(device),
        "mse_sum": 0.0,
        "pixel_count": 0,
        "lpips_sum": 0.0,
        "lpips_count": 0,
    }

    per_video = []
    for saved_path in saved_files:
        print(f"[{name}] {saved_path.name}", flush=True)
        pred, gt = load_pair(saved_path, device, args.resize, args.max_frames)
        t = pred.shape[1]
        context = min(max(0, args.context_frames), t)
        context_mask = torch.zeros(t, dtype=torch.bool, device=device)
        if context:
            context_mask[:context] = True

        if video_metric is not None:
            with torch.no_grad():
                video_metric(pred, gt, context_mask=context_mask)

        pred_frames = rearrange(pred[:, ~context_mask], "b t c h w -> (b t) c h w")
        gt_frames = rearrange(gt[:, ~context_mask], "b t c h w -> (b t) c h w")
        before_mse_sum = frame_state["mse_sum"]
        before_pixels = frame_state["pixel_count"]
        before_lpips_sum = frame_state["lpips_sum"]
        before_lpips_count = frame_state["lpips_count"]
        update_frame_metrics(pred_frames, gt_frames, registry, frame_state, args.frame_chunk_size)
        per_mse = (frame_state["mse_sum"] - before_mse_sum) / max(1, frame_state["pixel_count"] - before_pixels)
        per_lpips = (frame_state["lpips_sum"] - before_lpips_sum) / max(1, frame_state["lpips_count"] - before_lpips_count)
        with torch.no_grad():
            per_ssim = float(
                structural_similarity_index_measure(
                    pred_frames,
                    gt_frames,
                    data_range=1.0,
                ).item()
            )
        per_video.append(
            {
                "index": video_index(saved_path),
                "input_file": saved_path.name,
                "frames": int(t),
                "eval_frames": int(pred_frames.shape[0]),
                "mse": per_mse,
                "psnr_from_mse": 10.0 * math.log10(1.0 / max(per_mse, 1e-12)),
                "lpips": per_lpips,
                "ssim": per_ssim,
            }
        )

    aggregate: dict[str, float | int | list[int | None]] = {
        "num_videos": len(saved_files),
        "indices": [video_index(path) for path in saved_files],
        "mse": frame_state["mse_sum"] / max(1, frame_state["pixel_count"]),
        "lpips": frame_state["lpips_sum"] / max(1, frame_state["lpips_count"]),
        "ssim": float(frame_state["ssim"].compute().item()),
        "psnr": float(frame_state["psnr"].compute().item()),
    }
    if video_metric is not None:
        for metric_type, module in video_metric.items():
            value = module.compute()
            key = metric_type.value if hasattr(metric_type, "value") else str(metric_type)
            aggregate[key] = float(value.item() if torch.is_tensor(value) else value)
    return {"name": name, "gif_dir": str(gif_dir), "aggregate": aggregate, "per_video": per_video}


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    rows = []
    for result in results:
        row = {"name": result["name"], **result["aggregate"]}
        if isinstance(row.get("indices"), list):
            row["indices"] = ",".join(str(x) for x in row["indices"])
        rows.append(row)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="NAME=GIF_DIR")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--indices", type=str, default="0-11")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--pattern", type=str, default=None)
    parser.add_argument("--input_format", choices=["auto", "gif", "npz"], default="auto")
    parser.add_argument("--max_frames", type=int, default=None, help="Evaluate only this temporal prefix.")
    parser.add_argument("--context_frames", type=int, default=1)
    parser.add_argument("--resize", type=int, default=None, help="Optional square resize before metrics.")
    parser.add_argument("--frame_chunk_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include_fid", action="store_true", help="Also compute FID from frames.")
    parser.add_argument(
        "--compute_distribution_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute FVD/FID; disable for one-sample smoke tests.",
    )
    args = parser.parse_args()
    args.indices = parse_indices(args.indices)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    results = [compute_one_run(name, path, args, device) for name, path in args.run]
    config = {**vars(args), "run": [(name, str(path)) for name, path in args.run]}
    if isinstance(config.get("indices"), set):
        config["indices"] = sorted(config["indices"])
    report = {"config": config, "runs": results}
    with (output_dir / "gif_video_metrics_report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    write_csv(output_dir / "gif_video_metrics_summary.csv", results)
    print(json.dumps({r["name"]: r["aggregate"] for r in results}, indent=2))
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
