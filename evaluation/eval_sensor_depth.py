#!/usr/bin/env python3
"""Evaluate generated RGB-D clips against sensor depth ground truth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics.saved_video_io import (
    iter_saved_pairs,
    load_saved_pair,
    parse_indices,
    parse_run,
    video_index,
)


DEPTH_METRICS = (
    "abs_rel",
    "sq_rel",
    "rmse_m",
    "rmse_log",
    "log10",
    "silog",
    "delta1",
    "delta2",
    "delta3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="ARKitScenes")
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--indices")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--context-frames", type=int, default=1)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--max-gt-pixel-mae", type=float, default=2.0)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--point-cloud-frame-stride", type=int, default=4)
    parser.add_argument("--point-cloud-pixel-stride", type=int, default=4)
    parser.add_argument("--point-cloud-max-points", type=int, default=100000)
    parser.add_argument("--surface-frame-stride", type=int, default=4)
    parser.add_argument("--surface-pixel-stride", type=int, default=1)
    parser.add_argument("--surface-median-kernel", type=int, default=5)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_manifest(root: Path) -> list[dict[str, Any]]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing RGB-D manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    clips = payload.get("clips", [])
    if not clips:
        raise RuntimeError(f"No clips registered in {path}")
    for clip in clips:
        archive = Path(str(clip["archive"]))
        if not archive.is_file():
            archive = root / "test" / archive.name
        if not archive.is_file():
            raise FileNotFoundError(f"Prepared archive is missing: {archive}")
        clip["archive"] = str(archive.resolve())
    return clips


def to_thwc(video: np.ndarray) -> np.ndarray:
    video = np.asarray(video)
    if video.ndim != 4:
        raise ValueError(f"Expected rank-four video, got {video.shape}")
    if video.shape[1] in {1, 3, 4}:
        video = np.moveaxis(video, 1, -1)
    if video.shape[-1] == 4:
        video = video[..., :3]
    if video.shape[-1] != 3:
        raise ValueError(f"Expected RGB video, got {video.shape}")
    return np.ascontiguousarray(video.astype(np.uint8, copy=False))


def file_signature(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    value = f"{resolved}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


class MetricDepthPredictor:
    def __init__(self, model_name: str, device: torch.device, amp: bool):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.eval().to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.amp = amp and device.type == "cuda"

    @torch.inference_mode()
    def __call__(self, video: np.ndarray, batch_size: int) -> np.ndarray:
        predictions = []
        output_size = tuple(video.shape[1:3])
        for start in range(0, len(video), batch_size):
            frames = [np.ascontiguousarray(frame).copy() for frame in video[start : start + batch_size]]
            inputs = self.processor(images=frames, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.amp
                else nullcontext()
            )
            with amp_context:
                output = self.model(**inputs).predicted_depth
            output = F.interpolate(
                output[:, None].float(),
                size=output_size,
                mode="bicubic",
                align_corners=False,
            )[:, 0]
            predictions.append(output.clamp_min(1e-4).cpu().numpy())
        return np.concatenate(predictions, axis=0).astype(np.float32)


def cached_prediction(
    predictor: MetricDepthPredictor,
    video: np.ndarray,
    cache_dir: Path | None,
    source_name: str,
    index: int,
    source_path: Path,
    batch_size: int,
) -> np.ndarray:
    cache_path = None
    if cache_dir is not None:
        cache_path = (
            cache_dir
            / safe_slug(source_name)
            / f"video_{index:05d}_{file_signature(source_path)}.npz"
        )
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as payload:
                prediction = payload["depth"].astype(np.float32)
            if prediction.shape == video.shape[:3]:
                return prediction
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    prediction = predictor(video, batch_size)
    if cache_path is not None:
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, depth=prediction.astype(np.float16))
        temporary.replace(cache_path)
    return prediction


def depth_metrics(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    valid = valid & np.isfinite(prediction) & np.isfinite(target)
    prediction = prediction[valid].astype(np.float64)
    target = target[valid].astype(np.float64)
    if prediction.size == 0:
        return {name: float("nan") for name in DEPTH_METRICS} | {"valid_pixels": 0}
    prediction = np.maximum(prediction, 1e-6)
    target = np.maximum(target, 1e-6)
    difference = prediction - target
    log_difference = np.log(prediction) - np.log(target)
    ratio = np.maximum(prediction / target, target / prediction)
    return {
        "abs_rel": float(np.mean(np.abs(difference) / target)),
        "sq_rel": float(np.mean(difference**2 / target)),
        "rmse_m": float(np.sqrt(np.mean(difference**2))),
        "rmse_log": float(np.sqrt(np.mean(log_difference**2))),
        "log10": float(np.mean(np.abs(np.log10(prediction) - np.log10(target)))),
        "silog": float(
            np.sqrt(max(np.mean(log_difference**2) - np.mean(log_difference) ** 2, 0.0)) * 100.0
        ),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
        "valid_pixels": int(prediction.size),
    }


def median_scale(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    valid = valid & np.isfinite(prediction) & np.isfinite(target) & (prediction > 1e-6)
    if not valid.any():
        return float("nan")
    scale = float(np.median(target[valid] / prediction[valid]))
    return float(np.clip(scale, 1e-2, 1e2))


def summarize(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "count": 0}
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "std": float(finite.std()),
        "count": int(finite.size),
    }


def evaluate_clip(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    context_frames: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    first = min(max(context_frames, 0), len(prediction) - 1)
    prediction = prediction[first:]
    target = target[first:]
    valid = valid[first:]
    clip_scale = median_scale(prediction, target, valid)
    frame_rows: list[dict[str, float]] = []
    frame_scales = []
    for offset, (pred_frame, target_frame, valid_frame) in enumerate(
        zip(prediction, target, valid)
    ):
        direct = depth_metrics(pred_frame, target_frame, valid_frame)
        aligned = depth_metrics(pred_frame * clip_scale, target_frame, valid_frame)
        frame_scale = median_scale(pred_frame, target_frame, valid_frame)
        frame_scales.append(frame_scale)
        row: dict[str, float] = {
            "frame": float(first + offset),
            "clip_scale": clip_scale,
            "frame_scale": frame_scale,
        }
        row.update({f"metric_{key}": value for key, value in direct.items()})
        row.update({f"aligned_{key}": value for key, value in aligned.items()})
        frame_rows.append(row)
    summary: dict[str, float] = {"clip_scale": clip_scale}
    for prefix in ("metric", "aligned"):
        for metric in DEPTH_METRICS:
            summary[f"{prefix}_{metric}"] = float(
                summarize([float(row[f"{prefix}_{metric}"]) for row in frame_rows])["mean"]
            )
    valid_scales = np.asarray(
        [scale for scale in frame_scales if math.isfinite(scale) and scale > 0], dtype=np.float64
    )
    summary["temporal_scale_log_std"] = (
        float(np.std(np.log(valid_scales / clip_scale))) if valid_scales.size else float("nan")
    )
    summary["valid_fraction"] = float(valid.mean())
    return frame_rows, summary


def backproject_point_cloud(
    depth: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    context_frames: int,
    frame_stride: int,
    pixel_stride: int,
    max_points: int,
) -> np.ndarray:
    points = []
    for frame in range(context_frames, len(depth), frame_stride):
        frame_depth = depth[frame]
        height, width = frame_depth.shape
        yy, xx = np.mgrid[0:height:pixel_stride, 0:width:pixel_stride]
        z = frame_depth[::pixel_stride, ::pixel_stride]
        mask = valid[frame, ::pixel_stride, ::pixel_stride] & np.isfinite(z) & (z > 1e-6)
        if not mask.any():
            continue
        intrinsic = intrinsics[frame]
        camera_points = np.stack(
            [
                (xx[mask] - intrinsic[0, 2]) * z[mask] / intrinsic[0, 0],
                (yy[mask] - intrinsic[1, 2]) * z[mask] / intrinsic[1, 1],
                z[mask],
            ],
            axis=-1,
        )
        pose = camera_to_world[frame]
        world_points = (pose[:3, :3] @ camera_points.T).T + pose[:3, 3]
        points.append(world_points)
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    points_array = np.concatenate(points, axis=0).astype(np.float64)
    if len(points_array) > max_points:
        indices = np.linspace(0, len(points_array) - 1, max_points).round().astype(np.int64)
        points_array = points_array[indices]
    return points_array


def point_cloud_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    from scipy.spatial import cKDTree

    if len(predicted) == 0 or len(target) == 0:
        return {
            "point_accuracy_m": float("nan"),
            "point_completeness_m": float("nan"),
            "point_chamfer_l1_m": float("nan"),
            "point_fscore_5cm": float("nan"),
            "point_fscore_10cm": float("nan"),
        }
    target_tree = cKDTree(target)
    predicted_tree = cKDTree(predicted)
    accuracy = target_tree.query(predicted, workers=-1)[0]
    completeness = predicted_tree.query(target, workers=-1)[0]
    metrics = {
        "point_accuracy_m": float(np.mean(accuracy)),
        "point_completeness_m": float(np.mean(completeness)),
        "point_chamfer_l1_m": float(0.5 * (np.mean(accuracy) + np.mean(completeness))),
    }
    for threshold_cm in (5, 10):
        threshold = threshold_cm / 100.0
        precision = float(np.mean(accuracy < threshold))
        recall = float(np.mean(completeness < threshold))
        metrics[f"point_fscore_{threshold_cm}cm"] = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
    return metrics


def depth_point_map(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.mgrid[:height, :width]
    return np.stack(
        [
            (xx - intrinsics[0, 2]) * depth / intrinsics[0, 0],
            (yy - intrinsics[1, 2]) * depth / intrinsics[1, 1],
            depth,
        ],
        axis=-1,
    )


def point_map_normals(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    horizontal = points[1:-1, 2:] - points[1:-1, :-2]
    vertical = points[2:, 1:-1] - points[:-2, 1:-1]
    normals = np.cross(horizontal, vertical)
    magnitude = np.linalg.norm(normals, axis=-1)
    normals /= np.maximum(magnitude[..., None], 1e-12)
    return normals, magnitude > 1e-8


def surface_geometry_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
    context_frames: int,
    frame_stride: int,
    pixel_stride: int,
    median_kernel: int,
) -> dict[str, float]:
    normal_error_sum = 0.0
    point_to_plane_sum = 0.0
    point_epe_sum = 0.0
    threshold_hits = {11.25: 0, 22.5: 0, 30.0: 0}
    count = 0
    erosion_kernel = np.ones((median_kernel, median_kernel), dtype=np.uint8)
    for frame in range(context_frames, len(prediction), frame_stride):
        prediction_frame = prediction[frame].astype(np.float32)
        target_frame = target[frame].astype(np.float32)
        frame_valid = (
            valid[frame]
            & np.isfinite(prediction_frame)
            & np.isfinite(target_frame)
            & (prediction_frame > 1e-6)
            & (target_frame > 1e-6)
        )
        if median_kernel > 1:
            prediction_frame = cv2.medianBlur(prediction_frame, median_kernel)
            target_frame = cv2.medianBlur(target_frame, median_kernel)
            frame_valid = cv2.erode(
                frame_valid.astype(np.uint8), erosion_kernel, borderType=cv2.BORDER_CONSTANT
            ).astype(bool)
        predicted_points = depth_point_map(prediction_frame, intrinsics[frame])
        target_points = depth_point_map(target_frame, intrinsics[frame])
        predicted_normals, predicted_normal_valid = point_map_normals(predicted_points)
        target_normals, target_normal_valid = point_map_normals(target_points)
        normal_valid = (
            frame_valid[1:-1, 1:-1]
            & predicted_normal_valid
            & target_normal_valid
        )
        if pixel_stride > 1:
            normal_valid = normal_valid[::pixel_stride, ::pixel_stride]
            predicted_normals = predicted_normals[::pixel_stride, ::pixel_stride]
            target_normals = target_normals[::pixel_stride, ::pixel_stride]
            predicted_center = predicted_points[1:-1:pixel_stride, 1:-1:pixel_stride]
            target_center = target_points[1:-1:pixel_stride, 1:-1:pixel_stride]
        else:
            predicted_center = predicted_points[1:-1, 1:-1]
            target_center = target_points[1:-1, 1:-1]
        if not normal_valid.any():
            continue
        cosine = np.abs(
            np.sum(predicted_normals[normal_valid] * target_normals[normal_valid], axis=-1)
        ).clip(0.0, 1.0)
        normal_error = np.degrees(np.arccos(cosine))
        point_delta = predicted_center[normal_valid] - target_center[normal_valid]
        point_to_plane = np.abs(
            np.sum(point_delta * target_normals[normal_valid], axis=-1)
        )
        point_epe = np.linalg.norm(point_delta, axis=-1)
        normal_error_sum += float(normal_error.sum())
        point_to_plane_sum += float(point_to_plane.sum())
        point_epe_sum += float(point_epe.sum())
        for threshold in threshold_hits:
            threshold_hits[threshold] += int(np.sum(normal_error < threshold))
        count += int(len(normal_error))
    if count == 0:
        return {
            "surface_normal_mae_deg": float("nan"),
            "surface_normal_11_25": float("nan"),
            "surface_normal_22_5": float("nan"),
            "surface_normal_30": float("nan"),
            "point_to_plane_mae_m": float("nan"),
            "point_epe_corresponding_m": float("nan"),
            "surface_valid_points": 0,
        }
    return {
        "surface_normal_mae_deg": normal_error_sum / count,
        "surface_normal_11_25": threshold_hits[11.25] / count,
        "surface_normal_22_5": threshold_hits[22.5] / count,
        "surface_normal_30": threshold_hits[30.0] / count,
        "point_to_plane_mae_m": point_to_plane_sum / count,
        "point_epe_corresponding_m": point_epe_sum / count,
        "surface_valid_points": count,
    }


def aggregate_clip_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(
        key
        for key in rows[0]
        if key.startswith(("metric_", "aligned_"))
        or key.startswith("point_")
        or key.startswith("surface_")
        or key in {"temporal_scale_log_std", "valid_fraction", "gt_pixel_mae"}
    )
    return {key: summarize([float(row[key]) for row in rows if key in row]) for key in keys}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(value) else f"{value:.4f}"


def main() -> None:
    args = parse_args()
    if (
        args.batch_size < 1
        or args.context_frames < 0
        or args.point_cloud_frame_stride < 1
        or args.point_cloud_pixel_stride < 1
        or args.point_cloud_max_points < 1
        or args.surface_frame_stride < 1
        or args.surface_pixel_stride < 1
        or args.surface_median_kernel < 1
        or args.surface_median_kernel % 2 == 0
    ):
        raise ValueError("batch-size must be positive and context-frames non-negative")
    args.prepared_root = args.prepared_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clips = load_manifest(args.prepared_root)
    requested = parse_indices(args.indices)
    device = choose_device(args.device)
    predictor = MetricDepthPredictor(args.model, device, args.amp)
    model_slug = safe_slug(args.model)
    cache_dir = args.cache_dir.expanduser().resolve() / model_slug if args.cache_dir else None

    run_files: dict[str, dict[int, Path]] = {}
    for name, directory in args.run:
        files = iter_saved_pairs(directory, None, requested, args.max_files, "npz")
        indexed = {video_index(path): path for path in files if video_index(path) is not None}
        if not indexed:
            raise FileNotFoundError(f"No indexed lossless videos under {directory}")
        run_files[name] = {int(index): path for index, path in indexed.items()}

    selected_indices = sorted(set.intersection(*(set(files) for files in run_files.values())))
    if requested is not None:
        selected_indices = [index for index in selected_indices if index in requested]
    if args.max_files is not None:
        selected_indices = selected_indices[: args.max_files]
    if not selected_indices:
        raise RuntimeError("No common generated indices across requested runs")
    if selected_indices[-1] >= len(clips):
        raise IndexError(
            f"Generated index {selected_indices[-1]} exceeds {len(clips)} prepared clips"
        )

    frame_rows: list[dict[str, Any]] = []
    clip_rows: list[dict[str, Any]] = []
    gt_predictions: dict[int, np.ndarray] = {}
    for index in selected_indices:
        clip = clips[index]
        archive = Path(clip["archive"])
        with np.load(archive, allow_pickle=False) as payload:
            source_video = to_thwc(payload["video"])
            target_depth = payload["depth_m"].astype(np.float32)
            valid = payload["depth_valid"].astype(bool)
            intrinsics = payload["intrinsics"].astype(np.float64)
            camera_to_world = payload["camera_to_world"].astype(np.float64)
        valid &= np.isfinite(target_depth)
        valid &= (target_depth >= args.min_depth) & (target_depth <= args.max_depth)
        gt_predictions[index] = cached_prediction(
            predictor,
            source_video,
            cache_dir,
            "GT-input",
            index,
            archive,
            args.batch_size,
        )
        gt_frames, gt_summary = evaluate_clip(
            gt_predictions[index], target_depth, valid, args.context_frames
        )
        target_cloud = backproject_point_cloud(
            target_depth,
            valid,
            intrinsics,
            camera_to_world,
            args.context_frames,
            args.point_cloud_frame_stride,
            args.point_cloud_pixel_stride,
            args.point_cloud_max_points,
        )
        gt_cloud = backproject_point_cloud(
            gt_predictions[index] * gt_summary["clip_scale"],
            valid,
            intrinsics,
            camera_to_world,
            args.context_frames,
            args.point_cloud_frame_stride,
            args.point_cloud_pixel_stride,
            args.point_cloud_max_points,
        )
        gt_summary.update(point_cloud_metrics(gt_cloud, target_cloud))
        gt_summary.update(
            surface_geometry_metrics(
                gt_predictions[index] * gt_summary["clip_scale"],
                target_depth,
                valid,
                intrinsics,
                args.context_frames,
                args.surface_frame_stride,
                args.surface_pixel_stride,
                args.surface_median_kernel,
            )
        )
        gt_summary.update(
            {"source": "GT-input", "index": index, "clip_id": clip["clip_id"], "gt_pixel_mae": 0.0}
        )
        clip_rows.append(gt_summary)
        for row in gt_frames:
            row.update({"source": "GT-input", "index": index, "clip_id": clip["clip_id"]})
            frame_rows.append(row)

        for name, files in run_files.items():
            generated, saved_gt = load_saved_pair(files[index])
            frames = min(len(generated), len(saved_gt), len(source_video), len(target_depth))
            generated = generated[:frames]
            saved_gt = saved_gt[:frames]
            reference = source_video[:frames]
            pixel_mae = float(
                np.mean(np.abs(saved_gt.astype(np.float32) - reference.astype(np.float32)))
            )
            if pixel_mae > args.max_gt_pixel_mae:
                raise RuntimeError(
                    f"{name} index {index} does not match prepared GT: pixel MAE={pixel_mae:.3f}"
                )
            prediction = cached_prediction(
                predictor,
                generated,
                cache_dir,
                name,
                index,
                files[index],
                args.batch_size,
            )
            rows, summary = evaluate_clip(
                prediction[:frames], target_depth[:frames], valid[:frames], args.context_frames
            )
            predicted_cloud = backproject_point_cloud(
                prediction[:frames] * summary["clip_scale"],
                valid[:frames],
                intrinsics[:frames],
                camera_to_world[:frames],
                args.context_frames,
                args.point_cloud_frame_stride,
                args.point_cloud_pixel_stride,
                args.point_cloud_max_points,
            )
            method_target_cloud = (
                target_cloud
                if frames == len(target_depth)
                else backproject_point_cloud(
                    target_depth[:frames],
                    valid[:frames],
                    intrinsics[:frames],
                    camera_to_world[:frames],
                    args.context_frames,
                    args.point_cloud_frame_stride,
                    args.point_cloud_pixel_stride,
                    args.point_cloud_max_points,
                )
            )
            summary.update(point_cloud_metrics(predicted_cloud, method_target_cloud))
            summary.update(
                surface_geometry_metrics(
                    prediction[:frames] * summary["clip_scale"],
                    target_depth[:frames],
                    valid[:frames],
                    intrinsics[:frames],
                    args.context_frames,
                    args.surface_frame_stride,
                    args.surface_pixel_stride,
                    args.surface_median_kernel,
                )
            )
            summary.update(
                {"source": name, "index": index, "clip_id": clip["clip_id"], "gt_pixel_mae": pixel_mae}
            )
            clip_rows.append(summary)
            for row in rows:
                row.update({"source": name, "index": index, "clip_id": clip["clip_id"]})
                frame_rows.append(row)
        print(f"[{index + 1}/{len(clips)}] {clip['clip_id']}")

    source_names = ["GT-input", *run_files]
    aggregate = {
        name: aggregate_clip_rows([row for row in clip_rows if row["source"] == name])
        for name in source_names
    }
    report = {
        "protocol": {
            "dataset": args.dataset_name,
            "model": args.model,
            "model_role": "independent metric indoor depth evaluator",
            "alignment": "one median scale per clip",
            "point_cloud": {
                "frame_stride": args.point_cloud_frame_stride,
                "pixel_stride": args.point_cloud_pixel_stride,
                "max_points": args.point_cloud_max_points,
                "poses": "ground-truth camera-to-world",
            },
            "surface_geometry": {
                "frame_stride": args.surface_frame_stride,
                "pixel_stride": args.surface_pixel_stride,
                "median_kernel": args.surface_median_kernel,
                "normal_sign": "orientation invariant",
            },
            "context_frames_excluded": args.context_frames,
            "indices": selected_indices,
            "generated_output_used_for_selection": False,
        },
        "aggregate": aggregate,
        "per_clip": clip_rows,
    }
    (args.output_dir / "metric_depth_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "metric_depth_per_clip.csv", clip_rows)
    write_csv(args.output_dir / "metric_depth_per_frame.csv", frame_rows)

    lines = [
        f"# {args.dataset_name} Independent Metric Depth",
        "",
        "| Source | AbsRel metric | AbsRel aligned | delta1 aligned | Normal MAE (deg) | Normal <30deg | Point-plane (m) | Chamfer L1 (m) | F@5cm | F@10cm | scale drift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in source_names:
        values = aggregate[name]
        lines.append(
            f"| {name} | {fmt(values['metric_abs_rel']['mean'])} | "
            f"{fmt(values['aligned_abs_rel']['mean'])} | {fmt(values['aligned_delta1']['mean'])} | "
            f"{fmt(values['surface_normal_mae_deg']['mean'])} | "
            f"{fmt(values['surface_normal_30']['mean'])} | "
            f"{fmt(values['point_to_plane_mae_m']['mean'])} | "
            f"{fmt(values['point_chamfer_l1_m']['mean'])} | "
            f"{fmt(values['point_fscore_5cm']['mean'])} | "
            f"{fmt(values['point_fscore_10cm']['mean'])} | "
            f"{fmt(values['temporal_scale_log_std']['mean'])} |"
        )
    lines += [
        "",
        "`GT-input` measures the frozen evaluator's floor on the real RGB frames. Generated-video errors are computed against synchronized sensor depth; scale alignment uses one scalar for the entire clip and therefore does not hide framewise scale drift.",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved independent depth results to {args.output_dir}")


if __name__ == "__main__":
    main()
