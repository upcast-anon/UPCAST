#!/usr/bin/env python3
"""Localize generated ARKitScenes frames against synchronized RGB-D observations.

Each generated frame is matched to its real RGB counterpart. Reference keypoints
with valid sensor depth are lifted into the world frame, then PnP-RANSAC recovers
the camera pose represented by the generated image. Unlike monocular SLAM, this
protocol has metric scale and does not accumulate trajectory drift.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

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


@dataclass(frozen=True)
class ImageFeatures:
    points: np.ndarray
    descriptors: np.ndarray | None


def parse_thresholds(value: str) -> list[tuple[float, float]]:
    thresholds = []
    for item in value.split(","):
        translation, rotation = item.strip().split(":", 1)
        pair = (float(translation), float(rotation))
        if pair[0] <= 0 or pair[1] <= 0:
            raise argparse.ArgumentTypeError("Pose thresholds must be positive")
        thresholds.append(pair)
    if not thresholds:
        raise argparse.ArgumentTypeError("At least one pose threshold is required")
    return thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="ARKitScenes v1 raw RGB-D")
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--indices")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--context-frames", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-gt-pixel-mae", type=float, default=2.0)
    parser.add_argument("--detector", choices=("sift", "orb"), default="sift")
    parser.add_argument("--max-features", type=int, default=4096)
    parser.add_argument("--sift-contrast-threshold", type=float, default=0.02)
    parser.add_argument("--match-ratio", type=float, default=0.8)
    parser.add_argument("--mutual-matches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-matches", type=int, default=8)
    parser.add_argument("--min-inliers", type=int, default=8)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.2)
    parser.add_argument("--min-image-coverage", type=float, default=0.005)
    parser.add_argument("--min-cheirality", type=float, default=0.9)
    parser.add_argument("--ransac-reprojection-error", type=float, default=3.0)
    parser.add_argument("--ransac-confidence", type=float, default=0.999)
    parser.add_argument("--ransac-iterations", type=int, default=2000)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument(
        "--pose-thresholds",
        type=parse_thresholds,
        default=parse_thresholds("0.05:5,0.10:10,0.20:20"),
        help="Comma-separated translation-meters:rotation-degrees recall thresholds.",
    )
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def threshold_key(translation_m: float, rotation_deg: float) -> str:
    centimeters = int(round(100.0 * translation_m))
    degrees = int(round(rotation_deg))
    return f"pose_recall_{centimeters}cm_{degrees}deg"


def load_manifest(root: Path) -> list[dict[str, Any]]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing ARKitScenes manifest: {path}")
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
    value = np.asarray(video)
    if value.ndim != 4:
        raise ValueError(f"Expected rank-four video, got {value.shape}")
    if value.shape[-1] not in {1, 3, 4} and value.shape[1] in {1, 3, 4}:
        value = np.moveaxis(value, 1, -1)
    if value.shape[-1] == 1:
        value = np.repeat(value, 3, axis=-1)
    elif value.shape[-1] == 4:
        value = value[..., :3]
    if value.shape[-1] != 3:
        raise ValueError(f"Expected RGB video, got {value.shape}")
    return np.ascontiguousarray(value.astype(np.uint8, copy=False))


class FeatureMatcher:
    def __init__(
        self,
        detector: str,
        max_features: int,
        ratio: float,
        mutual: bool,
        sift_contrast_threshold: float,
    ) -> None:
        self.ratio = ratio
        self.mutual = mutual
        if detector == "sift":
            self.detector = cv2.SIFT_create(
                nfeatures=max_features,
                contrastThreshold=sift_contrast_threshold,
            )
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:
            self.detector = cv2.ORB_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def extract(self, image: np.ndarray) -> ImageFeatures:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        points = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float64)
        if not len(points):
            points = np.empty((0, 2), dtype=np.float64)
        return ImageFeatures(points=points, descriptors=descriptors)

    def _ratio_matches(self, first: np.ndarray, second: np.ndarray) -> dict[int, int]:
        accepted: dict[int, int] = {}
        for pair in self.matcher.knnMatch(first, second, k=2):
            if len(pair) == 2 and pair[0].distance < self.ratio * pair[1].distance:
                accepted[int(pair[0].queryIdx)] = int(pair[0].trainIdx)
        return accepted

    def match(
        self,
        query: ImageFeatures,
        reference: ImageFeatures,
    ) -> tuple[np.ndarray, np.ndarray]:
        if query.descriptors is None or reference.descriptors is None:
            empty = np.empty((0, 2), dtype=np.float64)
            return empty, empty.copy()
        if len(query.descriptors) < 2 or len(reference.descriptors) < 2:
            empty = np.empty((0, 2), dtype=np.float64)
            return empty, empty.copy()
        forward = self._ratio_matches(query.descriptors, reference.descriptors)
        if self.mutual:
            reverse = self._ratio_matches(reference.descriptors, query.descriptors)
            pairs = [(query_index, reference_index) for query_index, reference_index in forward.items()
                     if reverse.get(reference_index) == query_index]
        else:
            pairs = list(forward.items())
        if not pairs:
            empty = np.empty((0, 2), dtype=np.float64)
            return empty, empty.copy()
        query_points = np.asarray([query.points[first] for first, _ in pairs], dtype=np.float64)
        reference_points = np.asarray(
            [reference.points[second] for _, second in pairs], dtype=np.float64
        )
        return query_points, reference_points


def world_points_from_reference(
    reference_points: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    pixels = np.rint(reference_points).astype(np.int64)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    kept = np.flatnonzero(inside)
    pixels = pixels[inside]
    if not len(pixels):
        return np.empty((0, 3), dtype=np.float64), kept
    z = depth[pixels[:, 1], pixels[:, 0]].astype(np.float64)
    depth_valid = (
        valid[pixels[:, 1], pixels[:, 0]]
        & np.isfinite(z)
        & (z >= min_depth)
        & (z <= max_depth)
    )
    kept = kept[depth_valid]
    z = z[depth_valid]
    points = reference_points[kept]
    camera_points = np.stack(
        [
            (points[:, 0] - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (points[:, 1] - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        ],
        axis=-1,
    )
    world_points = (
        camera_to_world[:3, :3] @ camera_points.T
    ).T + camera_to_world[:3, 3]
    return world_points.astype(np.float64), kept


def rotation_error_deg(estimated: np.ndarray, target: np.ndarray) -> float:
    relative = estimated.T @ target
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    skew = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    )
    sine = 0.5 * np.linalg.norm(skew)
    return float(np.degrees(np.arctan2(sine, cosine)))


def image_coverage(points: np.ndarray, width: int, height: int) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull) / max(float(width * height), 1.0))


def solve_sensor_pose(
    world_points: np.ndarray,
    image_points: np.ndarray,
    intrinsics: np.ndarray,
    target_camera_to_world: np.ndarray,
    image_shape: tuple[int, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pnp_success": False,
        "reason": "too_few_depth_matches",
        "depth_match_count": int(len(world_points)),
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "image_coverage": 0.0,
        "cheirality_fraction": 0.0,
        "pnp_reprojection_rmse_px": float("nan"),
        "translation_error_m": float("nan"),
        "rotation_error_deg": float("nan"),
    }
    if len(world_points) < args.min_matches:
        return result
    try:
        solved, rotation_vector, translation, inliers = cv2.solvePnPRansac(
            world_points,
            image_points,
            intrinsics,
            None,
            iterationsCount=args.ransac_iterations,
            reprojectionError=args.ransac_reprojection_error,
            confidence=args.ransac_confidence,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        result["reason"] = "pnp_exception"
        return result
    if not solved or inliers is None:
        result["reason"] = "pnp_failed"
        return result
    inlier_indices = inliers.reshape(-1).astype(np.int64)
    result["inlier_count"] = int(len(inlier_indices))
    result["inlier_ratio"] = float(len(inlier_indices) / len(world_points))
    if len(inlier_indices) < args.min_inliers:
        result["reason"] = "too_few_inliers"
        return result
    if result["inlier_ratio"] < args.min_inlier_ratio:
        result["reason"] = "low_inlier_ratio"
        return result

    inlier_world = world_points[inlier_indices]
    inlier_image = image_points[inlier_indices]
    try:
        rotation_vector, translation = cv2.solvePnPRefineLM(
            inlier_world,
            inlier_image,
            intrinsics,
            None,
            rotation_vector,
            translation,
        )
    except cv2.error:
        pass
    world_to_camera_rotation, _ = cv2.Rodrigues(rotation_vector)
    translation = translation.reshape(3)
    camera_depth = (world_to_camera_rotation @ inlier_world.T).T[:, 2] + translation[2]
    result["cheirality_fraction"] = float(np.mean(camera_depth > 0.0))
    if result["cheirality_fraction"] < args.min_cheirality:
        result["reason"] = "low_cheirality"
        return result

    height, width = image_shape
    result["image_coverage"] = image_coverage(inlier_image, width, height)
    if result["image_coverage"] < args.min_image_coverage:
        result["reason"] = "low_image_coverage"
        return result

    projections, _ = cv2.projectPoints(
        inlier_world,
        rotation_vector,
        translation,
        intrinsics,
        None,
    )
    reprojection = projections[:, 0] - inlier_image
    result["pnp_reprojection_rmse_px"] = float(
        np.sqrt(np.mean(np.sum(reprojection**2, axis=-1)))
    )

    estimated_camera_to_world = np.eye(4, dtype=np.float64)
    estimated_camera_to_world[:3, :3] = world_to_camera_rotation.T
    estimated_camera_to_world[:3, 3] = -world_to_camera_rotation.T @ translation
    result["translation_error_m"] = float(
        np.linalg.norm(
            estimated_camera_to_world[:3, 3] - target_camera_to_world[:3, 3]
        )
    )
    result["rotation_error_deg"] = rotation_error_deg(
        estimated_camera_to_world[:3, :3],
        target_camera_to_world[:3, :3],
    )
    result["pnp_success"] = True
    result["reason"] = "ok"
    return result


def evaluate_frame(
    query: ImageFeatures,
    reference: ImageFeatures,
    matcher: FeatureMatcher,
    depth: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    query_points, reference_points = matcher.match(query, reference)
    world_points, kept = world_points_from_reference(
        reference_points,
        depth,
        valid,
        intrinsics,
        camera_to_world,
        args.min_depth,
        args.max_depth,
    )
    result = solve_sensor_pose(
        world_points,
        query_points[kept],
        intrinsics,
        camera_to_world,
        depth.shape,
        args,
    )
    result.update(
        {
            "query_keypoints": int(len(query.points)),
            "reference_keypoints": int(len(reference.points)),
            "match_count": int(len(query_points)),
        }
    )
    for translation_m, rotation_deg in args.pose_thresholds:
        result[threshold_key(translation_m, rotation_deg)] = int(
            result["pnp_success"]
            and result["translation_error_m"] <= translation_m
            and result["rotation_error_deg"] <= rotation_deg
        )
    return result


def finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def summarize(values: np.ndarray) -> dict[str, float | int]:
    if not len(values):
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "count": 0}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "count": int(len(values)),
    }


def summarize_clip(rows: list[dict[str, Any]], threshold_keys: list[str]) -> dict[str, Any]:
    successful = [row for row in rows if row["pnp_success"]]
    translation = finite_values(successful, "translation_error_m")
    rotation = finite_values(successful, "rotation_error_deg")
    summary: dict[str, Any] = {
        "evaluated_frames": len(rows),
        "successful_frames": len(successful),
        "pnp_success_rate": float(len(successful) / max(len(rows), 1)),
        "ate_rmse_m": float(np.sqrt(np.mean(translation**2))) if len(translation) else float("nan"),
        "translation_error_m_mean": float(translation.mean()) if len(translation) else float("nan"),
        "translation_error_m_median": float(np.median(translation)) if len(translation) else float("nan"),
        "rotation_error_deg_mean": float(rotation.mean()) if len(rotation) else float("nan"),
        "rotation_error_deg_median": float(np.median(rotation)) if len(rotation) else float("nan"),
    }
    for key in threshold_keys:
        summary[key] = float(np.mean([float(row[key]) for row in rows])) if rows else float("nan")
    for key in (
        "match_count",
        "inlier_count",
        "inlier_ratio",
        "image_coverage",
        "pnp_reprojection_rmse_px",
    ):
        values = finite_values(successful if key != "match_count" else rows, key)
        summary[f"{key}_mean"] = float(values.mean()) if len(values) else float("nan")
    return summary


def aggregate_clips(rows: list[dict[str, Any]], threshold_keys: list[str]) -> dict[str, Any]:
    metrics = [
        "pnp_success_rate",
        "ate_rmse_m",
        "translation_error_m_mean",
        "translation_error_m_median",
        "rotation_error_deg_mean",
        "rotation_error_deg_median",
        "match_count_mean",
        "inlier_count_mean",
        "inlier_ratio_mean",
        "image_coverage_mean",
        "pnp_reprojection_rmse_px_mean",
        *threshold_keys,
    ]
    return {metric: summarize(finite_values(rows, metric)) for metric in metrics}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.4f}"


def main() -> None:
    args = parse_args()
    if (
        args.context_frames < 0
        or args.frame_stride < 1
        or args.max_features < 8
        or args.min_matches < 4
        or args.min_inliers < 4
        or not 0.0 < args.match_ratio < 1.0
        or not 0.0 <= args.min_inlier_ratio <= 1.0
        or not 0.0 <= args.min_image_coverage <= 1.0
        or not 0.0 <= args.min_cheirality <= 1.0
    ):
        raise ValueError("Invalid feature matching or PnP arguments")
    args.prepared_root = args.prepared_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clips = load_manifest(args.prepared_root)
    requested = parse_indices(args.indices)
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
        raise IndexError(f"Generated index exceeds the {len(clips)} prepared clips")

    cv2.setRNGSeed(args.seed)
    matcher = FeatureMatcher(
        args.detector,
        args.max_features,
        args.match_ratio,
        args.mutual_matches,
        args.sift_contrast_threshold,
    )
    threshold_keys = [threshold_key(*pair) for pair in args.pose_thresholds]
    frame_rows: list[dict[str, Any]] = []
    clip_rows: list[dict[str, Any]] = []

    for index in selected_indices:
        clip = clips[index]
        with np.load(clip["archive"], allow_pickle=False) as payload:
            reference_video = to_thwc(payload["video"])
            depth = payload["depth_m"].astype(np.float64)
            valid = payload["depth_valid"].astype(bool)
            intrinsics = payload["intrinsics"].astype(np.float64)
            camera_to_world = payload["camera_to_world"].astype(np.float64)
        length = min(len(reference_video), len(depth), len(intrinsics), len(camera_to_world))
        if args.max_frames is not None and args.max_frames > 0:
            length = min(length, args.max_frames)
        frames = list(range(args.context_frames, length, args.frame_stride))
        reference_features = {frame: matcher.extract(reference_video[frame]) for frame in frames}

        sources: list[tuple[str, np.ndarray, float]] = [("GT-input", reference_video, 0.0)]
        for source, files in run_files.items():
            prediction, saved_target = load_saved_pair(files[index], max_frames=length)
            available = min(length, len(prediction), len(saved_target))
            if available <= args.context_frames:
                raise RuntimeError(f"Too few frames for {source} index {index}")
            gt_pixel_mae = float(
                np.mean(
                    np.abs(
                        saved_target[:available].astype(np.float32)
                        - reference_video[:available].astype(np.float32)
                    )
                )
            )
            if gt_pixel_mae > args.max_gt_pixel_mae:
                raise RuntimeError(
                    f"Saved target mismatch for {source} index {index}: MAE={gt_pixel_mae:.4f}"
                )
            sources.append((source, prediction[:available], gt_pixel_mae))

        for source, video, gt_pixel_mae in sources:
            rows = []
            source_frames = [frame for frame in frames if frame < len(video)]
            for frame in source_frames:
                query = reference_features[frame] if source == "GT-input" else matcher.extract(video[frame])
                result = evaluate_frame(
                    query,
                    reference_features[frame],
                    matcher,
                    depth[frame],
                    valid[frame],
                    intrinsics[frame],
                    camera_to_world[frame],
                    args,
                )
                row = {
                    "source": source,
                    "index": index,
                    "clip_id": clip.get("clip_id", str(index)),
                    "frame": frame,
                    "gt_pixel_mae": gt_pixel_mae,
                    **result,
                }
                rows.append(row)
                frame_rows.append(row)
            clip_summary = summarize_clip(rows, threshold_keys)
            clip_rows.append(
                {
                    "source": source,
                    "index": index,
                    "clip_id": clip.get("clip_id", str(index)),
                    "gt_pixel_mae": gt_pixel_mae,
                    **clip_summary,
                }
            )
            print(
                f"[{source}] {index:05d} success={clip_summary['pnp_success_rate']:.3f} "
                f"recall10={clip_summary.get('pose_recall_10cm_10deg', float('nan')):.3f}",
                flush=True,
            )

    source_order = ["GT-input", *run_files]
    aggregate = {
        source: aggregate_clips(
            [row for row in clip_rows if row["source"] == source], threshold_keys
        )
        for source in source_order
    }
    protocol = {
        "dataset": args.dataset_name,
        "indices": selected_indices,
        "context_frames": args.context_frames,
        "frame_stride": args.frame_stride,
        "detector": args.detector,
        "max_features": args.max_features,
        "sift_contrast_threshold": args.sift_contrast_threshold,
        "match_ratio": args.match_ratio,
        "mutual_matches": args.mutual_matches,
        "min_matches": args.min_matches,
        "min_inliers": args.min_inliers,
        "min_inlier_ratio": args.min_inlier_ratio,
        "min_image_coverage": args.min_image_coverage,
        "min_cheirality": args.min_cheirality,
        "ransac_reprojection_error": args.ransac_reprojection_error,
        "ransac_confidence": args.ransac_confidence,
        "ransac_iterations": args.ransac_iterations,
        "pose_thresholds": args.pose_thresholds,
        "reference": "synchronized real RGB keypoints lifted with sensor depth and GT pose",
        "failure_policy": "PnP failures count as zero for every pose-recall threshold",
    }
    report = {
        "protocol": protocol,
        "aggregate": aggregate,
        "per_clip": clip_rows,
    }
    (args.output_dir / "sensor_pose_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "sensor_pose_per_frame.csv", frame_rows)
    write_csv(args.output_dir / "sensor_pose_per_clip.csv", clip_rows)

    lines = [
        f"# {args.dataset_name} Sensor-Anchored PnP Pose",
        "",
        "Generated frames are localized against synchronized sensor RGB-D with mutual feature matching and PnP-RANSAC. Failures remain in the denominator for pose recall.",
        "",
        "| Source | PnP success | Recall 5cm/5deg | Recall 10cm/10deg | Recall 20cm/20deg | ATE RMSE (m) | Median t (m) | Median R (deg) | Reproj. (px) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source in source_order:
        values = aggregate[source]
        lines.append(
            f"| {source} | {fmt(values['pnp_success_rate']['mean'])} | "
            f"{fmt(values.get('pose_recall_5cm_5deg', {}).get('mean'))} | "
            f"{fmt(values.get('pose_recall_10cm_10deg', {}).get('mean'))} | "
            f"{fmt(values.get('pose_recall_20cm_20deg', {}).get('mean'))} | "
            f"{fmt(values['ate_rmse_m']['mean'])} | "
            f"{fmt(values['translation_error_m_median']['mean'])} | "
            f"{fmt(values['rotation_error_deg_median']['mean'])} | "
            f"{fmt(values['pnp_reprojection_rmse_px_mean']['mean'])} |"
        )
    lines += [
        "",
        "`GT-input` is a same-frame RGB-D/PnP numerical sanity check, not a learned-evaluator floor.",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved sensor-anchored pose report to {args.output_dir}")


if __name__ == "__main__":
    main()
