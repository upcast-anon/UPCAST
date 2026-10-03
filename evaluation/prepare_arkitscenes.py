#!/usr/bin/env python3
"""Prepare synchronized ARKitScenes RGB-D clips for DFoT evaluation.

The raw ARKitScenes low-resolution RGB, LiDAR depth, and confidence streams are
sampled at 60 FPS, while camera poses are provided at roughly 10 Hz. This
preprocessor intersects the sensor timestamps and interpolates the official
camera-to-world poses using linear translation and SO(3) SLERP. Clip selection
depends only on source validity and a fixed RE10K camera-motion reference.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.rgbd_utils import (
    MotionStats,
    RE10K_64_REFERENCE,
    motion_stats,
    square_resize,
    write_clip,
)


TIMESTAMP = re.compile(r"_(-?\d+(?:\.\d+)?)$")
DEFAULT_VIDEO_LIST = Path(__file__).with_name("arkitscenes_video_ids.txt")


@dataclass(frozen=True)
class Trajectory:
    timestamps: np.ndarray
    camera_to_world: np.ndarray


@dataclass(frozen=True)
class FrameRecord:
    frame_index: int
    timestamp: float
    image_path: Path
    depth_path: Path
    confidence_path: Path
    intrinsic_path: Path
    camera_to_world: np.ndarray
    intrinsic: np.ndarray
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("./data/arkitscenes/raw"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./data/arkitscenes-prepared"),
    )
    parser.add_argument("--split", default="Validation")
    parser.add_argument("--video-list", type=Path, default=DEFAULT_VIDEO_LIST)
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--max-videos", type=int, default=12)
    parser.add_argument("--clip-length", type=int, default=64)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--selection-step",
        type=int,
        default=6,
        help="Evaluate candidate starts every N synchronized source frames.",
    )
    parser.add_argument("--source-fps", type=float, default=60.0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--confidence-threshold", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--min-depth-valid-fraction", type=float, default=0.55)
    parser.add_argument("--intrinsic-tolerance-ms", type=int, default=2)
    parser.add_argument("--max-frame-gap-factor", type=float, default=2.1)
    parser.add_argument("--max-candidate-trials", type=int, default=30)
    parser.add_argument("--allow-out-of-range-motion", action="store_true")
    parser.add_argument("--allow-repeated-visits", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def timestamp_from_path(path: Path) -> float:
    match = TIMESTAMP.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot parse ARKitScenes timestamp from {path.name}")
    return float(match.group(1))


def timestamp_key(path: Path) -> int:
    return int(round(timestamp_from_path(path) * 1000.0))


def read_video_ids(args: argparse.Namespace) -> list[str]:
    if args.video_ids:
        values = args.video_ids
    elif args.video_list is not None and args.video_list.is_file():
        values = args.video_list.read_text(encoding="utf-8").splitlines()
    else:
        values = [path.name for path in sorted((args.raw_root / args.split).iterdir())]
    video_ids = []
    for value in values:
        value = value.split("#", 1)[0].strip()
        if value and value not in video_ids:
            video_ids.append(value)
    if not video_ids:
        raise ValueError("No ARKitScenes video IDs were provided")
    return video_ids


def read_visits(metadata_path: Path) -> dict[str, str]:
    if not metadata_path.is_file():
        return {}
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        return {
            str(row["video_id"]): str(row.get("visit_id", ""))
            for row in csv.DictReader(handle)
        }


def load_trajectory(path: Path) -> Trajectory:
    if not path.is_file():
        raise FileNotFoundError(f"Missing ARKitScenes trajectory: {path}")
    rows = np.loadtxt(path, dtype=np.float64)
    rows = np.atleast_2d(rows)
    if rows.shape[1] != 7 or len(rows) < 2 or not np.isfinite(rows).all():
        raise ValueError(f"Malformed ARKitScenes trajectory: {path}")
    order = np.argsort(rows[:, 0], kind="stable")
    rows = rows[order]
    _, unique = np.unique(rows[:, 0], return_index=True)
    rows = rows[np.sort(unique)]
    if len(rows) < 2 or np.any(np.diff(rows[:, 0]) <= 0):
        raise ValueError(f"Trajectory timestamps are not strictly increasing: {path}")

    world_to_camera_rotation = Rotation.from_rotvec(rows[:, 1:4]).as_matrix()
    camera_to_world_rotation = np.swapaxes(world_to_camera_rotation, 1, 2)
    camera_centers = -np.einsum(
        "nij,nj->ni", camera_to_world_rotation, rows[:, 4:7]
    )
    camera_to_world = np.repeat(np.eye(4, dtype=np.float64)[None], len(rows), axis=0)
    camera_to_world[:, :3, :3] = camera_to_world_rotation
    camera_to_world[:, :3, 3] = camera_centers
    return Trajectory(rows[:, 0], camera_to_world)


def interpolate_poses(trajectory: Trajectory, timestamps: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1:
        raise ValueError("Pose query timestamps must be one-dimensional")
    if timestamps.size == 0:
        return np.empty((0, 4, 4), dtype=np.float64)
    if timestamps[0] < trajectory.timestamps[0] or timestamps[-1] > trajectory.timestamps[-1]:
        raise ValueError("Pose query lies outside the official trajectory range")

    rotations = Slerp(
        trajectory.timestamps,
        Rotation.from_matrix(trajectory.camera_to_world[:, :3, :3]),
    )(timestamps).as_matrix()
    centers = np.column_stack(
        [
            np.interp(
                timestamps,
                trajectory.timestamps,
                trajectory.camera_to_world[:, axis, 3],
            )
            for axis in range(3)
        ]
    )
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(timestamps), axis=0)
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = centers
    return poses


def file_map(directory: Path, suffix: str) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing ARKitScenes asset directory: {directory}")
    result = {}
    for path in directory.glob(f"*.{suffix}"):
        result[timestamp_key(path)] = path.resolve()
    if not result:
        raise FileNotFoundError(f"No *.{suffix} files in {directory}")
    return result


def nearest_path(paths: dict[int, Path], key: int, tolerance: int) -> Path | None:
    keys = sorted(paths)
    position = bisect.bisect_left(keys, key)
    candidates = [index for index in (position - 1, position) if 0 <= index < len(keys)]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda index: abs(keys[index] - key))
    if abs(keys[nearest] - key) > tolerance:
        return None
    return paths[keys[nearest]]


def load_intrinsic(path: Path) -> tuple[np.ndarray, int, int]:
    values = np.asarray(np.loadtxt(path), dtype=np.float64).reshape(-1)
    if values.size != 6 or not np.isfinite(values).all():
        raise ValueError(f"Malformed ARKitScenes intrinsic file: {path}")
    width, height, fx, fy, cx, cy = values
    intrinsic = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return intrinsic, int(round(width)), int(round(height))


def load_video_records(video_root: Path, intrinsic_tolerance_ms: int) -> list[FrameRecord]:
    video_id = video_root.name
    rgb = file_map(video_root / "lowres_wide", "png")
    depth = file_map(video_root / "lowres_depth", "png")
    confidence = file_map(video_root / "confidence", "png")
    intrinsics = file_map(video_root / "lowres_wide_intrinsics", "pincam")
    trajectory = load_trajectory(video_root / "lowres_wide.traj")

    keys = sorted(set(rgb) & set(depth) & set(confidence))
    keys = [
        key
        for key in keys
        if trajectory.timestamps[0] <= key / 1000.0 <= trajectory.timestamps[-1]
        and nearest_path(intrinsics, key, intrinsic_tolerance_ms) is not None
    ]
    if not keys:
        raise ValueError(f"No synchronized RGB-D-pose timestamps for video {video_id}")
    timestamps = np.asarray(keys, dtype=np.float64) / 1000.0
    poses = interpolate_poses(trajectory, timestamps)

    records = []
    for index, (key, timestamp, pose) in enumerate(zip(keys, timestamps, poses)):
        intrinsic_path = nearest_path(intrinsics, key, intrinsic_tolerance_ms)
        assert intrinsic_path is not None
        intrinsic, width, height = load_intrinsic(intrinsic_path)
        records.append(
            FrameRecord(
                frame_index=index,
                timestamp=float(timestamp),
                image_path=rgb[key],
                depth_path=depth[key],
                confidence_path=confidence[key],
                intrinsic_path=intrinsic_path,
                camera_to_world=pose,
                intrinsic=intrinsic,
                width=width,
                height=height,
            )
        )
    return records


def candidate_windows(
    records: list[FrameRecord], args: argparse.Namespace
) -> list[tuple[MotionStats, list[FrameRecord]]]:
    span = (args.clip_length - 1) * args.frame_stride + 1
    if len(records) < span:
        return []
    expected_gap = args.frame_stride / args.source_fps
    max_gap = expected_gap * args.max_frame_gap_factor
    last_start = len(records) - span
    starts = list(range(0, last_start + 1, args.selection_step))
    if starts[-1] != last_start:
        starts.append(last_start)
    candidates = []
    for start in starts:
        selected = records[start : start + span : args.frame_stride]
        timestamps = np.asarray([record.timestamp for record in selected])
        gaps = np.diff(timestamps)
        if gaps.size and (np.any(gaps <= 0.0) or float(gaps.max()) > max_gap):
            continue
        stats = motion_stats(selected, "opencv")
        if stats.in_reference_envelope or args.allow_out_of_range_motion:
            candidates.append((stats, selected))
    return sorted(
        candidates,
        key=lambda item: (item[0].score, item[1][0].timestamp),
    )


def scaled_intrinsic(record: FrameRecord, width: int, height: int) -> np.ndarray:
    intrinsic = record.intrinsic.copy()
    intrinsic[0] *= width / record.width
    intrinsic[1] *= height / record.height
    intrinsic[2] = np.array([0.0, 0.0, 1.0])
    return intrinsic


def process_frame(
    record: FrameRecord, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
    depth_mm = cv2.imread(str(record.depth_path), cv2.IMREAD_UNCHANGED)
    confidence = cv2.imread(str(record.confidence_path), cv2.IMREAD_UNCHANGED)
    if image_bgr is None or depth_mm is None or confidence is None:
        raise ValueError(f"Failed to read synchronized frame at {record.timestamp:.3f}")
    if depth_mm.ndim != 2 or confidence.ndim != 2 or image_bgr.shape[:2] != depth_mm.shape:
        raise ValueError(f"Unexpected RGB-D shape at {record.timestamp:.3f}")
    if confidence.shape != depth_mm.shape:
        raise ValueError(f"Depth/confidence shape mismatch at {record.timestamp:.3f}")

    height, width = depth_mm.shape
    intrinsic = scaled_intrinsic(record, width, height)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb, output_intrinsic = square_resize(
        image_rgb, intrinsic, args.resolution, nearest=False
    )
    depth_mm, _ = square_resize(depth_mm, intrinsic, args.resolution, nearest=True)
    confidence, _ = square_resize(confidence, intrinsic, args.resolution, nearest=True)
    depth_m = depth_mm.astype(np.float32) / 1000.0
    valid = (
        np.isfinite(depth_m)
        & (depth_m > args.min_depth)
        & (depth_m < args.max_depth)
        & (confidence >= args.confidence_threshold)
    )

    world_to_camera = np.linalg.inv(record.camera_to_world)
    normalized_intrinsic = np.array(
        [
            output_intrinsic[0, 0],
            output_intrinsic[1, 1],
            output_intrinsic[0, 2],
            output_intrinsic[1, 2],
        ],
        dtype=np.float64,
    ) / args.resolution
    camera = np.concatenate([normalized_intrinsic, world_to_camera[:3].reshape(-1)])
    return (
        image_rgb,
        depth_m,
        valid,
        confidence,
        camera,
        output_intrinsic,
    )


def process_window(
    video_id: str,
    visit_id: str,
    records: list[FrameRecord],
    stats: MotionStats,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    images, depths, valid_masks, confidence_maps = [], [], [], []
    cameras, intrinsics = [], []
    for record in records:
        image, depth, valid, confidence, camera, intrinsic = process_frame(record, args)
        images.append(np.moveaxis(image, -1, 0))
        depths.append(depth)
        valid_masks.append(valid)
        confidence_maps.append(confidence)
        cameras.append(camera)
        intrinsics.append(intrinsic)

    depth_valid = np.stack(valid_masks)
    if float(depth_valid.mean()) < args.min_depth_valid_fraction:
        return None
    timestamps = np.asarray([record.timestamp for record in records], dtype=np.float64)
    measured_fps = float((len(timestamps) - 1) / (timestamps[-1] - timestamps[0]))
    clip_id = f"{video_id}_t{int(round(timestamps[0] * 1000)):012d}_s{args.frame_stride}"
    confidence_values, confidence_counts = np.unique(
        np.stack(confidence_maps), return_counts=True
    )
    confidence_histogram = {
        str(int(value)): int(count)
        for value, count in zip(confidence_values, confidence_counts)
    }
    arrays = {
        "video": np.stack(images).astype(np.uint8),
        "camera": np.asarray(cameras, dtype=np.float32),
        "depth_m": np.stack(depths).astype(np.float32),
        "depth_valid": depth_valid.astype(np.uint8),
        "image_valid": np.ones_like(depth_valid, dtype=np.uint8),
        "confidence": np.stack(confidence_maps).astype(np.uint8),
        "intrinsics": np.asarray(intrinsics, dtype=np.float32),
        "camera_to_world": np.asarray(
            [record.camera_to_world for record in records], dtype=np.float32
        ),
        "frame_indices": np.asarray(
            [record.frame_index for record in records], dtype=np.int32
        ),
        "frame_timestamps": timestamps,
        "fps": np.asarray(measured_fps, dtype=np.float32),
        "scene_id": np.asarray(video_id),
        "visit_id": np.asarray(visit_id),
        "clip_id": np.asarray(clip_id),
    }
    metadata = {
        "clip_id": clip_id,
        "scene_id": video_id,
        "video_id": video_id,
        "visit_id": visit_id,
        "frame_indices": arrays["frame_indices"].tolist(),
        "frame_timestamps": timestamps.tolist(),
        "rgb_source": str(records[0].image_path.parent),
        "depth_source": str(records[0].depth_path.parent),
        "confidence_source": str(records[0].confidence_path.parent),
        "motion": asdict(stats),
        "measured_fps": measured_fps,
        "window_duration_seconds": float(timestamps[-1] - timestamps[0]),
        "depth_valid_fraction": float(depth_valid.mean()),
        "confidence_threshold": args.confidence_threshold,
        "confidence_histogram": confidence_histogram,
    }
    return arrays, metadata


def main() -> None:
    args = parse_args()
    if (
        args.max_videos < 1
        or args.clip_length < 2
        or args.frame_stride < 1
        or args.selection_step < 1
        or args.source_fps <= 0
    ):
        raise ValueError("Video count, clip length, strides, and source FPS must be positive")
    video_ids = read_video_ids(args)
    visits = read_visits(args.raw_root / "metadata.csv")
    split_root = args.raw_root / args.split
    output_split = args.output_root / "test"
    output_split.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "version": 1,
        "dataset": "ARKitScenes v1 raw RGB-D",
        "source_split": args.split,
        "raw_root": str(args.raw_root.resolve()),
        "pose_convention": "official camera-to-world; linear translation + SO(3) SLERP",
        "clip_length": args.clip_length,
        "frame_stride": args.frame_stride,
        "source_fps": args.source_fps,
        "resolution": args.resolution,
        "confidence_threshold": args.confidence_threshold,
        "selection_uses_generated_output": False,
        "re10k_64_reference": RE10K_64_REFERENCE,
        "clips": [],
        "skipped_videos": [],
    }
    used_visits: set[str] = set()

    for video_id in video_ids:
        if len(manifest["clips"]) >= args.max_videos:
            break
        visit_id = visits.get(video_id, "")
        if (
            visit_id
            and visit_id.upper() != "NA"
            and visit_id in used_visits
            and not args.allow_repeated_visits
        ):
            manifest["skipped_videos"].append(
                {"video_id": video_id, "reason": f"visit {visit_id} already selected"}
            )
            continue
        try:
            records = load_video_records(
                split_root / video_id, args.intrinsic_tolerance_ms
            )
            windows = candidate_windows(records, args)
            if not windows:
                raise ValueError("no synchronized window in the fixed motion envelope")
            prepared = None
            for stats, selected in windows[: args.max_candidate_trials]:
                prepared = process_window(video_id, visit_id, selected, stats, args)
                if prepared is not None:
                    break
            if prepared is None:
                raise ValueError("candidate windows failed the depth-confidence validity gate")
            arrays, metadata = prepared
            output_path = output_split / f"{metadata['clip_id']}.npz"
            reused = output_path.exists() and not args.overwrite
            if not args.dry_run and not reused:
                write_clip(output_path, arrays)
            metadata["archive"] = str(output_path.resolve())
            metadata["reused"] = reused
            manifest["clips"].append(metadata)
            if visit_id and visit_id.upper() != "NA":
                used_visits.add(visit_id)
            print(
                f"[selected {len(manifest['clips']):02d}/{args.max_videos}] "
                f"{metadata['clip_id']} path={metadata['motion']['path_length']:.3f} "
                f"rot={metadata['motion']['rotation_sum_deg']:.2f} "
                f"depth={metadata['depth_valid_fraction']:.3f}"
            )
        except (FileNotFoundError, ValueError, OSError) as error:
            manifest["skipped_videos"].append(
                {"video_id": video_id, "reason": str(error)}
            )
            print(f"[skip] {video_id}: {error}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if len(manifest["clips"]) < args.max_videos:
        raise RuntimeError(
            f"Prepared only {len(manifest['clips'])}/{args.max_videos} clips; see {manifest_path}"
        )
    if not args.dry_run:
        metadata_dir = args.output_root / "metadata"
        if metadata_dir.exists():
            for path in metadata_dir.glob("*.pt"):
                path.unlink()
    print(f"Prepared {len(manifest['clips'])} ARKitScenes clips at {args.output_root}")


if __name__ == "__main__":
    main()
