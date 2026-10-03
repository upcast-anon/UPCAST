"""Shared camera-motion selection and RGB-D archive helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np


RE10K_64_REFERENCE = {
    "sample_count": 854,
    "path_length": {"p05": 0.158503, "p50": 0.895677, "p95": 2.112896},
    "displacement": {"p05": 0.155476, "p50": 0.863019, "p95": 2.095595},
    "max_translation_step": {"p50": 0.018364, "p95": 0.043959},
    "rotation_sum_deg": {"p05": 0.544140, "p50": 10.486805, "p95": 34.238981},
    "max_rotation_step_deg": {"p50": 0.272938, "p95": 0.827136},
}
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


class CameraRecord(Protocol):
    camera_to_world: np.ndarray


@dataclass(frozen=True)
class MotionStats:
    path_length: float
    displacement: float
    median_translation_step: float
    max_translation_step: float
    rotation_sum_deg: float
    max_rotation_step_deg: float
    score: float
    in_reference_envelope: bool


def opencv_camera_to_world(matrix: np.ndarray, convention: str) -> np.ndarray:
    if convention == "nerfstudio_opengl":
        return matrix @ OPENGL_TO_OPENCV
    return matrix.copy()


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def motion_stats(records: Sequence[CameraRecord], convention: str) -> MotionStats:
    poses = np.stack(
        [opencv_camera_to_world(record.camera_to_world, convention) for record in records]
    )
    centers = poses[:, :3, 3]
    translation_steps = np.linalg.norm(np.diff(centers, axis=0), axis=-1)
    rotation_steps = np.asarray(
        [rotation_angle_deg(poses[i - 1, :3, :3].T @ poses[i, :3, :3])
         for i in range(1, len(poses))], dtype=np.float64,
    )
    path_length = float(translation_steps.sum())
    displacement = float(np.linalg.norm(centers[-1] - centers[0]))
    rotation_sum = float(rotation_steps.sum())
    in_envelope = (
        RE10K_64_REFERENCE["path_length"]["p05"] <= path_length
        <= RE10K_64_REFERENCE["path_length"]["p95"]
        and RE10K_64_REFERENCE["rotation_sum_deg"]["p05"] <= rotation_sum
        <= RE10K_64_REFERENCE["rotation_sum_deg"]["p95"]
        and float(translation_steps.max(initial=0.0))
        <= 2.0 * RE10K_64_REFERENCE["max_translation_step"]["p95"]
        and float(rotation_steps.max(initial=0.0))
        <= 3.0 * RE10K_64_REFERENCE["max_rotation_step_deg"]["p95"]
    )
    path_term = abs(math.log(
        (path_length + 1e-6) / RE10K_64_REFERENCE["path_length"]["p50"]
    ))
    rotation_term = abs(math.log(
        (rotation_sum + 0.25) / (RE10K_64_REFERENCE["rotation_sum_deg"]["p50"] + 0.25)
    ))
    smoothness = float(np.std(translation_steps) / max(np.mean(translation_steps), 1e-6))
    return MotionStats(
        path_length=path_length, displacement=displacement,
        median_translation_step=float(np.median(translation_steps)),
        max_translation_step=float(translation_steps.max(initial=0.0)),
        rotation_sum_deg=rotation_sum,
        max_rotation_step_deg=float(rotation_steps.max(initial=0.0)),
        score=path_term + 0.35 * rotation_term + 0.1 * smoothness,
        in_reference_envelope=in_envelope,
    )


def square_resize(
    array: np.ndarray, intrinsic: np.ndarray, resolution: int, nearest: bool
) -> tuple[np.ndarray, np.ndarray]:
    height, width = array.shape[:2]
    side = min(height, width)
    left = (width - side) // 2
    top = (height - side) // 2
    cropped = array[top:top + side, left:left + side]
    adjusted = intrinsic.copy()
    adjusted[0, 2] -= left
    adjusted[1, 2] -= top
    scale = resolution / side
    adjusted[0] *= scale
    adjusted[1] *= scale
    adjusted[2] = np.array([0.0, 0.0, 1.0])
    resized = cv2.resize(
        cropped, (resolution, resolution),
        interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA,
    )
    return resized, adjusted


def write_clip(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
