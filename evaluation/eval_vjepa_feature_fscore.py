#!/usr/bin/env python3
"""Compute an exploratory PR-F1 from saved V-JEPA video features.

This is a feature-manifold precision/recall diagnostic derived from the
improved precision/recall protocol for generative models. It is not an
official JEDi metric and must not be presented as a pixel or geometry F-score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be NAME=FEATURE_DIR")
    name, directory = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("--run name cannot be empty")
    return name.strip(), Path(directory).expanduser()


def parse_neighbors(value: str) -> list[int]:
    neighbors = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not neighbors or neighbors[0] < 1:
        raise argparse.ArgumentTypeError("--neighbors must contain positive integers")
    return neighbors


def load_features(path: Path) -> np.ndarray:
    features = np.load(path, allow_pickle=False)
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError(f"Expected a [samples, dimensions] feature matrix in {path}")
    features = np.asarray(features, dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite feature values in {path}")
    return features


def normalize_features(features: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return features
    if mode != "l2":
        raise ValueError(f"Unsupported normalization: {mode}")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Cannot L2-normalize zero feature vectors")
    return features / norms


def neighborhood_radii(features: np.ndarray, neighbors: int) -> np.ndarray:
    if neighbors >= len(features):
        raise ValueError(
            f"neighbors={neighbors} requires at least {neighbors + 1} feature samples"
        )
    distances = cdist(features, features, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    return np.partition(distances, neighbors - 1, axis=1)[:, neighbors - 1]


def harmonic_mean(precision: np.ndarray | float, recall: np.ndarray | float) -> Any:
    precision_array = np.asarray(precision, dtype=np.float64)
    recall_array = np.asarray(recall, dtype=np.float64)
    denominator = precision_array + recall_array
    score = np.divide(
        2.0 * precision_array * recall_array,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(score) if score.ndim == 0 else score


def feature_precision_recall(
    predicted: np.ndarray,
    reference: np.ndarray,
    neighbors: int,
    normalization: str = "l2",
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    if predicted.shape[1] != reference.shape[1]:
        raise ValueError(
            f"Feature dimensions differ: {predicted.shape[1]} vs {reference.shape[1]}"
        )
    predicted = normalize_features(np.asarray(predicted, dtype=np.float64), normalization)
    reference = normalize_features(np.asarray(reference, dtype=np.float64), normalization)
    predicted_radii = neighborhood_radii(predicted, neighbors)
    reference_radii = neighborhood_radii(reference, neighbors)
    cross_distances = cdist(predicted, reference, metric="euclidean")

    precision_hits = np.any(
        cross_distances <= reference_radii[None, :], axis=1
    )
    recall_hits = np.any(cross_distances <= predicted_radii[:, None], axis=0)
    precision = float(precision_hits.mean())
    recall = float(recall_hits.mean())
    return (
        {
            "feature_precision": precision,
            "feature_recall": recall,
            "feature_f1": harmonic_mean(precision, recall),
        },
        precision_hits,
        recall_hits,
    )


def bootstrap_fixed_manifold(
    precision_hits: np.ndarray,
    recall_hits: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples < 1:
        return {}
    rng = np.random.default_rng(seed)
    precision_draws = precision_hits[
        rng.integers(0, len(precision_hits), size=(samples, len(precision_hits)))
    ].mean(axis=1)
    recall_draws = recall_hits[
        rng.integers(0, len(recall_hits), size=(samples, len(recall_hits)))
    ].mean(axis=1)
    f1_draws = harmonic_mean(precision_draws, recall_draws)

    def interval(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.quantile(values, [0.025, 0.975])]

    return {
        "method": "fixed-full-manifold nonparametric bootstrap",
        "samples": samples,
        "feature_precision_ci95": interval(precision_draws),
        "feature_recall_ci95": interval(recall_draws),
        "feature_f1_ci95": interval(f1_draws),
    }


def file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def fmt(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{100.0 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neighbors", type=parse_neighbors, default=parse_neighbors("1,3,5"))
    parser.add_argument("--primary-neighbor", type=int, default=3)
    parser.add_argument("--normalization", choices=("l2", "none"), default="l2")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--source-report", type=Path)
    args = parser.parse_args()

    if args.primary_neighbor not in args.neighbors:
        raise ValueError("primary-neighbor must be included in neighbors")
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap-samples must be non-negative")

    source_report = None
    if args.source_report:
        source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    results: dict[str, Any] = {}
    primary_hits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, directory in args.run:
        directory = directory.expanduser().resolve()
        predicted_path = directory / "pred_features.npy"
        reference_path = directory / "gt_features.npy"
        if not predicted_path.is_file() or not reference_path.is_file():
            raise FileNotFoundError(
                f"Expected pred_features.npy and gt_features.npy under {directory}"
            )
        predicted = load_features(predicted_path)
        reference = load_features(reference_path)
        sensitivity = {}
        for neighbors in args.neighbors:
            metrics, precision_hits, recall_hits = feature_precision_recall(
                predicted, reference, neighbors, args.normalization
            )
            sensitivity[str(neighbors)] = metrics
            if neighbors == args.primary_neighbor:
                primary_hits[name] = (precision_hits, recall_hits)
        precision_hits, recall_hits = primary_hits[name]
        primary = dict(sensitivity[str(args.primary_neighbor)])
        primary["uncertainty"] = bootstrap_fixed_manifold(
            precision_hits,
            recall_hits,
            args.bootstrap_samples,
            args.seed,
        )
        results[name] = {
            "num_predicted_features": int(len(predicted)),
            "num_reference_features": int(len(reference)),
            "feature_dim": int(predicted.shape[1]),
            "primary": primary,
            "neighbor_sensitivity": sensitivity,
            "files": {
                "predicted": file_metadata(predicted_path),
                "reference": file_metadata(reference_path),
            },
        }

    report = {
        "metric": "V-JEPA feature-manifold precision/recall F1",
        "status": "exploratory derived metric; not official JEDi or a pixel/geometry F-score",
        "protocol": {
            "normalization": args.normalization,
            "distance": "Euclidean after configured normalization",
            "neighbors": args.neighbors,
            "primary_neighbor": args.primary_neighbor,
            "precision": "fraction of generated features inside the reference kNN manifold",
            "recall": "fraction of reference features inside the generated kNN manifold",
            "feature_f1": "harmonic mean of feature precision and feature recall",
        },
        "source_report": str(args.source_report.resolve()) if args.source_report else None,
        "source_metric": source_report.get("metric") if source_report else None,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "vjepa_feature_fscore_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with (args.output_dir / "vjepa_feature_fscore_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "neighbors",
                "feature_precision",
                "feature_recall",
                "feature_f1",
                "num_features",
            ],
        )
        writer.writeheader()
        for name, result in results.items():
            for neighbors, metrics in result["neighbor_sensitivity"].items():
                writer.writerow(
                    {
                        "method": name,
                        "neighbors": neighbors,
                        **metrics,
                        "num_features": result["num_predicted_features"],
                    }
                )

    lines = [
        "# V-JEPA Feature PR-F1",
        "",
        "This is an exploratory feature-manifold diagnostic, not an official JEDi, pixel, or geometry F-score.",
        "",
        "| Method | k | Precision | Recall | PR-F1 | Features |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        for neighbors, metrics in result["neighbor_sensitivity"].items():
            marker = " (primary)" if int(neighbors) == args.primary_neighbor else ""
            lines.append(
                f"| {name} | {neighbors}{marker} | {fmt(metrics['feature_precision'])} | "
                f"{fmt(metrics['feature_recall'])} | {fmt(metrics['feature_f1'])} | "
                f"{result['num_predicted_features']} |"
            )
    (args.output_dir / "vjepa_feature_fscore_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({name: value["primary"] for name, value in results.items()}, indent=2))
    print(f"Saved V-JEPA feature PR-F1 report to {args.output_dir}")


if __name__ == "__main__":
    main()
