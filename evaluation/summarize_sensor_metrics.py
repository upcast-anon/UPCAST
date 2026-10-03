#!/usr/bin/env python3
"""Summarize the sensor-referenced ARKitScenes table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DEPTH_METRICS = {
    "depth_abs_rel": "aligned_abs_rel",
    "depth_delta1": "aligned_delta1",
    "scale_drift": "temporal_scale_log_std",
    "normal_mae_deg": "surface_normal_mae_deg",
    "chamfer_l1_m": "point_chamfer_l1_m",
    "fscore_5cm": "point_fscore_5cm",
    "fscore_10cm": "point_fscore_10cm",
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(report, method, metric):
    value = report["aggregate"][method][metric]
    return float(value["mean"] if isinstance(value, dict) else value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--methods", default="GF,Ours")
    parser.add_argument("--dataset-name", default="ARKitScenes")
    parser.add_argument("--output-prefix", default="arkitscenes_results")
    args = parser.parse_args()
    rgb = load_json(args.root / "metrics/rgb/gif_video_metrics_report.json")
    depth = load_json(args.root / "geometry/metric_depth/metric_depth_report.json")
    pose = load_json(args.root / "geometry/sensor_pose/sensor_pose_report.json")
    rgb_runs = {run["name"]: run["aggregate"] for run in rgb["runs"]}
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    if not methods:
        raise ValueError("At least one method is required.")
    rows = []
    for method in methods:
        row = {"method": method, "psnr": float(rgb_runs[method]["psnr"])}
        row.update({key: mean(depth, method, metric) for key, metric in DEPTH_METRICS.items()})
        row["pose_recall_10cm_10deg"] = mean(pose, method, "pose_recall_10cm_10deg")
        rows.append(row)
    output = args.root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.output_prefix}.json").write_text(
        json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8"
    )
    with (output / f"{args.output_prefix}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    headers = [
        "Method", "PSNR", "AbsRel", "delta1", "Scale drift", "Normal MAE (deg)",
        "Chamfer L1 (m)", "F@5cm", "F@10cm", "Pose R@10",
    ]
    lines = [
        f"# {args.dataset_name} Results", "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for row in rows:
        values = [row["method"]] + [
            f"{value:.4f}" if math.isfinite(value) else "n/a"
            for key, value in row.items() if key != "method"
        ]
        lines.append("| " + " | ".join(values) + " |")
    (output / f"{args.output_prefix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
