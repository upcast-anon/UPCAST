#!/usr/bin/env python3
"""Build paper-facing tables and paired uncertainty estimates for main runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DIRECTION = {
    "fvd": "down",
    "fid": "down",
    "lpips": "down",
    "mse": "down",
    "ssim": "up",
    "psnr": "up",
}
DISPLAY = {
    "fvd": "FVD",
    "fid": "FID",
    "lpips": "LPIPS",
    "mse": "MSE",
    "ssim": "SSIM",
    "psnr": "PSNR",
}

WORLD_DISPLAY = {
    "jedi_official_mmd2": "JEDi (↓)",
}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open() as handle:
        return json.load(handle)


def fmt(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    value = float(value)
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.3f}"
    return f"{value:.5f}"


def normalized_metric_key(key: str) -> str:
    if key == "psnr_from_mse":
        return "psnr"
    return key


def paired_bootstrap(
    candidate: dict[int, float],
    baseline: dict[int, float],
    direction: str,
    rng: np.random.Generator,
    samples: int,
) -> dict[str, Any]:
    indices = sorted(set(candidate) & set(baseline))
    if not indices:
        return {"n": 0}
    candidate_values = np.asarray([candidate[index] for index in indices], dtype=np.float64)
    baseline_values = np.asarray([baseline[index] for index in indices], dtype=np.float64)
    raw_delta = candidate_values - baseline_values
    favorable = -raw_delta if direction == "down" else raw_delta
    draws = rng.integers(0, len(indices), size=(samples, len(indices)))
    bootstrap = favorable[draws].mean(axis=1)
    return {
        "n": len(indices),
        "favorable_mean": float(favorable.mean()),
        "ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "probability_improved": float((bootstrap > 0).mean()),
        "win_fraction": float((favorable > 0).mean()),
    }


def read_pixel_report(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[int, dict[str, float]]]]:
    payload = load_json(path)
    if payload is None:
        return {}, {}
    aggregate: dict[str, dict[str, Any]] = {}
    per_video: dict[str, dict[int, dict[str, float]]] = {}
    for run in payload.get("runs", []):
        name = str(run["name"])
        aggregate[name] = run.get("aggregate", {})
        rows: dict[int, dict[str, float]] = {}
        for row in run.get("per_video", []):
            if row.get("index") is None:
                continue
            rows[int(row["index"])] = {
                normalized_metric_key(key): float(value)
                for key, value in row.items()
                if normalized_metric_key(key) in {"lpips", "mse", "ssim", "psnr"}
                and isinstance(value, (int, float))
            }
        per_video[name] = rows
    return aggregate, per_video


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:" for _ in headers[1:]]) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def get_nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def finite_mean(values: list[Any]) -> float | None:
    numeric = np.asarray(
        [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))],
        dtype=np.float64,
    )
    return float(numeric.mean()) if numeric.size else None


def named_aggregates(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not payload:
        return {}
    if isinstance(payload.get("runs"), list):
        return {
            str(run["name"]): run.get("aggregate", {})
            for run in payload["runs"]
            if isinstance(run, dict) and run.get("name") is not None
        }
    if isinstance(payload.get("results"), dict):
        return payload["results"]
    return {}


def aggregate_droid(payload: dict[str, Any] | None) -> dict[str, Any]:
    aggregate = payload.get("aggregate_metrics", {}) if payload else {}
    relative = aggregate.get("reprojection_error_relative_mean")
    return {
        "reprojection_error": aggregate.get("reprojection_error_gen_mean"),
        "relative_error": relative,
        "relative_distance_to_one": abs(float(relative) - 1.0)
        if isinstance(relative, (int, float)) and math.isfinite(float(relative))
        else None,
        "num_videos": get_nested(payload or {}, "summary", "successful"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--methods", default="DFoT,REPA,VideoREPA,GF,Ours")
    parser.add_argument("--protocols", default="long100:64,stress12:128,256")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()

    root = Path(args.root)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    protocol_horizons: dict[str, list[int]] = {}
    current_protocol = None
    for token in args.protocols.split(","):
        if ":" in token:
            current_protocol, horizon = token.split(":", 1)
            protocol_horizons[current_protocol] = [int(horizon)]
        elif current_protocol is not None:
            protocol_horizons[current_protocol].append(int(token))
        else:
            raise ValueError(f"Malformed --protocols token: {token}")

    rng = np.random.default_rng(args.seed)
    report: dict[str, Any] = {"methods": methods, "protocols": {}, "paired": {}}
    lines = ["# Main Experiment Results", ""]
    for protocol, horizons in protocol_horizons.items():
        report["protocols"][protocol] = {}
        lines.extend([f"## {protocol}", ""])
        for horizon in horizons:
            pixel_path = root / "metrics" / protocol / f"horizon_{horizon}" / "pixel" / "gif_video_metrics_report.json"
            aggregate, per_video = read_pixel_report(pixel_path)
            if not aggregate:
                continue
            report["protocols"][protocol][str(horizon)] = aggregate
            lines.extend([f"### {horizon} Frames", ""])
            rows = []
            for method in methods:
                values = aggregate.get(method, {})
                rows.append([method] + [fmt(values.get(key)) for key in DIRECTION])
            lines.extend(
                markdown_table(
                    ["Method"] + [f"{DISPLAY[key]} ({'↓' if DIRECTION[key] == 'down' else '↑'})" for key in DIRECTION],
                    rows,
                )
            )
            lines.append("")

            pair_key = f"{protocol}/horizon_{horizon}"
            report["paired"][pair_key] = {}
            if "Ours" in per_video:
                for baseline in [name for name in methods if name != "Ours" and name in per_video]:
                    report["paired"][pair_key][baseline] = {}
                    for metric in ["lpips", "mse", "ssim", "psnr"]:
                        ours_values = {
                            index: row[metric]
                            for index, row in per_video["Ours"].items()
                            if metric in row
                        }
                        baseline_values = {
                            index: row[metric]
                            for index, row in per_video[baseline].items()
                            if metric in row
                        }
                        report["paired"][pair_key][baseline][metric] = paired_bootstrap(
                            ours_values,
                            baseline_values,
                            DIRECTION[metric],
                            rng,
                            args.bootstrap_samples,
                        )

    auxiliary: dict[str, Any] = {}
    lines.extend(["## World Representation Metrics", ""])
    for protocol in protocol_horizons:
        jedi = load_json(root / "metrics" / protocol / "jedi" / "official_jedi_report.json")
        jedi_runs = named_aggregates(jedi)
        if not jedi_runs:
            continue
        auxiliary[protocol] = {"official_jedi": jedi}
        rows = []
        for method in methods:
            merged = jedi_runs.get(method, {})
            rows.append([method] + [fmt(merged.get(key)) for key in WORLD_DISPLAY])
        lines.extend([f"### {protocol}", ""])
        lines.extend(markdown_table(["Method", *WORLD_DISPLAY.values()], rows))
        lines.append("")
    report["auxiliary"] = auxiliary

    vjepa_fscore = load_json(
        root
        / "metrics"
        / "long100"
        / "jedi"
        / "vjepa_feature_fscore"
        / "vjepa_feature_fscore_report.json"
    )
    report["vjepa_feature_fscore"] = vjepa_fscore
    if vjepa_fscore:
        feature_rows = []
        for method in methods:
            values = get_nested(vjepa_fscore, "results", method, "primary")
            if not isinstance(values, dict):
                continue
            interval = get_nested(values, "uncertainty", "feature_f1_ci95")
            rendered_interval = (
                f"{fmt(interval[0])}–{fmt(interval[1])}"
                if isinstance(interval, list) and len(interval) == 2
                else "--"
            )
            feature_rows.append(
                [
                    method,
                    fmt(values.get("feature_precision")),
                    fmt(values.get("feature_recall")),
                    fmt(values.get("feature_f1")),
                    rendered_interval,
                ]
            )
        if feature_rows:
            primary_neighbor = get_nested(
                vjepa_fscore, "protocol", "primary_neighbor"
            )
            lines.extend(["## V-JEPA Feature PR-F1", ""])
            lines.extend(
                markdown_table(
                    ["Method", "Precision (↑)", "Recall (↑)", "PR-F1 (↑)", "F1 95% CI"],
                    feature_rows,
                )
            )
            lines.extend(
                [
                    f"Exploratory L2-normalized V-JEPA feature-manifold metric with k={primary_neighbor}; this is not an official JEDi, pixel, or geometry F-score.",
                    "",
                ]
            )

    geometry = {}
    geometry_rows = []
    for method in methods:
        payload = load_json(
            root / "metrics/long100/geometry/droid" / method / "eval/video_evaluation_results.json"
        )
        values = aggregate_droid(payload)
        geometry[method] = values
        geometry_rows.append([method, fmt(values.get("reprojection_error"))])
    report["geometry"] = geometry
    lines.extend(["## Reprojection Error", ""])
    lines.extend(markdown_table(["Method", "RPE (lower is better)"], geometry_rows))
    lines.append("")

    lines.extend(["## Paired Evidence", ""])
    paired_rows = []
    for protocol, baselines in report["paired"].items():
        for baseline, metrics in baselines.items():
            for metric, result in metrics.items():
                if not result.get("n"):
                    continue
                lo, hi = result["ci95"]
                paired_rows.append(
                    [
                        protocol,
                        f"Ours vs {baseline}",
                        DISPLAY[metric],
                        str(result["n"]),
                        f"{result['favorable_mean']:.6g} [{lo:.6g}, {hi:.6g}]",
                        f"{100.0 * result['probability_improved']:.1f}%",
                    ]
                )
    lines.extend(markdown_table(["Protocol", "Contrast", "Metric", "N", "Favorable Δ [95% CI]", "P(improve)"], paired_rows))
    lines.append("")

    output_dir = root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "main_results.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    (output_dir / "main_results.md").write_text("\n".join(lines) + "\n")
    print(f"Saved main experiment summary to {output_dir}")


if __name__ == "__main__":
    main()
