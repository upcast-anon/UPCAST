#!/usr/bin/env python3
"""Summarize matched UPCAST training stages."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METRICS = {
    "fvd": "down",
    "fid": "down",
    "lpips": "down",
    "mse": "down",
    "ssim": "up",
    "psnr": "up",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def read_pixel(path: Path) -> tuple[dict[str, dict[str, float]], dict[str, dict[int, dict[str, float]]]]:
    payload = load_json(path)
    aggregate: dict[str, dict[str, float]] = {}
    rows: dict[str, dict[int, dict[str, float]]] = {}
    for run in payload.get("runs", []):
        name = str(run["name"])
        aggregate[name] = {
            key: float(value)
            for key, value in run.get("aggregate", {}).items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        rows[name] = {
            int(row["index"]): {
                key: float(value)
                for key, value in row.items()
                if key in {"lpips", "mse", "ssim", "psnr", "psnr_from_mse"}
                and isinstance(value, (int, float))
            }
            for row in run.get("per_video", [])
            if row.get("index") is not None
        }
        for row in rows[name].values():
            if "psnr" not in row and "psnr_from_mse" in row:
                row["psnr"] = row["psnr_from_mse"]
    return aggregate, rows


def paired_bootstrap(
    before: dict[int, dict[str, float]],
    after: dict[int, dict[str, float]],
    metric: str,
    direction: str,
    generator: np.random.Generator,
    draws: int,
) -> dict[str, Any]:
    indices = sorted(
        index
        for index in set(before) & set(after)
        if metric in before[index] and metric in after[index]
    )
    if not indices:
        return {"n": 0}
    raw = np.asarray(
        [after[index][metric] - before[index][metric] for index in indices],
        dtype=np.float64,
    )
    favorable = -raw if direction == "down" else raw
    sample_ids = generator.integers(0, len(indices), size=(draws, len(indices)))
    distribution = favorable[sample_ids].mean(axis=1)
    return {
        "n": len(indices),
        "favorable_mean": float(favorable.mean()),
        "ci95": [
            float(np.quantile(distribution, 0.025)),
            float(np.quantile(distribution, 0.975)),
        ],
        "probability_improved": float((distribution > 0).mean()),
        "win_fraction": float((favorable > 0).mean()),
    }


def fmt(value: float | None) -> str:
    if value is None:
        return "--"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.3f}"
    return f"{value:.5f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--methods",
        default="DFoT,SharedSpace,SharedSpace+Distill,Full",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    protocols = ("long25",)
    generator = np.random.default_rng(args.seed)
    report: dict[str, Any] = {
        "methods": methods,
        "protocols": {},
        "contrasts": {},
        "interpretation_contract": (
            "Sequential checkpoint-lineage ablation with matched evaluation; "
            "not an equal-compute from-scratch architecture ablation."
        ),
    }
    lines = [
        "# Training-Stage Analysis",
        "",
        "All rows use the same dataset indices, sampling seed, context protocol, and raw-video metric implementation.",
        "This is a sequential checkpoint-lineage ablation, not an equal-compute from-scratch architecture ablation.",
        "",
    ]
    for protocol in protocols:
        aggregate, per_video = read_pixel(
            root / "metrics" / protocol / "pixel" / "gif_video_metrics_report.json"
        )
        jedi_path = root / "metrics" / protocol / "jedi" / "official_jedi_report.json"
        jedi = load_json(jedi_path).get("results", {}) if jedi_path.is_file() else {}
        report["protocols"][protocol] = {"pixel": aggregate, "jedi": jedi}
        lines.extend(
            [
                f"## {protocol}",
                "",
                "| Method | FVD ↓ | FID ↓ | LPIPS ↓ | MSE ↓ | SSIM ↑ | PSNR ↑ | JEDi ↓ |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in methods:
            values = aggregate.get(method, {})
            jedi_value = jedi.get(method, {}).get("jedi_official_mmd2")
            lines.append(
                "| {name} | {fvd} | {fid} | {lpips} | {mse} | {ssim} | {psnr} | {jedi} |".format(
                    name=method,
                    jedi=fmt(jedi_value),
                    **{metric: fmt(values.get(metric)) for metric in METRICS},
                )
            )
        lines.extend(["", "### Incremental paired effects", ""])
        report["contrasts"][protocol] = {}
        for before, after in zip(methods[:-1], methods[1:]):
            contrast = f"{after} - {before}"
            report["contrasts"][protocol][contrast] = {}
            lines.append(f"**{before} → {after}.**")
            for metric in ("lpips", "mse", "ssim", "psnr"):
                result = paired_bootstrap(
                    per_video.get(before, {}),
                    per_video.get(after, {}),
                    metric,
                    METRICS[metric],
                    generator,
                    args.bootstrap_samples,
                )
                report["contrasts"][protocol][contrast][metric] = result
                if result.get("n", 0):
                    lines.append(
                        f"- {metric.upper()}: favorable Δ={result['favorable_mean']:.6g}, "
                        f"95% CI=[{result['ci95'][0]:.6g}, {result['ci95'][1]:.6g}], "
                        f"P(improve)={100.0 * result['probability_improved']:.1f}% "
                        f"(N={result['n']})."
                    )
            lines.append("")

    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "training_stages.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (summary_dir / "training_stages.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
