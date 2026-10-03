#!/usr/bin/env python3
"""Validate a conditional teacher's dependence on physical transition tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-relative-gain", type=float, default=0.00025)
    parser.add_argument("--min-win-fraction", type=float, default=0.52)
    parser.add_argument("--min-shuffle-gain", type=float, default=0.000075)
    parser.add_argument(
        "--min-condition-win-fraction",
        type=float,
        default=0.53,
    )
    parser.add_argument("--min-samples", type=int, default=256)
    parser.add_argument("--allow-failure", action="store_true")
    return parser.parse_args()


def load_metadata(path: Path) -> dict:
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    metadata = payload.get("generative_unit")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint has no Physical conditioner metadata")
    return metadata


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.checkpoint)
    if int(metadata.get("training_contract_version", -1)) != 6:
        raise ValueError(
            "Expected Physical conditioner training contract version 6, got "
            f"{metadata.get('training_contract_version')!r}"
        )
    if str(metadata.get("condition_interface")) != "paired_video_fused":
        raise ValueError(
            "Expected paired_video_fused condition interface, got "
            f"{metadata.get('condition_interface')!r}"
        )
    if str(metadata.get("objective_contract")) != (
        "calibrated_gate_rms_identity"
    ):
        raise ValueError(
            "Expected calibrated_gate_rms_identity objective contract, got "
            f"{metadata.get('objective_contract')!r}"
        )
    if str(metadata.get("mode")) != "conditional_teacher":
        raise ValueError(
            "Expected a conditional_teacher checkpoint, got "
            f"{metadata.get('mode')!r}"
        )
    quality = metadata.get("teacher_quality")
    if not isinstance(quality, dict):
        raise ValueError("Checkpoint has no teacher_quality audit")

    checks = {
        "has_samples": float(quality.get("samples", 0.0)) >= args.min_samples,
        "relative_gain": float(quality.get("ema_relative_gain", float("-inf")))
        >= args.min_relative_gain,
        "win_fraction": float(quality.get("ema_win_fraction", 0.0))
        >= args.min_win_fraction,
        "shuffle_gain": float(quality.get("ema_shuffle_gain", float("-inf")))
        >= args.min_shuffle_gain,
        "condition_win_fraction": float(
            quality.get("ema_condition_win_fraction", 0.0)
        )
        >= args.min_condition_win_fraction,
    }
    passed = all(checks.values())
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "passed": passed,
        "checks": checks,
        "thresholds": {
            "min_relative_gain": args.min_relative_gain,
            "min_win_fraction": args.min_win_fraction,
            "min_shuffle_gain": args.min_shuffle_gain,
            "min_condition_win_fraction": args.min_condition_win_fraction,
            "min_samples": args.min_samples,
        },
        "quality": quality,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "teacher_quality_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Conditional Teacher Quality",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Result: **{'PASS' if passed else 'FAIL'}**",
        f"- Samples: {float(quality.get('samples', 0.0)):.0f} "
        f"(minimum {args.min_samples})",
        "- EMA relative FM gain: "
        f"{float(quality.get('ema_relative_gain', 0.0)):.6%} "
        f"(minimum {args.min_relative_gain:.6%})",
        "- EMA teacher win fraction: "
        f"{float(quality.get('ema_win_fraction', 0.0)):.4f} "
        f"(minimum {args.min_win_fraction:.4f})",
        "- EMA correct-vs-shuffle gain: "
        f"{float(quality.get('ema_shuffle_gain', 0.0)):.6%} "
        f"(minimum {args.min_shuffle_gain:.6%})",
        "- EMA correct-vs-shuffle win fraction: "
        f"{float(quality.get('ema_condition_win_fraction', 0.0)):.4f} "
        f"(minimum {args.min_condition_win_fraction:.4f})",
    ]
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))
    if not passed and not args.allow_failure:
        raise SystemExit(
            "Conditional teacher failed quality gates; student distillation "
            "was intentionally not started."
        )


if __name__ == "__main__":
    main()
