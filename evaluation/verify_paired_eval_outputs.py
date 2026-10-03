#!/usr/bin/env python3
"""Fail fast when supposedly paired video-evaluation outputs are not paired."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be NAME=RAW_DIRECTORY")
    name, directory = value.split("=", 1)
    return name, Path(directory).expanduser().resolve()


def tensor_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--expected_count", type=int, required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--allow-extra",
        action="store_true",
        help="Audit indices 0..N-1 while allowing additional files in a run directory.",
    )
    args = parser.parse_args()

    expected_indices = list(range(args.expected_count))
    reference_hashes: dict[int, str] = {}
    reference_shapes: dict[int, tuple[int, ...]] = {}
    report: dict[str, object] = {
        "expected_count": args.expected_count,
        "methods": {},
        "paired": True,
    }

    for run_index, (name, directory) in enumerate(args.run):
        files = {int(path.stem.split("_")[-1]): path for path in directory.glob("video_*.npz")}
        indices = sorted(files)
        audited_indices = [index for index in expected_indices if index in files]
        if audited_indices != expected_indices or (not args.allow_extra and indices != expected_indices):
            raise RuntimeError(
                f"{name} indices are not 0..{args.expected_count - 1}: {indices}"
            )

        method_hashes: dict[int, str] = {}
        for index in expected_indices:
            with np.load(files[index], allow_pickle=False) as payload:
                if "gt" not in payload or "gen" not in payload:
                    raise RuntimeError(f"{files[index]} must contain gt and gen arrays")
                gt = payload["gt"]
                generated = payload["gen"]
                if gt.shape != generated.shape:
                    raise RuntimeError(
                        f"{name} sample {index} shape mismatch: gt={gt.shape}, gen={generated.shape}"
                    )
                if not np.isfinite(gt).all() or not np.isfinite(generated).all():
                    raise RuntimeError(f"{name} sample {index} contains non-finite values")
                digest = tensor_digest(gt)
                method_hashes[index] = digest
                if run_index == 0:
                    reference_hashes[index] = digest
                    reference_shapes[index] = tuple(gt.shape)
                elif digest != reference_hashes[index]:
                    raise RuntimeError(
                        f"GT mismatch for {name} sample {index}; methods are not paired"
                    )
                elif tuple(gt.shape) != reference_shapes[index]:
                    raise RuntimeError(f"GT shape mismatch for {name} sample {index}")

        aggregate = hashlib.sha256(
            "".join(method_hashes[index] for index in expected_indices).encode()
        ).hexdigest()
        report["methods"][name] = {
            "directory": str(directory),
            "count": len(expected_indices),
            "available_count": len(indices),
            "aggregate_gt_sha256": aggregate,
        }

    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(args.run)} paired runs x {args.expected_count} samples: {output}")


if __name__ == "__main__":
    main()
