#!/usr/bin/env python3
"""Export a compact, value-verified inference checkpoint.

Lightning training checkpoints retain Adam moments and training-only auxiliary
modules.  Video evaluation only consumes the diffusion model, so repeatedly
loading those tensors wastes disk bandwidth without changing inference.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "map_location": "cpu",
        "weights_only": False,
    }
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def export_checkpoint(source: Path, output: Path, prefix: str) -> dict[str, Any]:
    payload = load_checkpoint(source)
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no state_dict: {source}")
    selected = {
        key: value
        for key, value in state.items()
        if key.startswith(prefix) and isinstance(value, torch.Tensor)
    }
    if not selected:
        raise ValueError(f"No tensors start with {prefix!r} in {source}")
    rejected_model_keys = [
        key
        for key, value in state.items()
        if isinstance(value, torch.Tensor) and not key.startswith(prefix)
    ]

    # These fields describe training-only modules that are deliberately absent
    # from a diffusion-only export.  Keeping them makes the base video model
    # attempt UPCAST architecture validation while loading the compact weights.
    training_only_metadata = {"spatial_unit"}
    compact = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "state_dict",
            "optimizer_states",
            *training_only_metadata,
        }
    }
    compact["state_dict"] = selected
    compact["optimizer_states"] = []
    compact["inference_export"] = {
        "source": str(source.resolve()),
        "state_prefix": prefix,
        "tensor_count": len(selected),
        "discarded_model_tensor_count": len(rejected_model_keys),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        torch.save(compact, temporary)
        verified = load_checkpoint(temporary)
        verified_state = verified.get("state_dict", {})
        if list(verified_state) != list(selected):
            raise RuntimeError("Exported checkpoint tensor keys changed during serialization")
        for key, expected in selected.items():
            actual = verified_state[key]
            if actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise RuntimeError(f"Exported tensor metadata mismatch for {key}")
            if not torch.equal(actual, expected):
                raise RuntimeError(f"Exported tensor values differ for {key}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "source_bytes": source.stat().st_size,
        "output_bytes": output.stat().st_size,
        "tensor_count": len(selected),
        "discarded_model_tensor_count": len(rejected_model_keys),
        "discarded_metadata_keys": sorted(training_only_metadata & payload.keys()),
        "global_step": payload.get("global_step"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prefix", default="diffusion_model.")
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    report = export_checkpoint(args.input, args.output, args.prefix)
    print(report, flush=True)


if __name__ == "__main__":
    main()
