#!/usr/bin/env python3
"""Merge offset evaluation chunks into globally indexed, symlinked artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics.saved_video_io import video_index


def parse_chunk(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--chunk must be OFFSET=RUN_ROOT")
    offset, root = value.split("=", 1)
    return int(offset), Path(root).expanduser().resolve()


def find_gif_dir(root: Path) -> Path:
    candidates = sorted(root.glob("wandb/*/files/media/videos/prediction_vis"))
    candidates += sorted(root.glob("**/files/media/videos/prediction_vis"))
    candidates = list(dict.fromkeys(path.resolve() for path in candidates if path.is_dir()))
    if not candidates:
        raise FileNotFoundError(f"No prediction_vis directory under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_raw_files(root: Path) -> list[Path]:
    direct = root / "raw"
    files = sorted(direct.glob("*/data.npz")) if direct.is_dir() else []
    if not files:
        files = sorted(root.glob("**/raw/*/data.npz"))
    return files


def replace_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", action="append", type=parse_chunk, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_count", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    gif_out = output_dir / "gifs"
    raw_out = output_dir / "raw"
    gif_out.mkdir(parents=True, exist_ok=True)
    raw_out.mkdir(parents=True, exist_ok=True)
    for path in [*gif_out.glob("video_*.gif"), *raw_out.glob("video_*.npz")]:
        if path.is_symlink():
            path.unlink()

    manifest: list[dict[str, object]] = []
    seen: set[int] = set()
    for offset, root in sorted(args.chunk, key=lambda item: item[0]):
        gif_dir = find_gif_dir(root)
        gifs = sorted(gif_dir.glob("*.gif"), key=lambda path: video_index(path) or 0)
        raw_files = find_raw_files(root)
        raw_by_local = {int(path.parent.name): path for path in raw_files if path.parent.name.isdigit()}
        for gif in gifs:
            local = video_index(gif)
            if local is None:
                continue
            global_index = offset + local
            if global_index in seen:
                raise RuntimeError(f"Duplicate global video index {global_index}")
            seen.add(global_index)
            digest_match = re.search(r"_([a-f0-9]+)\.gif$", gif.name)
            digest = digest_match.group(1) if digest_match else hashlib.sha1(str(gif).encode()).hexdigest()[:20]
            gif_target = gif_out / f"video_{global_index}_{global_index}_{digest}.gif"
            replace_symlink(gif, gif_target)
            raw_source = raw_by_local.get(local)
            raw_target = None
            if raw_source is not None:
                raw_target = raw_out / f"video_{global_index:05d}.npz"
                replace_symlink(raw_source, raw_target)
            manifest.append(
                {
                    "index": global_index,
                    "chunk_offset": offset,
                    "chunk_root": str(root),
                    "gif": str(gif_target),
                    "raw": str(raw_target) if raw_target else None,
                }
            )

    manifest.sort(key=lambda row: int(row["index"]))
    if args.expected_count is not None:
        expected = set(range(args.expected_count))
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        if missing or extra:
            raise RuntimeError(f"Merged indices mismatch: missing={missing}, extra={extra}")
        missing_raw = [row["index"] for row in manifest if row["raw"] is None]
        if missing_raw:
            raise RuntimeError(f"Missing lossless NPZ files for indices: {missing_raw}")
    payload = {
        "output_dir": str(output_dir),
        "count": len(manifest),
        "indices": [row["index"] for row in manifest],
        "chunks": [{"offset": offset, "root": str(root)} for offset, root in args.chunk],
        "items": manifest,
    }
    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Merged {len(manifest)} videos into {output_dir}")


if __name__ == "__main__":
    main()
