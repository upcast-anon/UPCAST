from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "evaluation" / "verify_paired_eval_outputs.py"


def write_sample(directory: Path, index: int, value: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    gt = np.full((2, 3, 4, 4), value, dtype=np.uint8)
    np.savez(directory / f"video_{index:05d}.npz", gt=gt, gen=gt.copy())


def test_paired_audit_accepts_identical_gt(tmp_path: Path) -> None:
    for index in range(2):
        write_sample(tmp_path / "a", index, index)
        write_sample(tmp_path / "b", index, index)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run",
            f"A={tmp_path / 'a'}",
            "--run",
            f"B={tmp_path / 'b'}",
            "--expected_count",
            "2",
            "--output_json",
            str(tmp_path / "report.json"),
        ],
        check=True,
    )
    assert (tmp_path / "report.json").is_file()


def test_paired_audit_rejects_different_gt(tmp_path: Path) -> None:
    write_sample(tmp_path / "a", 0, 0)
    write_sample(tmp_path / "b", 0, 1)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run",
            f"A={tmp_path / 'a'}",
            "--run",
            f"B={tmp_path / 'b'}",
            "--expected_count",
            "1",
            "--output_json",
            str(tmp_path / "report.json"),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "GT mismatch" in result.stderr
