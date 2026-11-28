"""Verify the supplementary single-memory diagnostic."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_single_memory_diagnostic(tmp_path):
    output = tmp_path / "diagnostic.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.diagnose_single_memory",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    archived = Path(
        "results/diagnostics/single_memory_v0_1.json"
    )

    actual = json.loads(output.read_text())
    expected = json.loads(archived.read_text())

    assert actual == expected
    assert actual["development_pairs"] == 100
    assert actual["heldout_pairs"] == 200

    metrics = actual["metrics"]

    assert metrics["old_memory"]["correct"] == 100
    assert metrics["new_memory"]["correct"] == 105

    assert metrics["old_memory"]["accuracy"] == pytest.approx(0.50)
    assert metrics["new_memory"]["accuracy"] == pytest.approx(0.525)
