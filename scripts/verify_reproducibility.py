"""Reproduce archived research metrics in an isolated directory."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    (
        "scripts.score_heldout",
        "results/heldout/metrics_v0_1.json",
    ),
    (
        "scripts.analyze_heldout",
        "results/heldout/error_diagnostics_v0_1.json",
    ),
    (
        "scripts.evaluate_rq2",
        "results/heldout/rq2_comparison_v0_1.json",
    ),
    (
        "scripts.score_rq1",
        "results/rq1/metrics_v0_1.json",
    ),
    (
        "scripts.statistics_rq1",
        "results/rq1/statistics_v0_1.json",
    ),
    (
        "scripts.score_rq3",
        "results/rq3/metrics_v0_1.json",
    ),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint_results():
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted((ROOT / "results").rglob("*"))
        if path.is_file()
    }


def main():
    original_hashes = fingerprint_results()

    print("=== RESEARCH REPRODUCIBILITY AUDIT ===")
    print("Archived result files:", len(original_hashes))

    completed = 0

    with TemporaryDirectory(prefix="ama_repro_") as directory:
        workspace = Path(directory)

        for name in (
            "configs",
            "datasets",
            "docs",
            "results",
            "scripts",
            "src",
        ):
            shutil.copytree(
                ROOT / name,
                workspace / name,
                ignore=shutil.ignore_patterns("__pycache__"),
            )

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(workspace), str(workspace / "src")])
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

        for module, filename in EXPERIMENTS:
            archived = ROOT / filename
            reproduced = workspace / filename

            assert sha256(archived) == sha256(reproduced)

            reproduced.unlink()

            process = subprocess.run(
                [sys.executable, "-m", module],
                cwd=workspace,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            if process.returncode != 0:
                print(process.stdout)
                print(process.stderr)
                process.check_returncode()

            assert reproduced.exists(), filename

            expected = json.loads(archived.read_text())
            actual = json.loads(reproduced.read_text())

            assert actual == expected, f"Reproduction mismatch: {filename}"

            identical = sha256(archived) == sha256(reproduced)

            print(f"\n{module}")
            print("JSON values: EXACT MATCH")
            print("Byte-identical:", identical)

            completed += 1

    assert fingerprint_results() == original_hashes

    print("\n=== FINAL AUDIT SUMMARY ===")
    print("Reproduced analyses:", completed)
    print("Original results preserved: PASS")
    print("New model inference: NONE")
    print("FULL REPRODUCIBILITY AUDIT: PASS")


if __name__ == "__main__":
    main()
