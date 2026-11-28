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
    dataset_path = ROOT / "datasets/heldout/dataset_freeze_v0_1.json"
    dataset = json.loads(dataset_path.read_text())

    if "metadata_correction" in dataset:
        verify_corrected_metadata(dataset)
        return

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



def verify_corrected_metadata(dataset):
    """Verify corrected metadata and reproduce the historical experiment."""
    import csv
    import tarfile

    correction = dataset["metadata_correction"]
    revision = correction["historical_revision"]

    for filename, expected in dataset["file_sha256"].items():
        assert sha256(ROOT / filename) == expected, filename

    policy_path = ROOT / "docs/policy_freeze_v0_1.json"
    policy = json.loads(policy_path.read_text())

    for filename, expected in policy["file_sha256"].items():
        assert sha256(ROOT / filename) == expected, filename

    print("Current manifests: PASS", flush=True)

    with TemporaryDirectory(prefix="ama_history_") as directory:
        workspace = Path(directory)
        archive = workspace / "historical.tar"

        subprocess.run(
            ["git", "archive", "--output", str(archive), revision],
            cwd=ROOT,
            check=True,
        )

        with tarfile.open(archive) as file:
            file.extractall(workspace, filter="data")

        archive.unlink()

        historical_manifest = (
            workspace / "datasets/heldout/dataset_freeze_v0_1.json"
        )

        assert sha256(historical_manifest) == (
            correction["historical_manifest_sha256"]
        )

        historical_policy = workspace / "docs/policy_freeze_v0_1.json"

        assert sha256(historical_policy) == (
            policy["metadata_correction"]["historical_manifest_sha256"]
        )

        changed = 0

        for folder in ("datasets", "results"):
            for original in sorted((workspace / folder).rglob("*.csv")):
                relative = original.relative_to(workspace)
                current = ROOT / relative

                if original.read_bytes() == current.read_bytes():
                    continue

                with original.open(newline="") as file:
                    before = list(csv.DictReader(file))

                with current.open(newline="") as file:
                    after = list(csv.DictReader(file))

                assert len(before) == len(after), relative

                for old, new in zip(before, after):
                    assert old.keys() == new.keys(), relative

                    for field in old:
                        if old[field] != new[field]:
                            assert field in ("source", "method"), relative
                            assert new[field] in (
                                "synthetic_manual",
                                "manual_protocol_adjudication",
                            ), relative

                changed += 1

        assert changed == 12

        for original in (workspace / "results").rglob("*"):
            if not original.is_file():
                continue

            relative = original.relative_to(workspace)

            if relative.as_posix() == (
                "results/development/policy_comparison_candidates_v0_1.csv"
            ):
                continue

            assert sha256(ROOT / relative) == sha256(original), relative

        freeze = Path("docs/rq3_freeze_v0_1.json")
        assert sha256(ROOT / freeze) == sha256(workspace / freeze)

        print("CSV metadata correction: PASS", flush=True)
        print("Scientific result integrity: PASS", flush=True)
        print("Historical manifest integrity: PASS", flush=True)

        subprocess.run(
            [sys.executable, "-m", "scripts.verify_reproducibility"],
            cwd=workspace,
            check=True,
        )

    print("METADATA CORRECTION AUDIT: PASS")


if __name__ == "__main__":
    main()
