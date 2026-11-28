"""Validate Batch 04 and the complete development corpus."""

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets"

FILES = [
    DATA / "pilot/annotation_pilot_v0_1.csv",
    DATA / "development/batch_01_draft.csv",
    DATA / "development/batch_02_draft.csv",
    DATA / "development/batch_03_draft.csv",
    DATA / "development/batch_04_draft.csv",
]

LABELS = ("consistent", "new", "updated", "contradictory")


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_batch04_balanced_labels():
    rows = load_rows(FILES[-1])

    assert len(rows) == 20

    assert Counter(row["label"] for row in rows) == {label: 5 for label in LABELS}

    for label, prefix in zip(LABELS, ("C", "N", "U", "X")):
        expected = {f"{prefix}{i:03d}" for i in range(21, 26)}
        actual = {row["pair_id"] for row in rows if row["label"] == label}

        assert actual == expected


def test_batch04_metadata():
    rows = load_rows(FILES[-1])

    for row in rows:
        assert all(isinstance(value, str) and value.strip() for value in row.values())

        assert row["source"] == "synthetic_manual"
        assert row["split"] == "development_candidate"

        assert row["marker_type"] in {
            "explicit",
            "implicit",
            "none",
        }

        old = datetime.fromisoformat(row["old_timestamp"])
        new = datetime.fromisoformat(row["new_timestamp"])

        assert old.utcoffset() is not None
        assert new.utcoffset() is not None
        assert old < new


def test_complete_development_integrity():
    rows = []

    for path in FILES:
        rows.extend(load_rows(path))

    assert len(rows) == 100

    assert len({row["pair_id"] for row in rows}) == 100

    assert len({row["persona_id"] for row in rows}) == 100

    assert Counter(row["label"] for row in rows) == {label: 25 for label in LABELS}

    pairs = [
        tuple(
            sorted(
                (
                    row["old_memory"].strip().casefold(),
                    row["new_memory"].strip().casefold(),
                )
            )
        )
        for row in rows
    ]

    assert len(set(pairs)) == 100

    assert all(row["split"] != "held_out" for row in rows)


def test_batch04_temporal_metadata():
    rows = {row["pair_id"]: row for row in load_rows(FILES[-1])}

    for i in range(21, 26):
        row = rows[f"U{i:03d}"]

        assert row["label"] == "updated"

        marker = "explicit" if i <= 23 else "implicit"

        assert row["marker_type"] == marker

    for prefix in ("C", "N", "X"):
        for i in range(21, 26):
            row = rows[f"{prefix}{i:03d}"]
            assert row["marker_type"] == "none"
