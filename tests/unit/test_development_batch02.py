"""Validate Batch 02 and the combined development corpus."""

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets"

PILOT = DATA / "pilot/annotation_pilot_v0_1.csv"
BATCH01 = DATA / "development/batch_01_draft.csv"
BATCH02 = DATA / "development/batch_02_draft.csv"

LABELS = {
    "consistent": 5,
    "new": 5,
    "updated": 5,
    "contradictory": 5,
}


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_batch02_balanced():
    rows = load_rows(BATCH02)

    assert len(rows) == 20
    assert Counter(row["label"] for row in rows) == LABELS

    for label, prefix in (
        ("consistent", "C"),
        ("new", "N"),
        ("updated", "U"),
        ("contradictory", "X"),
    ):
        expected = {f"{prefix}{number:03d}" for number in range(11, 16)}
        actual = {row["pair_id"] for row in rows if row["label"] == label}
        assert actual == expected


def test_batch02_metadata():
    rows = load_rows(BATCH02)

    for row in rows:
        assert all(isinstance(value, str) and value.strip() for value in row.values())
        assert row["source"] == "synthetic_ai_draft"
        assert row["split"] == "development_candidate"
        assert row["marker_type"] in {"explicit", "implicit", "none"}

        old_time = datetime.fromisoformat(row["old_timestamp"])
        new_time = datetime.fromisoformat(row["new_timestamp"])

        assert old_time.utcoffset() is not None
        assert new_time.utcoffset() is not None
        assert old_time < new_time


def test_combined_development_integrity():
    pilot = load_rows(PILOT)
    batch01 = load_rows(BATCH01)
    batch02 = load_rows(BATCH02)

    rows = pilot + batch01 + batch02

    assert len(rows) == 60
    assert len({row["pair_id"] for row in rows}) == 60
    assert len({row["persona_id"] for row in rows}) == 60

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

    assert len(set(pairs)) == 60

    counts = Counter(row["label"] for row in rows)
    assert all(count == 15 for count in counts.values())
    assert set(counts) == set(LABELS)

    assert all(row["split"] != "held_out" for row in rows)
