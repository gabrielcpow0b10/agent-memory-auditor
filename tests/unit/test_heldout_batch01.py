"""Structural validation for held-out Batch 01."""

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets"

HELDOUT = DATA / "heldout/batch_01_draft.csv"

DEVELOPMENT = [
    DATA / "pilot/annotation_pilot_v0_1.csv",
    *sorted((DATA / "development").glob("batch_*_draft.csv")),
]

LABELS = {
    "consistent": "HC",
    "new": "HN",
    "updated": "HU",
    "contradictory": "HX",
}


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def normalize(text):
    return " ".join(text.casefold().split())


def pair_key(row):
    return tuple(
        sorted(
            (
                normalize(row["old_memory"]),
                normalize(row["new_memory"]),
            )
        )
    )


def test_batch01_balance():
    rows = load_rows(HELDOUT)

    assert len(rows) == 40

    counts = Counter(row["label"] for row in rows)
    assert counts == {label: 10 for label in LABELS}

    expected_ids = {
        f"{prefix}{index:03d}" for prefix in LABELS.values() for index in range(1, 11)
    }

    assert {row["pair_id"] for row in rows} == expected_ids


def test_batch01_metadata():
    rows = load_rows(HELDOUT)

    for row in rows:
        assert row["source"] == "synthetic_manual"
        assert row["split"] == "held_out_candidate"
        assert row["rationale"].strip()

        old_time = datetime.fromisoformat(row["old_timestamp"])
        new_time = datetime.fromisoformat(row["new_timestamp"])

        assert old_time.utcoffset() is not None
        assert new_time.utcoffset() is not None
        assert old_time < new_time


def test_batch01_temporal_markers():
    rows = load_rows(HELDOUT)

    for row in rows:
        if row["label"] != "updated":
            assert row["marker_type"] == "none"
        else:
            assert row["marker_type"] in {
                "explicit",
                "implicit",
            }

    updated = [row for row in rows if row["label"] == "updated"]

    counts = Counter(row["marker_type"] for row in updated)

    assert counts == {
        "explicit": 5,
        "implicit": 5,
    }


def test_batch01_development_separation():
    development = []

    for path in DEVELOPMENT:
        development.extend(load_rows(path))

    heldout = load_rows(HELDOUT)
    combined = development + heldout

    assert len(development) == 100
    assert len(combined) == 140

    assert len({row["pair_id"] for row in combined}) == 140

    assert len({row["persona_id"] for row in combined}) == 140

    assert len({pair_key(row) for row in combined}) == 140
