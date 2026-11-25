"""Structural validation for held-out Batch 03."""

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets"

BATCH01 = DATA / "heldout/batch_01_draft.csv"
BATCH02 = DATA / "heldout/batch_02_draft.csv"
BATCH03 = DATA / "heldout/batch_03_draft.csv"

LABELS = {
    "consistent": "HC",
    "new": "HN",
    "updated": "HU",
    "contradictory": "HX",
}

FIELDS = [
    "pair_id",
    "persona_id",
    "topic",
    "old_memory",
    "new_memory",
    "old_timestamp",
    "new_timestamp",
    "label",
    "marker_type",
    "source",
    "split",
    "rationale",
]


def load(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def pair_key(row):
    return tuple(
        sorted(
            " ".join(text.casefold().split())
            for text in (
                row["old_memory"],
                row["new_memory"],
            )
        )
    )


def test_batch03_balance_and_ids():
    rows = load(BATCH03)

    assert len(rows) == 40

    assert Counter(r["label"] for r in rows) == {label: 10 for label in LABELS}

    for label, prefix in LABELS.items():
        expected = {f"{prefix}{i:03d}" for i in range(21, 31)}

        actual = {r["pair_id"] for r in rows if r["label"] == label}

        assert actual == expected


def test_batch03_schema_and_metadata():
    rows = load(BATCH03)

    for row in rows:
        assert list(row) == FIELDS

        assert all(isinstance(value, str) and value.strip() for value in row.values())

        assert row["source"] == "synthetic_ai_draft"
        assert row["split"] == "held_out_candidate"

        old = datetime.fromisoformat(row["old_timestamp"])

        new = datetime.fromisoformat(row["new_timestamp"])

        assert old.utcoffset() is not None
        assert new.utcoffset() is not None
        assert old < new


def test_batch03_temporal_markers():
    rows = load(BATCH03)

    for row in rows:
        if row["label"] != "updated":
            assert row["marker_type"] == "none"

    updated = [r for r in rows if r["label"] == "updated"]

    assert Counter(r["marker_type"] for r in updated) == {
        "explicit": 5,
        "implicit": 5,
    }


def test_combined_heldout_integrity():
    paths = [
        DATA / "pilot/annotation_pilot_v0_1.csv",
        *sorted((DATA / "development").glob("batch_*_draft.csv")),
    ]

    development = []

    for path in paths:
        development.extend(load(path))

    heldout = load(BATCH01) + load(BATCH02) + load(BATCH03)

    combined = development + heldout

    assert len(development) == 100
    assert len(heldout) == 120
    assert len(combined) == 220

    assert Counter(r["label"] for r in heldout) == {label: 30 for label in LABELS}

    assert len({r["pair_id"] for r in combined}) == 220

    assert len({r["persona_id"] for r in combined}) == 220

    assert len({pair_key(r) for r in combined}) == 220

    assert all(r["split"] == "held_out_candidate" for r in heldout)
