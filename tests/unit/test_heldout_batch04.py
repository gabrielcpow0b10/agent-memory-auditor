"""Structural validation for held-out Batch 04."""

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets"

BATCHES = [DATA / f"heldout/batch_{i:02d}_draft.csv" for i in range(1, 5)]

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


def test_batch04_balance_and_ids():
    rows = load(BATCHES[-1])

    assert len(rows) == 40

    assert Counter(r["label"] for r in rows) == {label: 10 for label in LABELS}

    for label, prefix in LABELS.items():
        expected = {f"{prefix}{i:03d}" for i in range(31, 41)}

        actual = {r["pair_id"] for r in rows if r["label"] == label}

        assert actual == expected


def test_batch04_schema_and_metadata():
    rows = load(BATCHES[-1])

    for row in rows:
        assert list(row) == FIELDS

        assert all(isinstance(value, str) and value.strip() for value in row.values())

        assert row["source"] == "synthetic_manual"
        assert row["split"] == "held_out_candidate"

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


def test_batch04_temporal_distractors():
    rows = load(BATCHES[-1])

    updated = [r for r in rows if r["label"] == "updated"]

    assert Counter(r["marker_type"] for r in updated) == {
        "explicit": 5,
        "implicit": 5,
    }

    distractors = [
        r for r in rows if r["label"] != "updated" and r["marker_type"] == "explicit"
    ]

    assert len(distractors) == 8

    assert Counter(r["label"] for r in distractors) == {
        "new": 4,
        "contradictory": 4,
    }


def test_combined_heldout_batch04_integrity():
    paths = [
        DATA / "pilot/annotation_pilot_v0_1.csv",
        *sorted((DATA / "development").glob("batch_*_draft.csv")),
    ]

    development = []

    for path in paths:
        development.extend(load(path))

    heldout = []

    for path in BATCHES:
        heldout.extend(load(path))

    combined = development + heldout

    assert len(development) == 100
    assert len(heldout) == 160
    assert len(combined) == 260

    assert Counter(r["label"] for r in heldout) == {label: 40 for label in LABELS}

    assert len({r["pair_id"] for r in combined}) == 260

    assert len({r["persona_id"] for r in combined}) == 260

    assert len({pair_key(r) for r in combined}) == 260

    assert all(r["split"] == "held_out_candidate" for r in heldout)
