"""Structural validation of the complete research corpus."""

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets"

HELDOUT = [DATA / f"heldout/batch_{i:02d}_draft.csv" for i in range(1, 6)]

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


def load(paths):
    rows = []

    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))

    return rows


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


def test_batch05_balance_and_ids():
    rows = load([HELDOUT[-1]])

    assert len(rows) == 40

    assert Counter(r["label"] for r in rows) == {label: 10 for label in LABELS}

    for label, prefix in LABELS.items():
        expected = {f"{prefix}{i:03d}" for i in range(41, 51)}

        actual = {r["pair_id"] for r in rows if r["label"] == label}

        assert actual == expected


def test_complete_heldout_schema_and_metadata():
    rows = load(HELDOUT)

    assert len(rows) == 200

    for row in rows:
        assert list(row) == FIELDS

        assert all(isinstance(value, str) and value.strip() for value in row.values())

        assert row["source"] == "synthetic_ai_draft"
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


def test_complete_heldout_temporal_distribution():
    rows = load(HELDOUT)

    updated = [r for r in rows if r["label"] == "updated"]

    assert len(updated) == 50

    assert Counter(r["marker_type"] for r in updated) == {
        "explicit": 25,
        "implicit": 25,
    }

    distractors = [
        r for r in rows if r["label"] != "updated" and r["marker_type"] != "none"
    ]

    assert len(distractors) == 16

    assert Counter(r["label"] for r in distractors) == {
        "new": 8,
        "contradictory": 8,
    }

    assert Counter(r["marker_type"] for r in distractors) == {
        "explicit": 15,
        "implicit": 1,
    }


def test_complete_300_pair_corpus_integrity():
    development = load(DEVELOPMENT)
    heldout = load(HELDOUT)
    combined = development + heldout

    assert len(development) == 100
    assert len(heldout) == 200
    assert len(combined) == 300

    assert Counter(r["label"] for r in development) == {label: 25 for label in LABELS}

    assert Counter(r["label"] for r in heldout) == {label: 50 for label in LABELS}

    assert Counter(r["label"] for r in combined) == {label: 75 for label in LABELS}

    assert len({r["pair_id"] for r in combined}) == 300

    assert len({r["persona_id"] for r in combined}) == 300

    assert len({pair_key(r) for r in combined}) == 300

    assert all(r["split"] == "held_out_candidate" for r in heldout)

    assert all(not r["split"].startswith("held_out") for r in development)
