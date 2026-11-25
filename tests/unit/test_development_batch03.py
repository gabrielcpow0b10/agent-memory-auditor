"""Validate Batch 03 and combined development data."""

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
]

LABELS = ("consistent", "new", "updated", "contradictory")


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_batch03_labels_and_ids():
    rows = load_rows(FILES[-1])

    assert len(rows) == 20
    assert Counter(r["label"] for r in rows) == {label: 5 for label in LABELS}

    for label, prefix in zip(LABELS, ("C", "N", "U", "X")):
        actual = {r["pair_id"] for r in rows if r["label"] == label}
        expected = {f"{prefix}{i:03d}" for i in range(16, 21)}
        assert actual == expected


def test_batch03_metadata_and_dates():
    rows = load_rows(FILES[-1])

    for row in rows:
        assert all(isinstance(v, str) and v.strip() for v in row.values())
        assert row["source"] == "synthetic_ai_draft"
        assert row["split"] == "development_candidate"
        assert row["marker_type"] in {"explicit", "implicit", "none"}

        old = datetime.fromisoformat(row["old_timestamp"])
        new = datetime.fromisoformat(row["new_timestamp"])

        assert old.utcoffset() is not None
        assert new.utcoffset() is not None
        assert old < new


def test_combined_80_pair_integrity():
    rows = []
    for path in FILES:
        rows.extend(load_rows(path))

    assert len(rows) == 80
    assert len({r["pair_id"] for r in rows}) == 80
    assert len({r["persona_id"] for r in rows}) == 80

    counts = Counter(r["label"] for r in rows)
    assert counts == {label: 20 for label in LABELS}

    pairs = [
        tuple(
            sorted(
                (
                    r["old_memory"].strip().casefold(),
                    r["new_memory"].strip().casefold(),
                )
            )
        )
        for r in rows
    ]

    assert len(set(pairs)) == 80
    assert all(r["split"] != "held_out" for r in rows)


def test_batch03_temporal_and_topic_metadata():
    rows = {r["pair_id"]: r for r in load_rows(FILES[-1])}

    assert rows["C018"]["topic"] == "licensing_history"

    for i in range(16, 21):
        row = rows[f"U{i:03d}"]
        assert row["label"] == "updated"

        expected_marker = "explicit" if i <= 18 else "implicit"
        assert row["marker_type"] == expected_marker

    for prefix in ("C", "N", "X"):
        for i in range(16, 21):
            assert rows[f"{prefix}{i:03d}"]["marker_type"] == "none"
