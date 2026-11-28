"""Validate the development corpus and its provenance."""

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PILOT = ROOT / "datasets/pilot/annotation_pilot_v0_1.csv"
DRAFT = ROOT / "datasets/development/batch_01_draft.csv"

LABELS = {
    "consistent",
    "new",
    "updated",
    "contradictory",
}


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_development_batch_is_balanced():
    rows = load_csv(DRAFT)

    assert len(rows) == 20

    counts = Counter(row["label"] for row in rows)

    assert set(counts) == LABELS
    assert all(count == 5 for count in counts.values())


def test_development_metadata_is_valid():
    rows = load_csv(DRAFT)

    for row in rows:
        assert all(isinstance(value, str) and value.strip() for value in row.values())

        assert row["source"] == "synthetic_manual"
        assert row["split"] == "development_candidate"

        old_time = datetime.fromisoformat(row["old_timestamp"])
        new_time = datetime.fromisoformat(row["new_timestamp"])

        assert old_time.utcoffset() is not None
        assert new_time.utcoffset() is not None
        assert old_time < new_time

        assert row["marker_type"] in {
            "explicit",
            "implicit",
            "none",
        }


def test_no_duplicates_across_development_data():
    pilot = load_csv(PILOT)
    draft = load_csv(DRAFT)
    combined = pilot + draft

    assert len(combined) == 40

    pair_ids = [row["pair_id"] for row in combined]
    personas = [row["persona_id"] for row in combined]

    assert len(set(pair_ids)) == 40
    assert len(set(personas)) == 40

    normalized_pairs = [
        tuple(
            sorted(
                (
                    row["old_memory"].strip().casefold(),
                    row["new_memory"].strip().casefold(),
                )
            )
        )
        for row in combined
    ]

    assert len(set(normalized_pairs)) == 40
