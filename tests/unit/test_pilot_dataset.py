"""Validate the synthetic annotation pilot."""

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

DATASET = (
    Path(__file__).resolve().parents[2]
    / "datasets/pilot/annotation_pilot_v0_1.csv"
)

EXPECTED_COLUMNS = {
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
}


def load_pilot() -> list[dict[str, str]]:
    with DATASET.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames is not None
        assert set(reader.fieldnames) == EXPECTED_COLUMNS
        return list(reader)


def test_pilot_has_balanced_labels() -> None:
    rows = load_pilot()

    assert len(rows) == 20
    assert Counter(row["label"] for row in rows) == {
        "consistent": 5,
        "new": 5,
        "updated": 5,
        "contradictory": 5,
    }


def test_pilot_has_unique_identifiers() -> None:
    rows = load_pilot()

    assert len({row["pair_id"] for row in rows}) == 20
    assert len({row["persona_id"] for row in rows}) == 20


def test_pilot_timestamps_are_valid() -> None:
    for row in load_pilot():
        old = datetime.fromisoformat(row["old_timestamp"])
        new = datetime.fromisoformat(row["new_timestamp"])

        assert old.utcoffset() is not None
        assert new.utcoffset() is not None
        assert old < new


def test_pilot_metadata_is_valid() -> None:
    rows = load_pilot()

    for row in rows:
        assert all(
            isinstance(value, str) and value.strip()
            for value in row.values()
        )
        assert row["source"] == "synthetic_manual"
        assert row["split"] == "pilot"
        assert row["marker_type"] in {
            "explicit",
            "implicit",
            "none",
        }

    markers = Counter(
        row["marker_type"]
        for row in rows
        if row["label"] == "updated"
    )

    assert markers == {"explicit": 3, "implicit": 2}
