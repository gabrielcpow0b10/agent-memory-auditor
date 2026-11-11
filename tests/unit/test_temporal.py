"""Tests for temporal memory-change detection."""

import csv
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_memory_auditor.schemas import MarkerType, MemoryRecord
from agent_memory_auditor.temporal import detect_temporal_change


def memory(
    memory_id: str,
    text: str,
    month: int,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        text=text,
        timestamp=datetime(2026, month, 10, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    ("new_text", "expected_type", "candidate"),
    [
        ("I moved to Boston.", MarkerType.EXPLICIT, True),
        (
            "I no longer work at the bank.",
            MarkerType.EXPLICIT,
            True,
        ),
        (
            "I switched my preferred meeting time to 11 AM.",
            MarkerType.EXPLICIT,
            True,
        ),
        (
            "I started using an iPhone.",
            MarkerType.IMPLICIT,
            True,
        ),
        (
            "I now realize I have never liked coffee.",
            MarkerType.EXPLICIT,
            False,
        ),
        ("I enjoy chess.", MarkerType.NONE, False),
    ],
)
def test_temporal_detection(
    new_text: str,
    expected_type: MarkerType,
    candidate: bool,
) -> None:
    result = detect_temporal_change(
        memory("old", "Previous fact.", 1),
        memory("new", new_text, 10),
    )

    assert result.marker_type == expected_type
    assert result.candidate_update is candidate


def test_timestamp_alone_does_not_prove_change() -> None:
    result = detect_temporal_change(
        memory("old", "I live in Miami.", 1),
        memory("new", "I work as a programmer.", 10),
    )

    assert result.later_timestamp
    assert not result.candidate_update


def test_older_incoming_timestamp_does_not_support_update() -> None:
    result = detect_temporal_change(
        memory("old", "I live in Miami.", 10),
        memory("new", "I moved to Boston.", 1),
    )

    assert not result.later_timestamp
    assert not result.candidate_update


def test_pilot_temporal_annotations() -> None:
    path = (
        Path(__file__).resolve().parents[2] / "datasets/pilot/annotation_pilot_v0_1.csv"
    )

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 20

    for row in rows:
        old = MemoryRecord(
            memory_id=f"{row['pair_id']}-old",
            text=row["old_memory"],
            timestamp=datetime.fromisoformat(row["old_timestamp"]),
        )
        new = MemoryRecord(
            memory_id=f"{row['pair_id']}-new",
            text=row["new_memory"],
            timestamp=datetime.fromisoformat(row["new_timestamp"]),
        )

        result = detect_temporal_change(old, new)

        assert result.marker_type.value == row["marker_type"], row["pair_id"]

        assert result.candidate_update == (row["label"] == "updated"), row["pair_id"]
