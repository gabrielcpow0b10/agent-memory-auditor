"""Unit tests for the memory data contracts."""

from datetime import UTC, datetime

import pytest

from agent_memory_auditor.schemas import (
    AuditRequest,
    AuditResult,
    MemoryLabel,
    MemoryRecord,
)


def make_memory(memory_id: str, text: str) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        text=text,
        timestamp=datetime(2026, 11, 7, tzinfo=UTC),
    )


def test_four_memory_labels() -> None:
    assert len(MemoryLabel) == 4


def test_valid_memory_record() -> None:
    memory = make_memory("m1", "I live in Miami.")
    assert memory.memory_id == "m1"


def test_empty_memory_rejected() -> None:
    with pytest.raises(ValueError):
        make_memory("m1", "   ")


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(ValueError):
        MemoryRecord(
            memory_id="m1",
            text="I live in Miami.",
            timestamp=datetime(2026, 11, 7),
        )


def test_duplicate_memory_ids_rejected() -> None:
    old = make_memory("m1", "I live in Miami.")
    incoming = make_memory("m1", "I moved to Boston.")

    with pytest.raises(ValueError):
        AuditRequest(history=(old,), incoming=incoming)


def test_new_memory_without_evidence() -> None:
    result = AuditResult(
        label=MemoryLabel.NEW,
        incoming_id="m2",
    )
    assert result.evidence_id is None


def test_contradiction_requires_evidence() -> None:
    with pytest.raises(ValueError):
        AuditResult(
            label=MemoryLabel.CONTRADICTORY,
            incoming_id="m2",
        )


def test_similarity_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        AuditResult(
            label=MemoryLabel.CONSISTENT,
            incoming_id="m2",
            evidence_id="m1",
            similarity=1.5,
        )
