"""Isolated tests for the original Stage 2 four-label decision baseline."""

from datetime import UTC, datetime

import numpy as np
import pytest

from agent_memory_auditor.pipeline import PipelineConfig, audit_memory
from agent_memory_auditor.schemas import AuditRequest, MemoryLabel, MemoryRecord


class FakeEncoder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(self, sentences: list[str], *, normalize_embeddings: bool) -> np.ndarray:
        assert normalize_embeddings
        return np.asarray([self.vectors[text] for text in sentences])


class FakeNLI:
    def __init__(
        self,
        outputs: dict[str, tuple[list[float], list[float]]],
    ) -> None:
        self.outputs = outputs
        self.calls = 0

    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        apply_softmax: bool,
    ) -> np.ndarray:
        assert apply_softmax
        assert sentences[0] == sentences[1][::-1]
        self.calls += 1
        return np.asarray(self.outputs[sentences[0][0]])


def memory(name: str, text: str, month: int = 1) -> MemoryRecord:
    return MemoryRecord(
        memory_id=name,
        text=text,
        timestamp=datetime(2026, month, 10, tzinfo=UTC),
    )


E = [0.01, 0.98, 0.01]
C = [0.98, 0.01, 0.01]
N = [0.01, 0.01, 0.98]


def test_empty_history_is_new() -> None:
    incoming = memory("new", "New fact.", 10)
    result = audit_memory(AuditRequest((), incoming), FakeEncoder({}), FakeNLI({}))
    assert result.label is MemoryLabel.NEW
    assert result.evidence_id is None


def test_equivalent_information_is_consistent() -> None:
    old = memory("old", "I live in Miami.")
    incoming = memory("new", "I reside in Miami.", 10)
    vectors = {old.text: [1.0, 0.0], incoming.text: [1.0, 0.0]}
    result = audit_memory(
        AuditRequest((old,), incoming),
        FakeEncoder(vectors),
        FakeNLI({old.text: (E, E)}),
    )
    assert result.label is MemoryLabel.CONSISTENT
    assert result.evidence_id == "old"
    assert result.similarity == pytest.approx(1.0)


def test_unrelated_memory_below_threshold_is_new_without_nli() -> None:
    old = memory("old", "I have a dog.")
    incoming = memory("new", "I like chess.", 10)
    model = FakeNLI({})
    result = audit_memory(
        AuditRequest((old,), incoming),
        FakeEncoder({old.text: [1.0, 0.0], incoming.text: [0.0, 1.0]}),
        model,
    )
    assert result.label is MemoryLabel.NEW
    assert model.calls == 0


def test_neutral_nli_is_new() -> None:
    old = memory("old", "I have a dog.")
    incoming = memory("new", "I also have a cat.", 10)
    result = audit_memory(
        AuditRequest((old,), incoming),
        FakeEncoder({old.text: [1.0, 0.0], incoming.text: [1.0, 0.0]}),
        FakeNLI({old.text: (N, N)}),
    )
    assert result.label is MemoryLabel.NEW


def test_contradiction_without_change_is_contradictory() -> None:
    old = memory("old", "I was born in 1990.")
    incoming = memory("new", "I was born in 1992.", 10)
    result = audit_memory(
        AuditRequest((old,), incoming),
        FakeEncoder({old.text: [1.0, 0.0], incoming.text: [1.0, 0.0]}),
        FakeNLI({old.text: (C, C)}),
    )
    assert result.label is MemoryLabel.CONTRADICTORY
    assert result.evidence_id == "old"


def test_contradiction_with_change_is_updated() -> None:
    old = memory("old", "I live in Miami.")
    incoming = memory("new", "I moved to Boston.", 10)
    result = audit_memory(
        AuditRequest((old,), incoming),
        FakeEncoder({old.text: [1.0, 0.0], incoming.text: [1.0, 0.0]}),
        FakeNLI({old.text: (C, C)}),
    )
    assert result.label is MemoryLabel.UPDATED
    assert result.change_signals == ("moved to",)


def test_now_alone_does_not_create_update() -> None:
    old = memory("old", "I have always enjoyed coffee.")
    incoming = memory("new", "I now realize I never liked coffee.", 10)
    result = audit_memory(
        AuditRequest((old,), incoming),
        FakeEncoder({old.text: [1.0, 0.0], incoming.text: [1.0, 0.0]}),
        FakeNLI({old.text: (C, C)}),
    )
    assert result.label is MemoryLabel.CONTRADICTORY
    assert "now" in result.change_signals


def test_reverse_direction_can_trigger_contradiction() -> None:
    old = memory("old", "Older fact.")
    incoming = memory("new", "Later fact.", 10)
    result = audit_memory(
        AuditRequest((old,), incoming),
        FakeEncoder({old.text: [1.0, 0.0], incoming.text: [1.0, 0.0]}),
        FakeNLI({old.text: (N, C)}),
    )
    assert result.label is MemoryLabel.CONTRADICTORY


def test_contradiction_has_priority_over_entailment() -> None:
    a = memory("a", "Equivalent fact.")
    b = memory("b", "Conflicting fact.")
    incoming = memory("new", "New statement.", 10)
    vectors = {
        a.text: [1.0, 0.0],
        b.text: [0.8, 0.6],
        incoming.text: [1.0, 0.0],
    }
    result = audit_memory(
        AuditRequest((a, b), incoming),
        FakeEncoder(vectors),
        FakeNLI({a.text: (E, E), b.text: (C, C)}),
    )
    assert result.label is MemoryLabel.CONTRADICTORY
    assert result.evidence_id == "b"


@pytest.mark.parametrize(
    "arguments",
    [
        {"top_k": 0},
        {"similarity_threshold": float("nan")},
        {"contradiction_threshold": 1.5},
        {"entailment_threshold": -0.1},
    ],
)
def test_invalid_pipeline_configuration_is_rejected(arguments: dict) -> None:
    with pytest.raises(ValueError):
        PipelineConfig(**arguments)
