"""Unit tests for bidirectional NLI."""

from datetime import UTC, datetime

import numpy as np
import pytest

from agent_memory_auditor.nli import (
    NLILabel,
    score_bidirectional,
)
from agent_memory_auditor.schemas import MemoryRecord


def memory(memory_id: str, text: str) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        text=text,
        timestamp=datetime(2026, 11, 7, tzinfo=UTC),
    )


class FakeNLIModel:
    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        apply_softmax: bool,
    ) -> np.ndarray:
        assert apply_softmax is True
        assert len(sentences) == 2
        return np.array([
            [0.05, 0.90, 0.05],
            [0.95, 0.02, 0.03],
        ])


def test_bidirectional_scores_are_preserved() -> None:
    result = score_bidirectional(
        memory("m1", "I live in Miami."),
        memory("m2", "I reside in Miami."),
        FakeNLIModel(),
    )

    assert result.old_memory_id == "m1"
    assert result.new_memory_id == "m2"
    assert result.forward.entailment == pytest.approx(0.90)
    assert result.reverse.contradiction == pytest.approx(0.95)


def test_directions_can_disagree() -> None:
    result = score_bidirectional(
        memory("m1", "Old fact."),
        memory("m2", "New fact."),
        FakeNLIModel(),
    )

    assert result.forward.predicted_label == NLILabel.ENTAILMENT
    assert result.reverse.predicted_label == NLILabel.CONTRADICTION


def test_invalid_probability_shape_is_rejected() -> None:
    class InvalidModel:
        def predict(self, sentences, *, apply_softmax):
            return np.array([[0.5, 0.5]])

    with pytest.raises(ValueError):
        score_bidirectional(
            memory("m1", "Old fact."),
            memory("m2", "New fact."),
            InvalidModel(),
        )


def test_nonfinite_probabilities_are_rejected() -> None:
    class InvalidModel:
        def predict(self, sentences, *, apply_softmax):
            return np.array([
                [float("nan"), 0.5, 0.5],
                [0.2, 0.3, 0.5],
            ])

    with pytest.raises(ValueError):
        score_bidirectional(
            memory("m1", "Old fact."),
            memory("m2", "New fact."),
            InvalidModel(),
        )
