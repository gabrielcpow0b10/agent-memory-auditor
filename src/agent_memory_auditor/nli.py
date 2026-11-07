"""Traceable bidirectional natural language inference."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from agent_memory_auditor.schemas import MemoryRecord


class NLILabel(str, Enum):
    """The three standard NLI classes."""

    CONTRADICTION = "contradiction"
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"


LABEL_ORDER = tuple(NLILabel)


class NLIModel(Protocol):
    """Interface for a three-class NLI model."""

    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        apply_softmax: bool,
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class NLIScores:
    """Scores for one premise-hypothesis direction."""

    contradiction: float
    entailment: float
    neutral: float

    @property
    def predicted_label(self) -> NLILabel:
        values = (
            self.contradiction,
            self.entailment,
            self.neutral,
        )
        return LABEL_ORDER[int(np.argmax(values))]


@dataclass(frozen=True, slots=True)
class BidirectionalNLIResult:
    """Both NLI directions for one memory pair."""

    old_memory_id: str
    new_memory_id: str
    forward: NLIScores
    reverse: NLIScores


def score_bidirectional(
    old: MemoryRecord,
    new: MemoryRecord,
    model: NLIModel,
) -> BidirectionalNLIResult:
    """Score both directions without resolving the memory label."""

    pairs = [
        (old.text, new.text),
        (new.text, old.text),
    ]

    probabilities = np.asarray(
        model.predict(pairs, apply_softmax=True),
        dtype=np.float64,
    )

    if probabilities.shape != (2, 3):
        raise ValueError("NLI model must return a (2, 3) array")

    if not np.isfinite(probabilities).all():
        raise ValueError("NLI probabilities must be finite")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("NLI probabilities must be between 0 and 1")

    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("NLI probabilities must sum to one")

    def to_scores(row: np.ndarray) -> NLIScores:
        return NLIScores(
            contradiction=float(row[0]),
            entailment=float(row[1]),
            neutral=float(row[2]),
        )

    return BidirectionalNLIResult(
        old_memory_id=old.memory_id,
        new_memory_id=new.memory_id,
        forward=to_scores(probabilities[0]),
        reverse=to_scores(probabilities[1]),
    )
