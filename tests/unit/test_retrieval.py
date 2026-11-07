"""Unit tests for semantic retrieval."""

from datetime import UTC, datetime

import numpy as np
import pytest

from agent_memory_auditor.retrieval import retrieve_top_k
from agent_memory_auditor.schemas import MemoryRecord


class FakeEncoder:
    def encode(
        self, sentences: list[str], *, normalize_embeddings: bool
    ) -> np.ndarray:
        assert normalize_embeddings
        assert len(sentences) == 4
        return np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.6],
                [1.0, 0.0],
            ]
        )


def make_memory(memory_id: str) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        text=f"Synthetic memory {memory_id}",
        timestamp=datetime(2026, 11, 7, tzinfo=UTC),
    )


def test_retrieval_ranks_related_memory_first() -> None:
    history = tuple(make_memory(f"m{i}") for i in range(3))
    results = retrieve_top_k(
        history,
        make_memory("new"),
        FakeEncoder(),
        top_k=2,
    )

    assert [item.memory_id for item in results] == ["m0", "m2"]
    assert [item.rank for item in results] == [1, 2]
    assert results[0].score == pytest.approx(1.0)


def test_empty_history_returns_no_candidates() -> None:
    results = retrieve_top_k(
        (),
        make_memory("new"),
        FakeEncoder(),
    )
    assert results == []


def test_invalid_top_k_is_rejected() -> None:
    with pytest.raises(ValueError):
        retrieve_top_k(
            (make_memory("m0"),),
            make_memory("new"),
            FakeEncoder(),
            top_k=0,
        )


def test_top_k_is_limited_by_history_size() -> None:
    history = tuple(make_memory(f"m{i}") for i in range(3))
    results = retrieve_top_k(
        history,
        make_memory("new"),
        FakeEncoder(),
        top_k=5,
    )
    assert len(results) == 3
