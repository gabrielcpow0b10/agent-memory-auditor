"""Tests for controlled retrieval experiments."""

import numpy as np
import pytest

from scripts.evaluate_rq1 import (
    build_history,
    classify_candidates,
    select_candidates,
)
from scripts.prepare_rq1 import normalize, valid_triple
from scripts.statistics_rq1 import macro_f1


@pytest.fixture
def example_case():
    return {
        "old_memory": "I live in Miami.",
        "incoming_memory": "I live in Boston.",
        "distractors": [{"text": f"Distractor memory {i}"} for i in range(99)],
    }


@pytest.mark.parametrize("size", [20, 50, 100])
@pytest.mark.parametrize("position", ["old", "middle", "recent"])
def test_controlled_history(example_case, size, position):
    history, incoming = build_history(example_case, size, position)

    expected_index = {
        "old": 0,
        "middle": size // 2,
        "recent": size - 1,
    }[position]

    assert len(history) == size
    assert history[expected_index].memory_id == "target"
    assert len({m.memory_id for m in history}) == size

    assert all(m.timestamp < incoming.timestamp for m in history)


@pytest.mark.parametrize(
    ("method", "count"),
    [
        ("recency", 5),
        ("exhaustive", 20),
        ("oracle", 1),
    ],
)
def test_candidate_counts(example_case, method, count):
    history, incoming = build_history(example_case, 20, "old")

    candidates = select_candidates(history, incoming, None, method)

    assert len(candidates) == count


def test_recency_misses_old_target(example_case):
    history, incoming = build_history(example_case, 20, "old")

    candidates = select_candidates(history, incoming, None, "recency")

    assert all(m.memory_id != "target" for m in candidates)


def test_recency_finds_recent_target(example_case):
    history, incoming = build_history(example_case, 20, "recent")

    candidates = select_candidates(history, incoming, None, "recency")

    assert any(m.memory_id == "target" for m in candidates)


def test_oracle_finds_target(example_case):
    history, incoming = build_history(example_case, 20, "middle")

    candidates = select_candidates(history, incoming, None, "oracle")

    assert candidates[0].memory_id == "target"


def test_semantic_retrieval(example_case):
    class FakeEncoder:
        def encode(self, sentences, *, normalize_embeddings):
            relevant = {
                example_case["old_memory"],
                example_case["incoming_memory"],
            }

            return np.array(
                [[1.0, 0.0] if text in relevant else [0.0, 1.0] for text in sentences]
            )

    history, incoming = build_history(example_case, 20, "old")

    candidates = select_candidates(history, incoming, FakeEncoder(), "semantic")

    assert len(candidates) == 5
    assert candidates[0].memory_id == "target"


def test_bidirectional_conflict(example_case):
    class FakeNLI:
        def predict(self, sentences, *, apply_softmax):
            return np.array([[0.90, 0.05, 0.05] for _ in sentences])

    history, incoming = build_history(example_case, 20, "old")

    candidates = select_candidates(history, incoming, None, "oracle")

    label, detected = classify_candidates(candidates, incoming, FakeNLI())

    assert label == "negative"
    assert detected is True


def test_normalization():
    assert normalize("  Hello   WORLD ") == "hello world"


def test_valid_triple():
    assert valid_triple(["i", "has_hobby", "reading"])
    assert not valid_triple(["i", "reading"])
    assert not valid_triple(["i", "", "reading"])


def test_macro_f1_perfect():
    assert macro_f1(np.eye(3)) == pytest.approx(1.0)


def test_macro_f1_zero():
    matrix = np.zeros((3, 3))
    assert macro_f1(matrix) == pytest.approx(0.0)


def test_macro_f1_batch():
    matrices = np.array(
        [
            np.eye(3),
            np.zeros((3, 3)),
        ]
    )

    np.testing.assert_allclose(macro_f1(matrices), [1.0, 0.0])
