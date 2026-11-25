"""Tests for the local LLM judge and statistical evaluation."""

import json

import numpy as np
import pytest

from scripts import evaluate_rq3
from scripts.evaluate_rq3 import (
    INPUT_FIELDS,
    LABELS,
    build_payload,
    parse_label,
)
from scripts.score_rq3 import (
    bootstrap_indices,
    bootstrap_scores,
    evaluate,
)


@pytest.mark.parametrize("label", LABELS)
def test_valid_label(label):
    content = json.dumps({"label": label})
    assert parse_label(content) == label


@pytest.mark.parametrize(
    "content",
    [
        "",
        "consistent",
        "[]",
        "{}",
        '{"label": "unknown"}',
        '{"label": "new", "extra": true}',
    ],
)
def test_invalid_label(content):
    assert parse_label(content) == "invalid"


@pytest.fixture
def example_pair():
    return {
        "pair_id": "TEST001",
        "old_memory": "I live in Miami.",
        "new_memory": "I reside in Miami.",
        "old_timestamp": "2025-01-01",
        "new_timestamp": "2025-02-01",
        "label": "consistent",
        "review_notes": "Test annotation",
    }


def test_payload_excludes_gold_labels(example_pair):
    payload = build_payload(example_pair, "Classify the memory pair.")

    supplied = json.loads(payload["messages"][1]["content"])

    assert set(supplied) == set(INPUT_FIELDS)
    assert "label" not in supplied
    assert "pair_id" not in supplied
    assert "review_notes" not in supplied

    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.0


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("stop", "consistent"),
        ("length", "invalid"),
    ],
)
def test_predict(monkeypatch, example_pair, reason, expected):
    def fake_api(path, payload):
        assert path == "/api/chat"

        return {
            "done": True,
            "done_reason": reason,
            "message": {"content": '{"label": "consistent"}'},
            "load_duration": 1000000,
            "total_duration": 10000000,
            "prompt_eval_count": 100,
            "eval_count": 8,
        }

    monkeypatch.setattr(evaluate_rq3, "api_request", fake_api)

    result = evaluate_rq3.predict(example_pair, "Classify the pair.")

    assert result["predicted"] == expected
    assert result["pair_id"] == "TEST001"
    assert result["latency_ms"] >= 0


def test_invalid_prediction_counts_as_error():
    expected = LABELS.copy()
    predicted = LABELS.copy()

    predicted[-1] = "invalid"

    result = evaluate(expected, predicted, [10.0] * 4)

    assert result["correct"] == 3
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["macro_f1"] == pytest.approx(0.75)
    assert result["invalid_responses"] == 1
    assert result["confusion_matrix"][-1][-1] == 1


def test_bootstrap_perfect_classification():
    expected = LABELS * 2

    indices = bootstrap_indices(expected)

    scores = bootstrap_scores(expected, expected, indices)

    np.testing.assert_allclose(scores, 1.0)


def test_bootstrap_reproducibility():
    expected = LABELS * 2

    first = bootstrap_indices(expected)
    second = bootstrap_indices(expected)

    np.testing.assert_array_equal(first, second)

    labels = np.asarray(expected)

    for label in LABELS:
        assert np.all((labels[first] == label).sum(axis=1) == 2)
