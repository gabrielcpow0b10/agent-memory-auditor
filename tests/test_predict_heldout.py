"""Unit tests for frozen held-out inference."""

import csv

import numpy as np
import pytest

from scripts.predict_heldout import load_inputs, predict_pair


class FakeEncoder:
    def __init__(self, related=True):
        self.related = related

    def encode(self, sentences, *, normalize_embeddings):
        if self.related:
            return np.array([[1.0, 0.0], [1.0, 0.0]])
        return np.array([[1.0, 0.0], [0.0, 1.0]])


class FakeNLI:
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def predict(self, sentences, *, apply_softmax):
        return np.array(self.probabilities)


def make_row(index=1):
    return {
        "review_id": f"R{index:03d}",
        "old_memory": "I use a bicycle.",
        "new_memory": "I use a bicycle.",
        "old_timestamp": "2026-01-01T12:00:00+00:00",
        "new_timestamp": "2026-10-01T12:00:00+00:00",
    }


def write_inputs(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_load_valid_inputs(tmp_path):
    path = tmp_path / "inputs.csv"
    rows = [make_row(i) for i in range(1, 201)]
    write_inputs(path, rows)

    assert load_inputs(path) == rows


def test_reject_gold_labels(tmp_path):
    path = tmp_path / "inputs.csv"
    rows = [{**make_row(i), "label": "consistent"} for i in range(1, 201)]
    write_inputs(path, rows)

    with pytest.raises(ValueError):
        load_inputs(path)


def test_reject_duplicate_ids(tmp_path):
    path = tmp_path / "inputs.csv"
    rows = [make_row(i) for i in range(1, 201)]
    rows[-1]["review_id"] = "R001"
    write_inputs(path, rows)

    with pytest.raises(ValueError):
        load_inputs(path)


def test_reject_missing_pair(tmp_path):
    path = tmp_path / "inputs.csv"
    rows = [make_row(i) for i in range(1, 200)]
    write_inputs(path, rows)

    with pytest.raises(ValueError):
        load_inputs(path)


def test_reject_empty_memory(tmp_path):
    path = tmp_path / "inputs.csv"
    rows = [make_row(i) for i in range(1, 201)]
    rows[0]["new_memory"] = ""
    write_inputs(path, rows)

    with pytest.raises(ValueError):
        load_inputs(path)


@pytest.mark.parametrize(
    "related,new_text,probabilities,expected",
    [
        (
            True,
            "I use a bicycle.",
            [[0.05, 0.90, 0.05], [0.05, 0.90, 0.05]],
            "consistent",
        ),
        (
            False,
            "I collect stamps.",
            [[0.05, 0.05, 0.90], [0.05, 0.05, 0.90]],
            "new",
        ),
        (
            True,
            "I switched to a scooter.",
            [[0.90, 0.05, 0.05], [0.90, 0.05, 0.05]],
            "updated",
        ),
        (
            True,
            "I do not use a bicycle.",
            [[0.90, 0.05, 0.05], [0.90, 0.05, 0.05]],
            "contradictory",
        ),
        (
            True,
            "I also use a scooter.",
            [[0.90, 0.05, 0.05], [0.05, 0.05, 0.90]],
            "new",
        ),
    ],
)
def test_frozen_policy_predictions(related, new_text, probabilities, expected):
    row = make_row()
    row["new_memory"] = new_text

    result = predict_pair(
        row,
        FakeEncoder(related),
        FakeNLI(probabilities),
    )

    assert result["review_id"] == "R001"
    assert result["policy"] == "C_consensus"
    assert result["predicted"] == expected
    assert result["latency_ms"] >= 0
    assert "expected" not in result
    assert "label" not in result
    assert "correct" not in result
