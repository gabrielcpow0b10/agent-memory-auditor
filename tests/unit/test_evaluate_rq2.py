"""Tests for temporal policy evaluation."""

import pytest

from scripts.evaluate_rq2 import (
    bootstrap_macro_f1,
    evaluate,
    paired_mcnemar,
    resolve_baseline,
)


@pytest.mark.parametrize(
    ("label", "policy", "expected"),
    [
        ("consistent", "most_recent_wins", "consistent"),
        ("new", "most_recent_wins", "new"),
        ("contradictory", "most_recent_wins", "updated"),
        ("updated", "most_recent_wins", "updated"),
        ("consistent", "never_update", "consistent"),
        ("new", "never_update", "new"),
        ("updated", "never_update", "contradictory"),
        ("contradictory", "never_update", "contradictory"),
    ],
)
def test_temporal_baselines(label, policy, expected):
    assert resolve_baseline(label, policy) == expected


def test_invalid_policy():
    with pytest.raises(ValueError, match="Unknown temporal policy"):
        resolve_baseline("updated", "invalid")


def test_perfect_classification():
    labels = [
        "consistent",
        "new",
        "updated",
        "contradictory",
    ]

    result = evaluate(labels, labels)

    assert result["correct"] == 4
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0


def test_mcnemar_identical_predictions():
    labels = ["consistent", "new"]

    result = paired_mcnemar(labels, labels, labels)

    assert result["pipeline_only_correct"] == 0
    assert result["baseline_only_correct"] == 0
    assert result["p_value"] == 1.0


def test_mcnemar_exact():
    expected = ["updated"] * 5
    first = ["updated"] * 5
    second = ["contradictory"] * 5

    result = paired_mcnemar(expected, first, second)

    assert result["pipeline_only_correct"] == 5
    assert result["baseline_only_correct"] == 0
    assert result["p_value"] == pytest.approx(0.0625)
    assert not result["significant_at_0_05"]


def test_bootstrap_perfect_classification():
    labels = [
        "consistent",
        "new",
        "updated",
        "contradictory",
    ]

    result = bootstrap_macro_f1(labels, labels, samples=100)

    assert result["lower"] == 1.0
    assert result["upper"] == 1.0


def test_bootstrap_reproducibility():
    expected = [
        "consistent",
        "consistent",
        "new",
        "new",
        "updated",
        "updated",
        "contradictory",
        "contradictory",
    ]

    predicted = expected.copy()
    predicted[4] = "contradictory"

    first = bootstrap_macro_f1(expected, predicted, samples=100)

    second = bootstrap_macro_f1(expected, predicted, samples=100)

    assert first == second
    assert 0 <= first["lower"] <= first["upper"] <= 1
