"""Paired cluster bootstrap for the RQ1 retrieval benchmark."""

import csv
import json
from pathlib import Path

import numpy as np

from scripts.score_rq1 import (
    DESIGN,
    DESIGN_HASH,
    LABELS,
    OUTPUT,
    PREDICTION_HASH,
    sha256,
)

RESULT = Path("results/rq1/statistics_v0_1.json")

SEED = 4304
SAMPLES = 5000


def macro_f1(matrix):
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)

    denominator = matrix.sum(axis=-1) + matrix.sum(axis=-2)

    scores = np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=float),
        where=denominator != 0,
    )

    return scores.mean(axis=-1)


def interval(observed, samples):
    lower, upper = np.percentile(samples, [2.5, 97.5])

    return {
        "difference": float(observed),
        "lower": float(lower),
        "upper": float(upper),
    }


def main():
    assert not RESULT.exists()
    assert sha256(OUTPUT) == PREDICTION_HASH
    assert sha256(DESIGN) == DESIGN_HASH

    with OUTPUT.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2160

    methods = ["recency", "exhaustive", "semantic", "oracle"]

    case_ids = sorted({row["case_id"] for row in rows})
    assert len(case_ids) == 60

    case_index = {case: i for i, case in enumerate(case_ids)}

    labels = {}

    matrices = {method: np.zeros((60, 3, 3), dtype=int) for method in methods}

    retrieved = {method: np.zeros(60) for method in methods}

    detected = {method: np.zeros(60) for method in methods}

    keys = set()

    for row in rows:
        case = row["case_id"]
        i = case_index[case]
        method = row["method"]

        key = (case, row["history_size"], row["position"], method)

        assert key not in keys
        keys.add(key)

        expected = row["expected"]

        if case in labels:
            assert labels[case] == expected

        labels[case] = expected

        actual = LABELS.index(expected)
        predicted = LABELS.index(row["predicted"])

        matrices[method][i, actual, predicted] += 1

        if expected == "negative":
            retrieved[method][i] += (row["target_retrieved"] == "True") / 9

            detected[method][i] += (row["target_conflict_detected"] == "True") / 9

    for method in methods:
        assert np.all(matrices[method].sum(axis=(1, 2)) == 9)

    groups = [
        np.array([case_index[case] for case in case_ids if labels[case] == label])
        for label in LABELS
    ]

    assert all(len(group) == 20 for group in groups)

    rng = np.random.default_rng(SEED)

    indices = np.concatenate(
        [
            rng.choice(
                group,
                size=(SAMPLES, len(group)),
                replace=True,
            )
            for group in groups
        ],
        axis=1,
    )

    observed = {}
    samples = {}

    for method in methods:
        matrix = matrices[method].sum(axis=0)

        resampled = matrices[method][indices].sum(axis=1)

        observed[method] = {
            "macro_f1": macro_f1(matrix),
            "conflict_retrieval_recall": retrieved[method].sum() / 20,
            "target_conflict_recall": detected[method].sum() / 20,
        }

        samples[method] = {
            "macro_f1": macro_f1(resampled),
            "conflict_retrieval_recall": retrieved[method][indices].sum(axis=1) / 20,
            "target_conflict_recall": detected[method][indices].sum(axis=1) / 20,
        }

    comparisons = {}

    print("=== RQ1 PAIRED CLUSTER BOOTSTRAP ===")

    for baseline in ("recency", "exhaustive"):
        comparison = {}

        print(f"\nSemantic minus {baseline}")

        for metric in observed["semantic"]:
            difference = observed["semantic"][metric] - observed[baseline][metric]

            distribution = samples["semantic"][metric] - samples[baseline][metric]

            result = interval(difference, distribution)

            comparison[metric] = result

            print(
                metric,
                f"delta={result['difference']:.4f}",
                f"CI95=[{result['lower']:.4f}, {result['upper']:.4f}]",
            )

        comparisons[baseline] = comparison

    output = {
        "research_question": "RQ1",
        "method": "paired_stratified_cluster_bootstrap",
        "confidence_level": 0.95,
        "samples": SAMPLES,
        "seed": SEED,
        "base_cases": 60,
        "negative_base_cases": 20,
        "histories_per_case": 9,
        "prediction_sha256": sha256(OUTPUT),
        "design_sha256": sha256(DESIGN),
        "comparisons": comparisons,
        "analysis_type": "exploratory",
    }

    with RESULT.open("x", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
        file.write("\n")

    print("\nSaved:", RESULT)
    print("RQ1 STATISTICAL ANALYSIS: PASS")


if __name__ == "__main__":
    main()
