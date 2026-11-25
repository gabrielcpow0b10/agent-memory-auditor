"""Evaluate temporal resolution baselines on frozen predictions."""

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import classification_report, confusion_matrix

BASE = Path("datasets/heldout")
RESULTS = Path("results/heldout")
OUTPUT = RESULTS / "rq2_comparison_v0_1.json"

LABELS = [
    "consistent",
    "new",
    "updated",
    "contradictory",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def resolve_baseline(predicted, policy):
    if policy not in {"most_recent_wins", "never_update"}:
        raise ValueError("Unknown temporal policy")

    if predicted not in {"updated", "contradictory"}:
        return predicted

    if policy == "most_recent_wins":
        return "updated"

    return "contradictory"


def evaluate(expected, predicted):
    matrix = confusion_matrix(expected, predicted, labels=LABELS)

    report = classification_report(
        expected,
        predicted,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    return {
        "correct": int(np.trace(matrix)),
        "total": len(expected),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def paired_mcnemar(expected, first, second):
    first_correct = np.asarray(first) == expected
    second_correct = np.asarray(second) == expected

    b = int(np.sum(first_correct & ~second_correct))
    c = int(np.sum(~first_correct & second_correct))

    p_value = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0

    return {
        "pipeline_only_correct": b,
        "baseline_only_correct": c,
        "p_value": p_value,
        "significant_at_0_05": p_value < 0.05,
    }


def bootstrap_macro_f1(expected, predicted, samples=5000):
    rng = np.random.default_rng(4304)

    y = np.array([LABELS.index(v) for v in expected])
    p = np.array([LABELS.index(v) for v in predicted])

    groups = [np.flatnonzero(y == i) for i in range(len(LABELS))]

    scores = []

    for _ in range(samples):
        indices = np.concatenate(
            [rng.choice(group, len(group), replace=True) for group in groups]
        )

        matrix = np.bincount(
            4 * y[indices] + p[indices],
            minlength=16,
        ).reshape(4, 4)

        tp = np.diag(matrix)

        denominator = matrix.sum(axis=0) + matrix.sum(axis=1)

        f1 = np.divide(
            2 * tp,
            denominator,
            out=np.zeros(4),
            where=denominator != 0,
        )

        scores.append(float(f1.mean()))

    lower, upper = np.percentile(scores, [2.5, 97.5])

    return {
        "samples": samples,
        "seed": 4304,
        "method": "stratified_percentile_bootstrap",
        "confidence_level": 0.95,
        "lower": float(lower),
        "upper": float(upper),
    }


def main():
    assert not OUTPUT.exists()

    pred_path = RESULTS / "predictions_v0_1.csv"
    gold_path = BASE / "heldout_gold_v0_1.csv"
    inputs_path = BASE / "heldout_inputs_v0_1.csv"
    freeze_path = BASE / "dataset_freeze_v0_1.json"

    metrics = json.loads((RESULTS / "metrics_v0_1.json").read_text())

    manifest = json.loads(freeze_path.read_text())

    assert sha256(pred_path) == metrics["prediction_sha256"]
    assert sha256(gold_path) == metrics["gold_sha256"]
    assert sha256(freeze_path) == metrics["dataset_manifest_sha256"]

    for filename, expected_hash in manifest["file_sha256"].items():
        assert sha256(Path(filename)) == expected_hash, filename

    gold_rows = load_csv(gold_path)
    pred_rows = load_csv(pred_path)
    input_rows = load_csv(inputs_path)

    gold = {r["review_id"]: r for r in gold_rows}
    pred = {r["review_id"]: r for r in pred_rows}
    inputs = {r["review_id"]: r for r in input_rows}

    assert len(gold_rows) == len(gold) == 200
    assert len(pred_rows) == len(pred) == 200
    assert len(input_rows) == len(inputs) == 200

    assert set(gold) == set(pred) == set(inputs)

    assert all(r["policy"] == "C_consensus" for r in pred_rows)
    assert all(r["predicted"] in LABELS for r in pred_rows)
    assert all(r["label"] in LABELS for r in gold_rows)

    assert all(
        datetime.fromisoformat(r["new_timestamp"])
        > datetime.fromisoformat(r["old_timestamp"])
        for r in input_rows
    )

    ids = sorted(gold)
    expected = [gold[i]["label"] for i in ids]
    original = [pred[i]["predicted"] for i in ids]

    policies = {"C_consensus": original}

    for policy in ("most_recent_wins", "never_update"):
        policies[policy] = [resolve_baseline(label, policy) for label in original]

    results = {}

    print("=== RQ2 TEMPORAL POLICY COMPARISON ===")

    for name, predictions in policies.items():
        result = evaluate(expected, predictions)

        result["macro_f1_ci95"] = bootstrap_macro_f1(expected, predictions)

        results[name] = result

        print(f"\n{name}")
        print("Correct:", result["correct"])
        print(f"Accuracy: {result['accuracy']:.2%}")
        print(f"Macro F1: {result['macro_f1']:.4f}")

        ci = result["macro_f1_ci95"]

        print(f"Macro F1 95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")

        for label in ("updated", "contradictory"):
            recall = result["classification_report"][label]["recall"]
            print(f"{label} recall: {recall:.2%}")

        print("Confusion matrix:", result["confusion_matrix"])

    assert results["C_consensus"]["correct"] == 141

    assert np.isclose(
        results["C_consensus"]["macro_f1"],
        metrics["primary_evaluation"]["classification_report"]["macro avg"]["f1-score"],
    )

    comparisons = {}

    print("\n=== PAIRED MCNEMAR TESTS ===")

    for baseline in ("most_recent_wins", "never_update"):
        comparison = paired_mcnemar(
            expected,
            original,
            policies[baseline],
        )

        comparisons[baseline] = comparison
        print(baseline, comparison)

    output = {
        "research_question": "RQ2",
        "total_pairs": 200,
        "label_order": LABELS,
        "policies": results,
        "mcnemar": comparisons,
        "prediction_sha256": sha256(pred_path),
        "gold_sha256": sha256(gold_path),
        "frozen_policy_modified": False,
        "new_inference_executed": False,
        "baseline_definition": "shared_retrieval_and_nli_conflicts",
        "analysis_type": "planned_comparison_after_primary_evaluation",
    }

    with OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
        file.write("\n")

    print("\nSaved:", OUTPUT)
    print("RQ2 EVALUATION: PASS")


if __name__ == "__main__":
    main()
