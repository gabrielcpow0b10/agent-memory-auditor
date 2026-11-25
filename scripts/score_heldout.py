"""Reproduce metrics from frozen held-out predictions."""

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE = Path("datasets/heldout")
RESULTS = Path("results/heldout")

GOLD = BASE / "heldout_gold_v0_1.csv"
FREEZE = BASE / "dataset_freeze_v0_1.json"

PREDICTIONS = RESULTS / "predictions_v0_1.csv"
OUTPUT = RESULTS / "metrics_v0_1.json"

LABELS = [
    "consistent",
    "new",
    "updated",
    "contradictory",
]

PREDICTION_HASH = "2719241450538dcb7815776c66f0bbbe736b03d2a8e586fc28e7260987efec33"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def evaluate(gold, predictions, ids):
    expected = [gold[i]["label"] for i in ids]

    predicted = [predictions[i]["predicted"] for i in ids]

    correct = sum(a == b for a, b in zip(expected, predicted, strict=True))

    return {
        "total": len(ids),
        "correct": correct,
        "accuracy": float(accuracy_score(expected, predicted)),
        "classification_report": classification_report(
            expected,
            predicted,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            expected,
            predicted,
            labels=LABELS,
        ).tolist(),
    }


def main():
    manifest = json.loads(FREEZE.read_text())

    for filename, expected in manifest["file_sha256"].items():
        assert sha256(Path(filename)) == expected, filename

    assert sha256(PREDICTIONS) == PREDICTION_HASH

    gold_rows = load_csv(GOLD)
    prediction_rows = load_csv(PREDICTIONS)

    gold = {row["review_id"]: row for row in gold_rows}

    predictions = {row["review_id"]: row for row in prediction_rows}

    assert len(gold_rows) == len(gold) == 200
    assert len(prediction_rows) == len(predictions) == 200
    assert set(gold) == set(predictions)

    assert all(row["policy"] == "C_consensus" for row in prediction_rows)

    assert all(row["predicted"] in LABELS for row in prediction_rows)

    assert all(row["label"] in LABELS for row in gold_rows)

    ids = sorted(gold)

    primary = evaluate(gold, predictions, ids)

    groups = {
        "explicit_updates": [
            i
            for i in ids
            if gold[i]["label"] == "updated" and gold[i]["marker_type"] == "explicit"
        ],
        "implicit_updates": [
            i
            for i in ids
            if gold[i]["label"] == "updated" and gold[i]["marker_type"] == "implicit"
        ],
        "temporal_distractors": [
            i
            for i in ids
            if gold[i]["label"] != "updated" and gold[i]["marker_type"] != "none"
        ],
    }

    assert len(groups["explicit_updates"]) == 25
    assert len(groups["implicit_updates"]) == 25
    assert len(groups["temporal_distractors"]) == 16

    subgroups = {}

    for name, group in groups.items():
        correct = sum(gold[i]["label"] == predictions[i]["predicted"] for i in group)

        subgroups[name] = {
            "total": len(group),
            "correct": correct,
            "accuracy": correct / len(group),
        }

    excluded = set(manifest["sensitivity_excluded_review_ids"])

    assert len(excluded) == 8
    assert excluded <= set(ids)

    retained = [i for i in ids if i not in excluded]

    assert len(retained) == 192

    sensitivity = evaluate(gold, predictions, retained)

    latencies = [float(row["latency_ms"]) for row in prediction_rows]

    assert all(value >= 0 for value in latencies)

    metrics = {
        "policy": "C_consensus",
        "label_order": LABELS,
        "primary_evaluation": primary,
        "temporal_subgroups": subgroups,
        "sensitivity_evaluation": sensitivity,
        "sensitivity_exclusions": sorted(excluded),
        "latency": {
            "mean_ms": mean(latencies),
            "median_ms": median(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
        },
        "prediction_sha256": sha256(PREDICTIONS),
        "gold_sha256": sha256(GOLD),
        "dataset_manifest_sha256": sha256(FREEZE),
    }

    assert primary["correct"] == 141
    assert sensitivity["correct"] == 136

    if OUTPUT.exists():
        original_hash = sha256(OUTPUT)

        archived = json.loads(OUTPUT.read_text())

        assert metrics == archived, "Metrics mismatch"

        assert sha256(OUTPUT) == original_hash

        print("Archived metrics comparison: EXACT MATCH")
        print("Original file preserved: PASS")

    else:
        with OUTPUT.open("x", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
            file.write("\n")

    print("=== HELD-OUT REPRODUCIBILITY ===")
    print("Pairs:", primary["total"])
    print("Correct:", primary["correct"])
    print("Accuracy:", primary["accuracy"])

    print(
        "Macro F1:",
        primary["classification_report"]["macro avg"]["f1-score"],
    )

    print("Sensitivity pairs:", sensitivity["total"])
    print("Metrics SHA-256:", sha256(OUTPUT))
    print("HELD-OUT REPRODUCIBILITY: PASS")


if __name__ == "__main__":
    main()
