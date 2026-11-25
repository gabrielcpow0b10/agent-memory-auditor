"""Analyze frozen held-out classification errors."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

BASE = Path("datasets/heldout")
RESULTS = Path("results/heldout")
OUTPUT = RESULTS / "error_diagnostics_v0_1.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def main():
    metrics = json.loads((RESULTS / "metrics_v0_1.json").read_text())

    pred_path = RESULTS / "predictions_v0_1.csv"
    gold_path = BASE / "heldout_gold_v0_1.csv"
    freeze_path = BASE / "dataset_freeze_v0_1.json"

    assert sha256(pred_path) == metrics["prediction_sha256"]
    assert sha256(gold_path) == metrics["gold_sha256"]
    assert sha256(freeze_path) == metrics["dataset_manifest_sha256"]

    gold_rows = load_csv(gold_path)
    pred_rows = load_csv(pred_path)

    gold = {r["review_id"]: r for r in gold_rows}
    pred = {r["review_id"]: r for r in pred_rows}

    assert len(gold_rows) == len(gold) == 200
    assert len(pred_rows) == len(pred) == 200
    assert set(gold) == set(pred)

    errors = [i for i in sorted(gold) if gold[i]["label"] != pred[i]["predicted"]]

    updates = [i for i in sorted(gold) if gold[i]["label"] == "updated"]

    temporal_detected = [i for i in updates if pred[i]["temporal_candidate"] == "True"]

    distribution = Counter(
        gold[i]["label"] + "_to_" + pred[i]["predicted"] for i in errors
    )

    causes = Counter()

    for i in errors:
        if gold[i]["label"] != "updated":
            continue

        row = pred[i]

        temporal = row["temporal_candidate"] == "True"

        consensus = (
            min(
                float(row["forward_contradiction"]),
                float(row["reverse_contradiction"]),
            )
            > 0.50
        )

        if float(row["similarity"]) <= 0.35:
            causes["retrieval_gate"] += 1
        elif not temporal and consensus:
            causes["temporal_gate"] += 1
        elif not temporal and not consensus:
            causes["temporal_and_nli"] += 1
        elif temporal and not consensus:
            causes["nli_consensus"] += 1
        else:
            causes["other"] += 1

    retrieval_rejected_errors = sum(
        float(pred[i]["similarity"]) <= 0.35 for i in errors
    )

    diagnostics = {
        "policy": "C_consensus",
        "total_pairs": len(gold),
        "total_errors": len(errors),
        "error_distribution": dict(distribution),
        "update_error_causes": dict(causes),
        "actual_updates": len(updates),
        "temporal_candidates_detected": len(temporal_detected),
        "temporal_candidates_missed": (len(updates) - len(temporal_detected)),
        "temporal_candidate_recall": (len(temporal_detected) / len(updates)),
        "retrieval_rejected_errors": retrieval_rejected_errors,
        "error_review_ids": errors,
        "prediction_sha256": sha256(pred_path),
        "gold_sha256": sha256(gold_path),
        "diagnostic_only": True,
    }

    assert len(errors) == 59
    assert len(updates) == 50
    assert len(temporal_detected) == 8
    assert sum(causes.values()) == 45

    with OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(diagnostics, file, indent=2)
        file.write("\n")

    print("=== FINAL ERROR DIAGNOSTICS ===")
    print("Total errors:", len(errors))
    print("Update error causes:", dict(causes))
    print("Temporal candidates:", len(temporal_detected))
    print("Retrieval-rejected errors:", retrieval_rejected_errors)
    print("Saved:", OUTPUT)
    print("ERROR DIAGNOSTICS: PASS")


if __name__ == "__main__":
    main()
