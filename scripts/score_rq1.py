"""Score the completed Dialogue NLI retrieval benchmark."""

import csv
import json
from math import isfinite
from pathlib import Path
from statistics import mean, median

from sklearn.metrics import classification_report, confusion_matrix

from scripts.evaluate_rq1 import (
    DESIGN,
    METHODS,
    OUTPUT,
    POSITIONS,
    SIZES,
    validate_design,
)
from scripts.predict_heldout import sha256

METRICS = Path("results/rq1/metrics_v0_1.json")
LABELS = ["positive", "neutral", "negative"]

PREDICTION_HASH = "10cc50eebefb1c8cf6b3ef71285e8acb06c278ff9f19a2460dc07157f948a88f"

DESIGN_HASH = "a117f5bdc42ff6de8f2f19034c5e9e3de284df35ba11b2194527d4daa2750efc"


def evaluate(rows):
    expected = [r["expected"] for r in rows]
    predicted = [r["predicted"] for r in rows]

    negatives = [r for r in rows if r["expected"] == "negative"]

    compatible = [r for r in rows if r["expected"] != "negative"]

    report = classification_report(
        expected,
        predicted,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    retrieved = sum(r["target_retrieved"] == "True" for r in negatives)

    detected = sum(r["target_conflict_detected"] == "True" for r in negatives)

    latency = [float(r["latency_ms"]) for r in rows]

    comparisons = [int(r["nli_pair_comparisons"]) for r in rows]

    return {
        "evaluations": len(rows),
        "correct": sum(a == b for a, b in zip(expected, predicted)),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "negative_histories": len(negatives),
        "conflict_targets_retrieved": retrieved,
        "conflict_targets_detected": detected,
        "conflict_retrieval_recall": retrieved / len(negatives),
        "target_conflict_recall": detected / len(negatives),
        "all_target_retrieval_rate": sum(r["target_retrieved"] == "True" for r in rows)
        / len(rows),
        "compatible_false_alarm_rate": sum(
            r["predicted"] == "negative" for r in compatible
        )
        / len(compatible),
        "nli_pair_comparisons": sum(comparisons),
        "mean_nli_comparisons": mean(comparisons),
        "nli_directional_inputs": 2 * sum(comparisons),
        "mean_retrieval_ms": mean(float(r["retrieval_ms"]) for r in rows),
        "mean_latency_ms": mean(latency),
        "median_latency_ms": median(latency),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            expected, predicted, labels=LABELS
        ).tolist(),
    }


def main():
    assert not METRICS.exists()
    assert sha256(OUTPUT) == PREDICTION_HASH
    assert sha256(DESIGN) == DESIGN_HASH

    cases = validate_design()

    with OUTPUT.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file, strict=True))

    expected_keys = [
        (
            case["case_id"],
            str(size),
            position,
            case["label"],
            method,
        )
        for case in cases
        for size in SIZES
        for position in POSITIONS
        for method in METHODS
    ]

    actual_keys = [
        (
            row["case_id"],
            row["history_size"],
            row["position"],
            row["expected"],
            row["method"],
        )
        for row in rows
    ]

    assert len(rows) == 2160
    assert actual_keys == expected_keys
    assert len(set(actual_keys)) == 2160

    for row in rows:
        assert None not in row
        assert all(v is not None for v in row.values())
        assert row["predicted"] in LABELS

        for key in (
            "target_retrieved",
            "target_conflict_detected",
        ):
            assert row[key] in {"True", "False"}

        count = {
            "recency": 5,
            "exhaustive": int(row["history_size"]),
            "semantic": 5,
            "oracle": 1,
        }[row["method"]]

        assert int(row["nli_pair_comparisons"]) == count
        assert int(row["nli_directional_inputs"]) == 2 * count

        retrieval = float(row["retrieval_ms"])
        latency = float(row["latency_ms"])

        assert isfinite(retrieval)
        assert isfinite(latency)
        assert 0 <= retrieval <= latency

        if row["method"] in {"exhaustive", "oracle"}:
            assert row["target_retrieved"] == "True"

        if row["method"] == "recency":
            assert (row["target_retrieved"] == "True") == (row["position"] == "recent")

        if row["target_conflict_detected"] == "True":
            assert row["target_retrieved"] == "True"
            assert row["predicted"] == "negative"

    total = sum(int(r["nli_pair_comparisons"]) for r in rows)

    assert total == 36540

    print("Final prediction validation: PASS")
    print("Evaluations:", len(rows))
    print("NLI pair comparisons:", total)

    overall = {}
    by_size = {}
    by_position = {}
    by_condition = {}

    print("\n=== RQ1 OVERALL RESULTS ===")

    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]

        assert len(subset) == 540

        result = evaluate(subset)
        overall[method] = result

        print(f"\n{method.upper()}")
        print(f"Accuracy: {result['accuracy']:.2%}")
        print(f"Macro F1: {result['macro_f1']:.4f}")
        print(
            "Conflict retrieval recall:",
            f"{result['conflict_retrieval_recall']:.2%}",
        )
        print(
            "Target conflict recall:",
            f"{result['target_conflict_recall']:.2%}",
        )
        print(
            "Compatible false alarm rate:",
            f"{result['compatible_false_alarm_rate']:.2%}",
        )
        print(
            "NLI comparisons:",
            result["nli_pair_comparisons"],
        )
        print(
            "Mean latency:",
            f"{result['mean_latency_ms']:.3f} ms",
        )
        print(
            "Median latency:",
            f"{result['median_latency_ms']:.3f} ms",
        )
        print("Confusion matrix:", result["confusion_matrix"])

    print("\n=== RESULTS BY HISTORY SIZE ===")

    for size in SIZES:
        by_size[str(size)] = {}

        print(f"\nHistory size: {size}")

        for method in METHODS:
            subset = [
                r
                for r in rows
                if r["method"] == method and int(r["history_size"]) == size
            ]

            result = evaluate(subset)
            by_size[str(size)][method] = result

            print(
                method,
                f"Recall={result['conflict_retrieval_recall']:.2%}",
                f"F1={result['macro_f1']:.4f}",
                f"Latency={result['mean_latency_ms']:.2f} ms",
            )

    print("\n=== RESULTS BY TARGET POSITION ===")

    for position in POSITIONS:
        by_position[position] = {}

        print(f"\nPosition: {position}")

        for method in METHODS:
            subset = [
                r for r in rows if r["method"] == method and r["position"] == position
            ]

            result = evaluate(subset)
            by_position[position][method] = result

            print(
                method,
                f"Retrieval={result['conflict_retrieval_recall']:.2%}",
                f"Detection={result['target_conflict_recall']:.2%}",
            )

    for size in SIZES:
        for position in POSITIONS:
            key = f"{size}_{position}"
            by_condition[key] = {}

            for method in METHODS:
                subset = [
                    r
                    for r in rows
                    if r["method"] == method
                    and int(r["history_size"]) == size
                    and r["position"] == position
                ]

                assert len(subset) == 60

                by_condition[key][method] = evaluate(subset)

    metrics = {
        "research_question": "RQ1",
        "dataset": "Dialogue NLI",
        "base_cases": 60,
        "controlled_histories": 540,
        "method_evaluations": 2160,
        "label_order": LABELS,
        "prediction_sha256": sha256(OUTPUT),
        "design_sha256": sha256(DESIGN),
        "overall": overall,
        "by_history_size": by_size,
        "by_position": by_position,
        "by_condition": by_condition,
        "execution": {
            "device": "cpu",
            "sessions": 2,
            "recovered_cases": 31,
            "resumed_cases": 29,
            "model_loading_included": False,
        },
        "limitations": [
            "Histories share 60 underlying source cases.",
            "Distractor neutrality is not independently verified.",
            "History labels are inherited from target DNLI pairs.",
            "Method execution order was fixed.",
            "Inference resumed after an SSH interruption.",
            "Three-class retrieval experiment, not four-label evaluation.",
        ],
    }

    with METRICS.open("x", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")

    print("\nSaved:", METRICS)
    print("RQ1 METRICS: PASS")


if __name__ == "__main__":
    main()
