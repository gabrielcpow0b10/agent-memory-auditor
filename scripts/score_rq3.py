"""Compare frozen NLP and local LLM held-out predictions."""

import json
from math import isfinite
from pathlib import Path
from statistics import mean, median

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from scripts.evaluate_rq2 import paired_mcnemar
from scripts.evaluate_rq3 import LABELS, parse_label, read_csv, sha256
from scripts.predict_rq3_heldout import FREEZE, FREEZE_HASH, OUTPUT

BASE = Path("datasets/heldout")
NLP = Path("results/heldout/predictions_v0_1.csv")
NLP_METRICS = Path("results/heldout/metrics_v0_1.json")
GOLD = BASE / "heldout_gold_v0_1.csv"
DATASET = BASE / "dataset_freeze_v0_1.json"

RESULT = Path("results/rq3/metrics_v0_1.json")

LLM_HASH = "e885855d7dfe547cfdd20cf5372a94141da894bd1f8b31a11b6830c52e9358d1"

NLP_HASH = "2719241450538dcb7815776c66f0bbbe736b03d2a8e586fc28e7260987efec33"

SEED = 4304
SAMPLES = 5000
PREDICTION_LABELS = [*LABELS, "invalid"]


def evaluate(expected, predicted, latencies):
    report = classification_report(
        expected,
        predicted,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    correct = sum(a == b for a, b in zip(expected, predicted))

    matrix = confusion_matrix(
        expected,
        predicted,
        labels=PREDICTION_LABELS,
    )

    return {
        "total": len(expected),
        "correct": correct,
        "accuracy": correct / len(expected),
        "macro_f1": report["macro avg"]["f1-score"],
        "invalid_responses": predicted.count("invalid"),
        "classification_report": report,
        "confusion_matrix": matrix[:4].tolist(),
        "mean_latency_ms": mean(latencies),
        "median_latency_ms": median(latencies),
    }


def bootstrap_indices(expected):
    rng = np.random.default_rng(SEED)
    labels = np.asarray(expected)

    groups = [np.flatnonzero(labels == label) for label in LABELS]

    assert all(len(group) > 0 for group in groups)

    return np.concatenate(
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


def bootstrap_scores(expected, predicted, indices):
    y = np.array([LABELS.index(v) for v in expected])

    p = np.array([PREDICTION_LABELS.index(v) for v in predicted])

    encoded = 5 * y[indices] + p[indices]

    offsets = 20 * np.arange(len(indices))[:, None]

    matrices = np.bincount(
        (encoded + offsets).ravel(),
        minlength=20 * len(indices),
    ).reshape(-1, 4, 5)

    tp = np.diagonal(
        matrices[:, :, :4],
        axis1=1,
        axis2=2,
    )

    denominator = matrices.sum(axis=2) + matrices.sum(axis=1)[:, :4]

    scores = np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=float),
        where=denominator != 0,
    )

    return scores.mean(axis=1)


def confidence_interval(values):
    lower, upper = np.percentile(values, [2.5, 97.5])

    return {
        "lower": float(lower),
        "upper": float(upper),
    }


def main():
    assert not RESULT.exists()

    assert sha256(FREEZE) == FREEZE_HASH
    assert sha256(OUTPUT) == LLM_HASH
    assert sha256(NLP) == NLP_HASH

    freeze = json.loads(FREEZE.read_text())
    dataset = json.loads(DATASET.read_text())
    previous = json.loads(NLP_METRICS.read_text())

    for manifest in (freeze, dataset):
        for filename, digest in manifest["file_sha256"].items():
            assert sha256(Path(filename)) == digest, filename

    assert sha256(GOLD) == previous["gold_sha256"]
    assert sha256(NLP) == previous["prediction_sha256"]
    assert sha256(DATASET) == previous["dataset_manifest_sha256"]

    gold_rows = read_csv(GOLD)
    nlp_rows = read_csv(NLP)

    llm_rows = [
        json.loads(line) for line in OUTPUT.read_text().splitlines() if line.strip()
    ]

    gold = {r["review_id"]: r for r in gold_rows}
    nlp = {r["review_id"]: r for r in nlp_rows}
    llm = {r["review_id"]: r for r in llm_rows}

    assert len(gold_rows) == len(gold) == 200
    assert len(nlp_rows) == len(nlp) == 200
    assert len(llm_rows) == len(llm) == 200

    assert set(gold) == set(nlp) == set(llm)

    ids = sorted(gold)

    for review_id in ids:
        row = llm[review_id]

        assert row["freeze_sha256"] == FREEZE_HASH
        assert row["prompt_sha256"] == freeze["prompt_sha256"]
        assert row["model_digest"] == (freeze["model_identity"]["digest"])

        parsed = parse_label(row["raw_response"])

        if row["done_reason"] == "length":
            parsed = "invalid"

        assert parsed == row["predicted"]
        assert row["predicted"] in PREDICTION_LABELS

        assert nlp[review_id]["policy"] == "C_consensus"
        assert nlp[review_id]["predicted"] in LABELS
        assert gold[review_id]["label"] in LABELS

        for record in (row, nlp[review_id]):
            latency = float(record["latency_ms"])
            assert isfinite(latency) and latency >= 0

    expected = [gold[i]["label"] for i in ids]
    nlp_pred = [nlp[i]["predicted"] for i in ids]
    llm_pred = [llm[i]["predicted"] for i in ids]

    assert all(expected.count(label) == 50 for label in LABELS)

    nlp_latency = [float(nlp[i]["latency_ms"]) for i in ids]

    llm_latency = [float(llm[i]["latency_ms"]) for i in ids]

    pipeline = evaluate(expected, nlp_pred, nlp_latency)
    judge = evaluate(expected, llm_pred, llm_latency)

    assert pipeline["correct"] == 141
    assert judge["invalid_responses"] == 0

    assert np.isclose(
        pipeline["macro_f1"],
        previous["primary_evaluation"]["classification_report"]["macro avg"][
            "f1-score"
        ],
    )

    indices = bootstrap_indices(expected)

    nlp_samples = bootstrap_scores(expected, nlp_pred, indices)

    llm_samples = bootstrap_scores(expected, llm_pred, indices)

    pipeline["macro_f1_ci95"] = confidence_interval(nlp_samples)

    judge["macro_f1_ci95"] = confidence_interval(llm_samples)

    delta = {
        "macro_f1_difference": (judge["macro_f1"] - pipeline["macro_f1"]),
        "paired_ci95": confidence_interval(llm_samples - nlp_samples),
        "accuracy_difference": (judge["accuracy"] - pipeline["accuracy"]),
    }

    mcnemar = paired_mcnemar(expected, nlp_pred, llm_pred)

    mcnemar["llm_only_correct"] = mcnemar.pop("baseline_only_correct")

    latency_ratio = judge["mean_latency_ms"] / pipeline["mean_latency_ms"]

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
        subgroups[name] = {
            "total": len(group),
            "pipeline_correct": sum(
                gold[i]["label"] == nlp[i]["predicted"] for i in group
            ),
            "llm_correct": sum(gold[i]["label"] == llm[i]["predicted"] for i in group),
        }

    excluded = set(dataset["sensitivity_excluded_review_ids"])

    assert len(excluded) == 8
    assert excluded <= set(ids)

    retained = [i for i in ids if i not in excluded]

    assert len(retained) == 192

    sensitivity = {}

    for name, records in (
        ("C_consensus", nlp),
        ("Qwen3_8B", llm),
    ):
        sensitivity[name] = evaluate(
            [gold[i]["label"] for i in retained],
            [records[i]["predicted"] for i in retained],
            [float(records[i]["latency_ms"]) for i in retained],
        )

    output = {
        "research_question": "RQ3",
        "heldout_pairs": 200,
        "label_order": LABELS,
        "confusion_matrix_columns": PREDICTION_LABELS,
        "pipeline": pipeline,
        "llm": judge,
        "llm_minus_pipeline": delta,
        "mcnemar": mcnemar,
        "bootstrap": {
            "method": "paired_stratified_percentile",
            "seed": SEED,
            "samples": SAMPLES,
            "confidence_level": 0.95,
        },
        "llm_to_pipeline_latency_ratio": latency_ratio,
        "temporal_subgroups": subgroups,
        "sensitivity_evaluation": sensitivity,
        "sensitivity_exclusions": sorted(excluded),
        "model_identity": freeze["model_identity"],
        "prediction_sha256": sha256(OUTPUT),
        "pipeline_prediction_sha256": sha256(NLP),
        "gold_sha256": sha256(GOLD),
        "freeze_sha256": sha256(FREEZE),
        "limitations": [
            "Pipeline held-out results preceded LLM setup.",
            "Pipeline used CPU; LLM was observed using GPU.",
            "Latency includes local LLM API overhead.",
            "Latency is not a direct energy or monetary cost.",
            "Results are limited to a synthetic corpus.",
        ],
    }

    print("=== RQ3 FINAL RESULTS ===")

    for name, result in (
        ("C_consensus", pipeline),
        ("Qwen3_8B", judge),
    ):
        print(f"\n{name}")
        print("Correct:", result["correct"])
        print(f"Accuracy: {result['accuracy']:.2%}")
        print(f"Macro F1: {result['macro_f1']:.4f}")
        print("Macro F1 CI95:", result["macro_f1_ci95"])
        print("Mean latency:", result["mean_latency_ms"], "ms")
        print("Median latency:", result["median_latency_ms"], "ms")

        for label in LABELS:
            recall = result["classification_report"][label]["recall"]
            print(f"{label} recall: {recall:.2%}")

        print("Confusion matrix:", result["confusion_matrix"])

    print("\n=== STATISTICAL COMPARISON ===")
    print("LLM minus pipeline:", delta)
    print("McNemar:", mcnemar)
    print(f"LLM latency ratio: {latency_ratio:.2f}x")

    print("\n=== TEMPORAL SUBGROUPS ===")

    for name, result in subgroups.items():
        print(name, result)

    print("\n=== SENSITIVITY EVALUATION ===")

    for name, result in sensitivity.items():
        print(
            name,
            f"Accuracy={result['accuracy']:.2%}",
            f"F1={result['macro_f1']:.4f}",
        )

    with RESULT.open("x", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
        file.write("\n")

    print("\nSaved:", RESULT)
    print("RQ3 FINAL EVALUATION: PASS")


if __name__ == "__main__":
    main()
