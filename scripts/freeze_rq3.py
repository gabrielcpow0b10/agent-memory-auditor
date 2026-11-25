"""Freeze the local LLM configuration before held-out evaluation."""

import json
from pathlib import Path
from statistics import mean, median

from sklearn.metrics import classification_report, confusion_matrix

from scripts.evaluate_rq3 import (
    FILES,
    LABELS,
    OPTIONS,
    OUTPUT,
    PROMPT,
    REVIEW,
    SCHEMA,
    load_development,
    model_identity,
    parse_label,
    sha256,
)

FREEZE = Path("docs/rq3_freeze_v0_1.json")

HELDOUT = Path("datasets/heldout/heldout_inputs_v0_1.csv")
DATASET_FREEZE = Path("datasets/heldout/dataset_freeze_v0_1.json")

HELDOUT_OUTPUT = Path("results/rq3/heldout_v0_1.jsonl")

DEVELOPMENT_HASH = "e96fd3025f972c3ec3fc812c6a27407c915f44bc4d4a62f873f9c8b4e329f407"


def main():
    assert not FREEZE.exists()
    assert not HELDOUT_OUTPUT.exists()
    assert not HELDOUT_OUTPUT.with_suffix(".partial.jsonl").exists()

    assert sha256(OUTPUT) == DEVELOPMENT_HASH

    identity = model_identity()
    prompt_hash = sha256(PROMPT)

    inputs = load_development()

    predictions = [
        json.loads(line) for line in OUTPUT.read_text().splitlines() if line.strip()
    ]

    assert len(predictions) == 100

    assert [row["pair_id"] for row in predictions] == [row["pair_id"] for row in inputs]

    for original, prediction in zip(inputs, predictions):
        assert prediction["expected"] == original["label"]
        assert prediction["model_digest"] == identity["digest"]
        assert prediction["prompt_sha256"] == prompt_hash
        assert prediction["predicted"] in LABELS

        assert parse_label(prediction["raw_response"]) == prediction["predicted"]

    expected = [r["expected"] for r in predictions]
    predicted = [r["predicted"] for r in predictions]

    correct = sum(a == b for a, b in zip(expected, predicted))

    assert correct == 91

    report = classification_report(
        expected,
        predicted,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    latency = [float(row["latency_ms"]) for row in predictions]

    dataset = json.loads(DATASET_FREEZE.read_text())

    assert sha256(HELDOUT) == dataset["file_sha256"][str(HELDOUT)]

    files = [
        *FILES,
        REVIEW,
        PROMPT,
        OUTPUT,
        Path("scripts/evaluate_rq3.py"),
        Path("docs/annotation_guide.md"),
        HELDOUT,
        DATASET_FREEZE,
    ]

    manifest = {
        "version": "rq3-freeze-v0.1",
        "research_question": "RQ3",
        "status": "frozen_before_llm_heldout_inference",
        "model_identity": identity,
        "quantization": "Q4_K_M",
        "hardware": "Apple M4, 16 GB RAM",
        "observed_processor": "100% GPU",
        "prompt_sha256": prompt_hash,
        "prompt_selection": {
            "split": "development",
            "initial_prompt_retained": True,
            "additional_prompt_iterations": 0,
        },
        "options": OPTIONS,
        "response_schema": SCHEMA,
        "thinking": False,
        "stream": False,
        "keep_alive": "15m",
        "label_order": LABELS,
        "development": {
            "total": 100,
            "correct": correct,
            "accuracy": correct / 100,
            "macro_f1": report["macro avg"]["f1-score"],
            "invalid_responses": 0,
            "classification_report": report,
            "confusion_matrix": confusion_matrix(
                expected,
                predicted,
                labels=LABELS,
            ).tolist(),
            "mean_latency_ms": mean(latency),
            "median_latency_ms": median(latency),
        },
        "planned_heldout_pairs": 200,
        "llm_heldout_inference_executed": False,
        "nlp_heldout_results_already_available": True,
        "invalid_responses_count_as_errors": True,
        "file_sha256": {str(path): sha256(path) for path in files},
    }

    with FREEZE.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")

    print("=== RQ3 CONFIGURATION FREEZE ===")
    print("Development pairs:", len(predictions))
    print("Correct:", correct)
    print("Accuracy:", correct / 100)
    print("Macro F1:", report["macro avg"]["f1-score"])
    print("Mean latency:", mean(latency), "ms")
    print("Median latency:", median(latency), "ms")
    print("Model digest:", identity["digest"])
    print("Prompt SHA-256:", prompt_hash)
    print("Freeze SHA-256:", sha256(FREEZE))
    print("Saved:", FREEZE)
    print("RQ3 CONFIGURATION FREEZE: PASS")


if __name__ == "__main__":
    main()
