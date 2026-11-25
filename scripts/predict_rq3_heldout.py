"""Evaluate the frozen local LLM on held-out memory pairs."""

import argparse
import json
import os
from pathlib import Path

from scripts.evaluate_rq3 import (
    INPUT_FIELDS,
    LABELS,
    OPTIONS,
    PROMPT,
    SCHEMA,
    load_development,
    model_identity,
    predict,
    read_csv,
    sha256,
)

FREEZE = Path("docs/rq3_freeze_v0_1.json")

INPUT = Path("datasets/heldout/heldout_inputs_v0_1.csv")

OUTPUT = Path("results/rq3/heldout_v0_1.jsonl")

FREEZE_HASH = "e4aa509b21496eb589abbb1091f2f5812faf67e51353f1f840fdb3c3efe56b6e"


def verify_configuration():
    assert sha256(FREEZE) == FREEZE_HASH

    manifest = json.loads(FREEZE.read_text())

    assert manifest["status"] == ("frozen_before_llm_heldout_inference")

    assert manifest["llm_heldout_inference_executed"] is False
    assert manifest["planned_heldout_pairs"] == 200
    assert manifest["label_order"] == LABELS
    assert manifest["options"] == OPTIONS
    assert manifest["response_schema"] == SCHEMA
    assert manifest["thinking"] is False
    assert manifest["stream"] is False
    assert manifest["keep_alive"] == "15m"

    for filename, expected in manifest["file_sha256"].items():
        assert sha256(Path(filename)) == expected, filename

    assert sha256(PROMPT) == manifest["prompt_sha256"]

    identity = model_identity()

    assert identity == manifest["model_identity"]

    return manifest


def load_heldout():
    rows = read_csv(INPUT)

    assert len(rows) == 200

    assert {row["review_id"] for row in rows} == {f"R{i:03d}" for i in range(1, 201)}

    allowed = {"review_id", *INPUT_FIELDS}

    for row in rows:
        assert set(row) == allowed

        assert all(row[field].strip() for field in INPUT_FIELDS)

    return sorted(
        rows,
        key=lambda row: row["review_id"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    manifest = verify_configuration()
    rows = load_heldout()

    partial = OUTPUT.with_suffix(".partial.jsonl")

    assert not OUTPUT.exists()
    assert not partial.exists()

    print("Frozen configuration: PASS")
    print("Model identity: PASS")
    print("Input validation: PASS")
    print("Held-out pairs:", len(rows))
    print("Gold labels supplied to LLM: NO")
    print("RQ3 HELD-OUT PREFLIGHT: PASS", flush=True)

    if args.check:
        return

    prompt = PROMPT.read_text(encoding="utf-8")

    # Warm up using development, never held-out.
    warmup = predict(load_development()[0], prompt)

    assert warmup["predicted"] in LABELS

    print("Development warm-up: PASS", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    invalid = 0
    total = 0

    with partial.open("x", encoding="utf-8") as file:
        for index, row in enumerate(rows, start=1):
            inference_input = {
                "pair_id": row["review_id"],
                **{field: row[field] for field in INPUT_FIELDS},
            }

            result = predict(inference_input, prompt)

            assert result.pop("pair_id") == row["review_id"]

            result["review_id"] = row["review_id"]

            result["model_digest"] = manifest["model_identity"]["digest"]

            result["prompt_sha256"] = manifest["prompt_sha256"]

            result["freeze_sha256"] = FREEZE_HASH

            assert result["predicted"] in [*LABELS, "invalid"]

            if result["predicted"] == "invalid":
                invalid += 1

            file.write(json.dumps(result) + "\n")
            file.flush()
            os.fsync(file.fileno())

            total += 1

            if index % 10 == 0:
                print(
                    f"Held-out: {index}/200",
                    flush=True,
                )

    assert total == 200

    verify_configuration()

    assert not OUTPUT.exists()

    partial.rename(OUTPUT)

    print("\n=== RQ3 HELD-OUT INFERENCE ===")
    print("Predictions:", total)
    print("Invalid responses:", invalid)
    print("Saved:", OUTPUT)
    print("SHA-256:", sha256(OUTPUT))
    print("RQ3 HELD-OUT INFERENCE: PASS")


if __name__ == "__main__":
    main()
