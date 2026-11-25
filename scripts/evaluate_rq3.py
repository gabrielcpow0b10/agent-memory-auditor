"""Evaluate a local LLM memory judge on development data."""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from http.client import HTTPConnection
from pathlib import Path
from time import perf_counter

from sklearn.metrics import classification_report

MODEL = "qwen3:8b-q4_K_M"

PROMPT = Path("configs/rq3_prompt_v0_1.txt")

OUTPUT = Path("results/rq3/development_v0_1.jsonl")

REVIEW = Path("datasets/development/annotation_review_v0_1.csv")

FILES = [
    Path("datasets/pilot/annotation_pilot_v0_1.csv"),
    *[Path(f"datasets/development/batch_{i:02d}_draft.csv") for i in range(1, 5)],
]

LABELS = [
    "consistent",
    "new",
    "updated",
    "contradictory",
]

INPUT_FIELDS = (
    "old_memory",
    "new_memory",
    "old_timestamp",
    "new_timestamp",
)

OPTIONS = {
    "temperature": 0.0,
    "seed": 4304,
    "num_ctx": 2048,
    "num_predict": 64,
}

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": LABELS,
        }
    },
    "required": ["label"],
    "additionalProperties": False,
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_development():
    rows = [row for path in FILES for row in read_csv(path)]

    reviewed = read_csv(REVIEW)

    review = {row["pair_id"]: row for row in reviewed}

    assert len(rows) == 100
    assert len(reviewed) == len(review) == 100
    assert len({r["pair_id"] for r in rows}) == 100

    assert set(review) == {r["pair_id"] for r in rows}

    for row in rows:
        approved = review[row["pair_id"]]

        assert approved["review_status"] == "approved"
        assert approved["reviewed_label"] == row["label"]

        for field in ("old_memory", "new_memory"):
            assert approved[field] == row[field]

        assert all(row[field].strip() for field in INPUT_FIELDS)

    assert Counter(r["label"] for r in rows) == {label: 25 for label in LABELS}

    return sorted(rows, key=lambda row: row["pair_id"])


def api_request(path, payload=None):
    connection = HTTPConnection("127.0.0.1", 11434, timeout=180)

    try:
        if payload is None:
            connection.request("GET", path)
        else:
            connection.request(
                "POST",
                path,
                body=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )

        response = connection.getresponse()
        raw = response.read()

        if response.status != 200:
            raise RuntimeError(f"Ollama HTTP {response.status}: {raw!r}")

        return json.loads(raw)

    finally:
        connection.close()


def model_identity():
    models = api_request("/api/tags")["models"]

    matches = [item for item in models if item["name"] == MODEL]

    assert len(matches) == 1

    return {
        "model": MODEL,
        "digest": matches[0]["digest"],
        "ollama_version": api_request("/api/version")["version"],
    }


def build_payload(row, prompt):
    memories = {field: row[field] for field in INPUT_FIELDS}

    return {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": json.dumps(memories),
            },
        ],
        "stream": False,
        "think": False,
        "format": SCHEMA,
        "options": OPTIONS,
        "keep_alive": "15m",
    }


def parse_label(content):
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return "invalid"

    if not isinstance(result, dict):
        return "invalid"

    if set(result) != {"label"}:
        return "invalid"

    if result["label"] not in LABELS:
        return "invalid"

    return result["label"]


def predict(row, prompt):
    payload = build_payload(row, prompt)

    start = perf_counter()

    response = api_request("/api/chat", payload)

    latency = (perf_counter() - start) * 1000

    assert response["done"] is True

    content = response["message"]["content"]

    predicted = parse_label(content)

    if response.get("done_reason") == "length":
        predicted = "invalid"

    return {
        "pair_id": row["pair_id"],
        "predicted": predicted,
        "raw_response": content,
        "done_reason": response.get("done_reason"),
        "latency_ms": round(latency, 3),
        "load_duration_ms": (response["load_duration"] / 1_000_000),
        "total_duration_ms": (response["total_duration"] / 1_000_000),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "eval_count": response.get("eval_count"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    rows = load_development()
    prompt = PROMPT.read_text(encoding="utf-8")
    identity = model_identity()

    print("Development validation: PASS")
    print("Development pairs:", len(rows))
    print("Model:", json.dumps(identity))
    print("Prompt SHA-256:", sha256(PROMPT))

    if not args.smoke:
        assert not OUTPUT.exists()

        partial = OUTPUT.with_suffix(".partial.jsonl")
        assert not partial.exists()

    warmup = predict(rows[0], prompt)

    print("Smoke prediction:", warmup["predicted"])
    print("Expected:", rows[0]["label"])
    print("Raw response:", warmup["raw_response"])

    assert warmup["predicted"] in LABELS

    print("RQ3 API SMOKE TEST: PASS", flush=True)

    if args.smoke:
        return

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    outputs = []

    with partial.open("x", encoding="utf-8") as file:
        for index, row in enumerate(rows, start=1):
            result = predict(row, prompt)

            result["expected"] = row["label"]

            result["model_digest"] = identity["digest"]

            result["prompt_sha256"] = sha256(PROMPT)

            outputs.append(result)

            file.write(json.dumps(result) + "\n")
            file.flush()
            os.fsync(file.fileno())

            if index % 10 == 0:
                print(
                    f"Development: {index}/100",
                    flush=True,
                )

    assert len(outputs) == 100
    assert model_identity() == identity

    partial.rename(OUTPUT)

    expected = [row["expected"] for row in outputs]
    predicted = [row["predicted"] for row in outputs]

    print("\n=== DEVELOPMENT RESULTS ===")

    print(
        classification_report(
            expected,
            predicted,
            labels=LABELS,
            zero_division=0,
        )
    )

    print(
        "Correct:",
        sum(a == b for a, b in zip(expected, predicted)),
    )

    print("Invalid responses:", predicted.count("invalid"))

    print("Saved:", OUTPUT)
    print("SHA-256:", sha256(OUTPUT))
    print("RQ3 DEVELOPMENT: PASS")


if __name__ == "__main__":
    main()
