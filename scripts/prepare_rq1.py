"""Prepare controlled Dialogue NLI retrieval experiments."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

CACHE = Path.home() / ".cache/agent-memory-auditor/dnli"
ARCHIVE = CACHE / "dialogue_nli.zip"
OUTPUT = CACHE / "rq1_design_v0_1.json"

ARCHIVE_SHA256 = "1164b0d9a0a1a6006891a6d4435a6a813464bc9b1e2f1ec5ce28c47267ad5e42"

SEED = 4304
CASES_PER_LABEL = 20
DISTRACTORS = 99
HISTORY_SIZES = [20, 50, 100]
POSITIONS = ["old", "middle", "recent"]


def normalize(text):
    return " ".join(text.casefold().split())


def read_split(archive, split):
    filename = f"dnli/dialogue_nli/dialogue_nli_{split}.jsonl"

    rows = json.loads(archive.read(filename))

    assert isinstance(rows, list)
    assert all(isinstance(row, dict) for row in rows)

    for index, row in enumerate(rows, start=1):
        if "id" not in row:
            row["id"] = f"dnli_{split}_row_{index:06d}"

    assert len({row["id"] for row in rows}) == len(rows)

    return sorted(rows, key=lambda row: row["id"])


def valid_triple(triple):
    return (
        isinstance(triple, list)
        and len(triple) == 3
        and all(isinstance(value, str) for value in triple)
        and bool(triple[1].strip())
    )


def valid_pair(row):
    return (
        bool(normalize(row["sentence1"]))
        and bool(normalize(row["sentence2"]))
        and valid_triple(row["triple1"])
        and valid_triple(row["triple2"])
    )


def main():
    assert not OUTPUT.exists()

    with ARCHIVE.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()

    assert digest == ARCHIVE_SHA256

    rng = random.Random(SEED)

    with ZipFile(ARCHIVE) as archive:
        test = read_split(archive, "verified_test")
        development = read_split(archive, "dev")

    assert len(test) == 12376
    assert len(development) == 16500

    selected = []
    used_texts = set()

    for label in ("negative", "positive", "neutral"):
        candidates = [row for row in test if row["label"] == label and valid_pair(row)]

        rng.shuffle(candidates)

        count = 0

        for row in candidates:
            first = normalize(row["sentence1"])
            second = normalize(row["sentence2"])

            if first == second:
                continue

            if first in used_texts or second in used_texts:
                continue

            selected.append(row)
            used_texts.update((first, second))

            count += 1

            if count == CASES_PER_LABEL:
                break

        assert count == CASES_PER_LABEL

    pool = []

    for row in development:
        for number in (1, 2):
            text = row[f"sentence{number}"]
            triple = row[f"triple{number}"]

            if not valid_triple(triple):
                continue

            normalized = normalize(text)

            if not normalized or normalized in used_texts:
                continue

            pool.append(
                {
                    "source_id": f"{row['id']}:s{number}",
                    "text": text,
                    "predicate": triple[1].casefold(),
                }
            )

    cases = []

    for index, row in enumerate(selected, start=1):
        predicates = {
            row["triple1"][1].casefold(),
            row["triple2"][1].casefold(),
        }

        candidates = pool.copy()
        rng.shuffle(candidates)

        distractors = []
        seen = set(used_texts)

        for candidate in candidates:
            normalized = normalize(candidate["text"])

            if normalized in seen:
                continue

            if candidate["predicate"] in predicates:
                continue

            distractors.append(candidate)
            seen.add(normalized)

            if len(distractors) == DISTRACTORS:
                break

        assert len(distractors) == DISTRACTORS

        cases.append(
            {
                "case_id": f"RQ1-{index:03d}",
                "source_id": row["id"],
                "label": row["label"],
                "old_memory": row["sentence1"],
                "incoming_memory": row["sentence2"],
                "distractors": distractors,
            }
        )

    distribution = Counter(case["label"] for case in cases)

    assert len(cases) == 60

    assert distribution == {
        "negative": 20,
        "positive": 20,
        "neutral": 20,
    }

    experiment = {
        "research_question": "RQ1",
        "source": "Dialogue NLI",
        "archive_sha256": digest,
        "target_split": "verified_test",
        "distractor_split": "dev",
        "seed": SEED,
        "history_sizes": HISTORY_SIZES,
        "positions": POSITIONS,
        "top_k": 5,
        "cases": cases,
    }

    with OUTPUT.open("x", encoding="utf-8") as file:
        json.dump(experiment, file, indent=2)
        file.write("\n")

    print("=== RQ1 EXPERIMENTAL DESIGN ===")
    print("Base cases:", len(cases))
    print("Labels:", dict(distribution))
    print("Distractors per case:", DISTRACTORS)
    print("History sizes:", HISTORY_SIZES)
    print("Positions:", POSITIONS)
    print("Controlled histories:", len(cases) * 9)
    print("SHA-256:", hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
    print("Saved:", OUTPUT)
    print("RQ1 DATASET PREPARATION: PASS")


if __name__ == "__main__":
    main()
