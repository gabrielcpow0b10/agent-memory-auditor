"""Evaluate memory retrieval strategies using Dialogue NLI."""

import argparse
import csv
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import perf_counter

from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder, SentenceTransformer

from agent_memory_auditor.nli import score_bidirectional
from agent_memory_auditor.retrieval import retrieve_top_k
from agent_memory_auditor.schemas import MemoryRecord
from scripts.predict_heldout import sha256, verify_environment
from scripts.prepare_rq1 import normalize

DESIGN = Path.home() / ".cache/agent-memory-auditor/dnli/rq1_design_v0_1.json"

OUTPUT = Path("results/rq1/predictions_v0_1.csv")

METHODS = (
    "recency",
    "exhaustive",
    "semantic",
    "oracle",
)

SIZES = (20, 50, 100)
POSITIONS = ("old", "middle", "recent")
TOP_K = 5
THRESHOLD = 0.50

BASE_TIME = datetime(2020, 1, 1, tzinfo=UTC)


def validate_design():
    design = json.loads(DESIGN.read_text())

    assert design["seed"] == 4304
    assert design["history_sizes"] == list(SIZES)
    assert design["positions"] == list(POSITIONS)
    assert design["top_k"] == TOP_K

    cases = design["cases"]

    assert len(cases) == 60
    assert len({c["case_id"] for c in cases}) == 60
    assert len({c["source_id"] for c in cases}) == 60

    assert Counter(c["label"] for c in cases) == {
        "positive": 20,
        "neutral": 20,
        "negative": 20,
    }

    target_texts = {
        normalize(c[field])
        for c in cases
        for field in ("old_memory", "incoming_memory")
    }

    assert len(target_texts) == 120

    for case in cases:
        distractors = case["distractors"]

        assert len(distractors) == 99

        texts = {normalize(d["text"]) for d in distractors}

        assert len(texts) == 99
        assert not texts.intersection(target_texts)

    print("Design SHA-256:", sha256(DESIGN))
    print("Design validation: PASS")

    return cases


def build_history(case, size, position):
    records = [
        (f"d{i:03d}", d["text"])
        for i, d in enumerate(
            case["distractors"][: size - 1],
            start=1,
        )
    ]

    target_index = {
        "old": 0,
        "middle": size // 2,
        "recent": size - 1,
    }[position]

    records.insert(
        target_index,
        ("target", case["old_memory"]),
    )

    history = tuple(
        MemoryRecord(
            memory_id=memory_id,
            text=text,
            timestamp=BASE_TIME + timedelta(days=index),
        )
        for index, (memory_id, text) in enumerate(records)
    )

    incoming = MemoryRecord(
        memory_id="incoming",
        text=case["incoming_memory"],
        timestamp=BASE_TIME + timedelta(days=101),
    )

    return history, incoming


def select_candidates(history, incoming, encoder, method):
    if method == "recency":
        return history[-TOP_K:]

    if method == "exhaustive":
        return history

    if method == "oracle":
        return tuple(m for m in history if m.memory_id == "target")

    if method == "semantic":
        ranked = retrieve_top_k(
            history,
            incoming,
            encoder,
            top_k=TOP_K,
        )

        by_id = {memory.memory_id: memory for memory in history}

        return tuple(by_id[candidate.memory_id] for candidate in ranked)

    raise ValueError("Unknown retrieval method")


def classify_candidates(candidates, incoming, model):
    conflict = False
    entailment = False
    target_conflict = False

    for memory in candidates:
        scores = score_bidirectional(memory, incoming, model)

        contradiction = min(
            scores.forward.contradiction,
            scores.reverse.contradiction,
        )

        if contradiction > THRESHOLD:
            conflict = True

            if memory.memory_id == "target":
                target_conflict = True

        elif scores.forward.entailment > THRESHOLD:
            entailment = True

    if conflict:
        label = "negative"
    elif entailment:
        label = "positive"
    else:
        label = "neutral"

    return label, target_conflict


def evaluate_history(history, incoming, encoder, model, method):
    start = perf_counter()

    candidates = select_candidates(history, incoming, encoder, method)

    retrieval_ms = (perf_counter() - start) * 1000

    predicted, target_conflict = classify_candidates(candidates, incoming, model)

    elapsed_ms = (perf_counter() - start) * 1000

    return {
        "method": method,
        "predicted": predicted,
        "target_retrieved": any(m.memory_id == "target" for m in candidates),
        "target_conflict_detected": target_conflict,
        "nli_pair_comparisons": len(candidates),
        "nli_directional_inputs": 2 * len(candidates),
        "retrieval_ms": round(retrieval_ms, 3),
        "latency_ms": round(elapsed_ms, 3),
    }


def load_models(revisions):
    paths = {}

    for name, revision in revisions["models"].items():
        path = snapshot_download(
            repo_id=name,
            revision=revision,
            local_files_only=True,
        )

        assert Path(path).name == revision

        paths[name] = path

    encoder = SentenceTransformer(
        paths["sentence-transformers/all-MiniLM-L6-v2"],
        device="cpu",
        local_files_only=True,
    )

    model = CrossEncoder(
        paths["cross-encoder/nli-deberta-v3-base"],
        device="cpu",
        local_files_only=True,
    )

    assert model.model.config.id2label == {
        0: "contradiction",
        1: "entailment",
        2: "neutral",
    }

    return encoder, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    cases = validate_design()
    revisions = verify_environment()

    print("Frozen environment: PASS")

    for case in cases:
        for size in SIZES:
            for position in POSITIONS:
                history, incoming = build_history(case, size, position)

                assert len(history) == size
                assert len({m.memory_id for m in history}) == size

                assert sum(m.memory_id == "target" for m in history) == 1

                assert all(m.timestamp < incoming.timestamp for m in history)

    print("Controlled histories: 540")
    print("Method evaluations: 2160")
    print("RQ1 PREFLIGHT: PASS")

    if args.check:
        return

    partial = OUTPUT.with_suffix(".partial.csv")

    if OUTPUT.exists() or partial.exists():
        raise FileExistsError("RQ1 output already exists")

    encoder, model = load_models(revisions)

    print("Frozen models loaded: PASS", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    total = 0

    with partial.open("x", encoding="utf-8", newline="") as file:
        writer = None

        for index, case in enumerate(cases, start=1):
            for size in SIZES:
                for position in POSITIONS:
                    history, incoming = build_history(case, size, position)

                    for method in METHODS:
                        row = {
                            "case_id": case["case_id"],
                            "history_size": size,
                            "position": position,
                            "expected": case["label"],
                        }

                        row.update(
                            evaluate_history(
                                history,
                                incoming,
                                encoder,
                                model,
                                method,
                            )
                        )

                        if writer is None:
                            writer = csv.DictWriter(
                                file,
                                fieldnames=list(row),
                                lineterminator="\n",
                            )
                            writer.writeheader()

                        writer.writerow(row)
                        total += 1

            file.flush()

            print(
                f"Cases: {index}/60 | Evaluations: {total}/2160",
                flush=True,
            )

    assert total == 2160
    assert not OUTPUT.exists()

    partial.rename(OUTPUT)

    print("Saved:", OUTPUT)
    print("SHA-256:", sha256(OUTPUT))
    print("RQ1 BENCHMARK: PASS")


if __name__ == "__main__":
    main()
