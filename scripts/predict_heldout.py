"""Generate held-out predictions using the frozen policy."""

import csv
import hashlib
import json
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder, SentenceTransformer

from agent_memory_auditor.nli import score_bidirectional
from agent_memory_auditor.retrieval import retrieve_top_k
from agent_memory_auditor.schemas import MemoryRecord
from agent_memory_auditor.temporal import detect_temporal_change
from scripts.compare_policies import POLICIES, classify

BASE = Path("datasets/heldout")
INPUT = BASE / "heldout_inputs_v0_1.csv"
DATASET_FREEZE = BASE / "dataset_freeze_v0_1.json"

POLICY_FREEZE = Path("docs/policy_freeze_v0_1.json")
MODEL_FREEZE = Path("docs/model_revisions_v0_1.json")

OUTPUT = Path("results/heldout/predictions_v0_1.csv")

POLICY = "C_consensus"

INPUT_FIELDS = {
    "review_id",
    "old_memory",
    "new_memory",
    "old_timestamp",
    "new_timestamp",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(path):
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if set(reader.fieldnames or []) != INPUT_FIELDS or len(
            reader.fieldnames or []
        ) != len(INPUT_FIELDS):
            raise ValueError("Unexpected inference input fields")

        rows = list(reader)

    expected_ids = {f"R{i:03d}" for i in range(1, 201)}

    if len(rows) != 200 or {row["review_id"] for row in rows} != expected_ids:
        raise ValueError("Invalid held-out review IDs")

    if any(
        not row.get(field) or not row[field].strip()
        for row in rows
        for field in INPUT_FIELDS
    ):
        raise ValueError("Missing inference input")

    return rows


def verify_environment():
    dataset = json.loads(DATASET_FREEZE.read_text())
    policy = json.loads(POLICY_FREEZE.read_text())
    models = json.loads(MODEL_FREEZE.read_text())

    assert dataset["selected_policy"] == POLICY
    assert policy["selected_policy"] == POLICY
    assert models["selected_policy"] == POLICY
    assert policy["held_out_used_for_selection"] is False

    assert POLICIES[POLICY] == (0.35, True)
    assert models["device"] == "cpu"
    assert models["local_files_only"] is True

    assert sha256(INPUT) == dataset["file_sha256"][str(INPUT)]

    assert sha256(POLICY_FREEZE) == dataset["file_sha256"][str(POLICY_FREEZE)]

    for filename, expected in policy["file_sha256"].items():
        assert sha256(Path(filename)) == expected, filename

    for package, expected in models["package_versions"].items():
        assert version(package) == expected, package

    assert sys.version.split()[0] == models["python_version"]

    assert models["development_reproduction"]["archived_prediction_matches"] == 100

    return models


def predict_pair(row, encoder, model):
    start = perf_counter()

    review_id = row["review_id"]

    old = MemoryRecord(
        memory_id=review_id + "-old",
        text=row["old_memory"],
        timestamp=datetime.fromisoformat(row["old_timestamp"]),
    )

    new = MemoryRecord(
        memory_id=review_id + "-new",
        text=row["new_memory"],
        timestamp=datetime.fromisoformat(row["new_timestamp"]),
    )

    candidate = retrieve_top_k((old,), new, encoder)[0]
    nli = score_bidirectional(old, new, model)
    temporal = detect_temporal_change(old, new)

    predicted = classify(
        candidate.score,
        nli,
        temporal,
        *POLICIES[POLICY],
    )

    return {
        "review_id": review_id,
        "policy": POLICY,
        "predicted": predicted,
        "similarity": round(candidate.score, 6),
        "forward_contradiction": round(nli.forward.contradiction, 6),
        "forward_entailment": round(nli.forward.entailment, 6),
        "forward_neutral": round(nli.forward.neutral, 6),
        "reverse_contradiction": round(nli.reverse.contradiction, 6),
        "reverse_entailment": round(nli.reverse.entailment, 6),
        "reverse_neutral": round(nli.reverse.neutral, 6),
        "temporal_marker": temporal.marker_type.value,
        "temporal_candidate": temporal.candidate_update,
        "matched_markers": "|".join(temporal.matched_markers),
        "latency_ms": round((perf_counter() - start) * 1000, 3),
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    revisions = verify_environment()
    rows = load_inputs(INPUT)

    print("Environment verification: PASS", flush=True)
    print("Inference input validation: PASS", flush=True)

    paths = {
        name: snapshot_download(
            repo_id=name,
            revision=revision,
            local_files_only=True,
        )
        for name, revision in revisions["models"].items()
    }

    for name, revision in revisions["models"].items():
        assert Path(paths[name]).name == revision

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

    outputs = []

    for index, row in enumerate(rows, start=1):
        outputs.append(predict_pair(row, encoder, model))

        if index % 10 == 0:
            print(f"Processed {index}/200", flush=True)

    assert len(outputs) == 200
    assert len({r["review_id"] for r in outputs}) == 200

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(outputs[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(outputs)

    print("\nPredictions:", len(outputs))
    print("Saved:", OUTPUT)
    print("Prediction SHA-256:", sha256(OUTPUT))
    print("HELD-OUT INFERENCE: PASS")


if __name__ == "__main__":
    main()
