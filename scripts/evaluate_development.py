"""Evaluate four policies on the full development candidate set."""

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

from compare_policies import POLICIES, classify
from sentence_transformers import CrossEncoder, SentenceTransformer

from agent_memory_auditor.nli import score_bidirectional
from agent_memory_auditor.retrieval import retrieve_top_k
from agent_memory_auditor.schemas import MemoryRecord
from agent_memory_auditor.temporal import detect_temporal_change

FILES = [
    Path("datasets/pilot/annotation_pilot_v0_1.csv"),
    Path("datasets/development/batch_01_draft.csv"),
    Path("datasets/development/batch_02_draft.csv"),
    Path("datasets/development/batch_03_draft.csv"),
    Path("datasets/development/batch_04_draft.csv"),
]

BASELINE = Path("results/pilot/baseline_v0_1_predictions.csv")
OUTPUT = Path("results/development/policy_comparison_candidates_v0_1.csv")


def main():
    assert not OUTPUT.exists(), "Evaluation output already exists."

    data = []

    for path in FILES:
        with path.open(newline="", encoding="utf-8") as file:
            data.extend(csv.DictReader(file))

    assert len(data) == 100
    assert len({row["pair_id"] for row in data}) == 100
    assert Counter(row["label"] for row in data) == {
        "consistent": 25,
        "new": 25,
        "updated": 25,
        "contradictory": 25,
    }

    with BASELINE.open(newline="", encoding="utf-8") as file:
        archived = {row["pair_id"]: row["predicted"] for row in csv.DictReader(file)}

    assert len(archived) == 20

    print("=== LOADING LOCAL NLP MODELS ===", flush=True)

    encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    model = CrossEncoder(
        "cross-encoder/nli-deberta-v3-base",
        device="cpu",
    )

    outputs = []

    for index, row in enumerate(data, start=1):
        old = MemoryRecord(
            memory_id=row["pair_id"] + "-old",
            text=row["old_memory"],
            timestamp=datetime.fromisoformat(row["old_timestamp"]),
        )

        new = MemoryRecord(
            memory_id=row["pair_id"] + "-new",
            text=row["new_memory"],
            timestamp=datetime.fromisoformat(row["new_timestamp"]),
        )

        candidate = retrieve_top_k((old,), new, encoder)[0]

        nli = score_bidirectional(old, new, model)
        temporal = detect_temporal_change(old, new)

        for name, (threshold, consensus) in POLICIES.items():
            predicted = classify(
                candidate.score,
                nli,
                temporal,
                threshold,
                consensus,
            )

            if name == "A_baseline" and row["pair_id"] in archived:
                assert predicted == archived[row["pair_id"]], (
                    row["pair_id"],
                    predicted,
                    archived[row["pair_id"]],
                )

            outputs.append(
                {
                    "policy": name,
                    "pair_id": row["pair_id"],
                    "source": row["source"],
                    "expected": row["label"],
                    "predicted": predicted,
                    "correct": int(predicted == row["label"]),
                    "similarity": round(candidate.score, 6),
                    "forward_contradiction": round(nli.forward.contradiction, 6),
                    "reverse_contradiction": round(nli.reverse.contradiction, 6),
                    "temporal_candidate": temporal.candidate_update,
                }
            )

        if index % 10 == 0:
            print(f"Processed {index}/100 pairs", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(outputs[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(outputs)

    print("\n=== DEVELOPMENT POLICY COMPARISON ===")

    assert len(outputs) == 400

    for name in POLICIES:
        selected = [row for row in outputs if row["policy"] == name]

        correct = sum(row["correct"] for row in selected)
        errors = [row["pair_id"] for row in selected if not row["correct"]]

        print(f"\n{name}: {correct}/100")
        print("Errors:", errors)

        for label in ("consistent", "new", "updated", "contradictory"):
            group = [row for row in selected if row["expected"] == label]
            matches = sum(row["correct"] for row in group)
            print(f"  {label}: {matches}/25")

    print("\nArchived 20-pair baseline preserved.")
    print("Saved:", OUTPUT)
    print("Status: exploratory development analysis.")
    print("NOT held-out benchmark accuracy.")


if __name__ == "__main__":
    main()
