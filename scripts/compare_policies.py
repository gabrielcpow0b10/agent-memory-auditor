"""Compare exploratory policies on the development pilot only."""

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer

from agent_memory_auditor.nli import score_bidirectional
from agent_memory_auditor.retrieval import retrieve_top_k
from agent_memory_auditor.schemas import MemoryRecord
from agent_memory_auditor.temporal import detect_temporal_change

DATASET = Path("datasets/pilot/annotation_pilot_v0_1.csv")
BASELINE = Path("results/pilot/baseline_v0_1_predictions.csv")
OUTPUT = Path("results/pilot/policy_comparison_v0_1.csv")

POLICIES = {
    "A_baseline": (0.35, False),
    "B_retrieval": (0.25, False),
    "C_consensus": (0.35, True),
    "D_combined": (0.25, True),
}


def classify(similarity, nli, temporal, threshold, consensus):
    if similarity <= threshold:
        return "new"

    forward = nli.forward.contradiction
    reverse = nli.reverse.contradiction

    contradiction = min(forward, reverse) if consensus else max(forward, reverse)

    if contradiction > 0.50:
        return "updated" if temporal.candidate_update else "contradictory"

    if nli.forward.entailment > 0.50:
        return "consistent"

    return "new"


def main():
    encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    nli_model = CrossEncoder(
        "cross-encoder/nli-deberta-v3-base",
        device="cpu",
    )

    with DATASET.open(newline="", encoding="utf-8") as file:
        data = list(csv.DictReader(file))

    with BASELINE.open(newline="", encoding="utf-8") as file:
        archived = {row["pair_id"]: row["predicted"] for row in csv.DictReader(file)}

    outputs = []

    for row in data:
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

        nli = score_bidirectional(old, new, nli_model)
        temporal = detect_temporal_change(old, new)

        for name, (threshold, consensus) in POLICIES.items():
            predicted = classify(
                candidate.score,
                nli,
                temporal,
                threshold,
                consensus,
            )

            if name == "A_baseline":
                assert predicted == archived[row["pair_id"]], (
                    row["pair_id"],
                    predicted,
                    archived[row["pair_id"]],
                )

            outputs.append(
                {
                    "policy": name,
                    "pair_id": row["pair_id"],
                    "expected": row["label"],
                    "predicted": predicted,
                    "correct": int(predicted == row["label"]),
                    "similarity": round(candidate.score, 6),
                    "forward_contradiction": round(nli.forward.contradiction, 6),
                    "reverse_contradiction": round(nli.reverse.contradiction, 6),
                }
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(outputs[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(outputs)

    print("\n=== DEVELOPMENT PILOT COMPARISON ===")

    for name in POLICIES:
        selected = [row for row in outputs if row["policy"] == name]

        correct = sum(row["correct"] for row in selected)
        errors = [row["pair_id"] for row in selected if not row["correct"]]

        print(f"\n{name}: {correct}/20")
        print("Errors:", errors or "None")

        by_label = Counter(row["expected"] for row in selected if row["correct"])
        print("Correct by label:", dict(by_label))

    print("\nBaseline reproduced successfully.")
    print("Results saved:", OUTPUT)
    print("Exploratory only — not held-out evaluation.")


if __name__ == "__main__":
    main()
