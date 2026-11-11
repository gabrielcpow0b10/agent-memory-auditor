"""Demonstrate bidirectional NLI on synthetic memories."""

from datetime import UTC, datetime

from sentence_transformers import CrossEncoder

from agent_memory_auditor.nli import score_bidirectional
from agent_memory_auditor.schemas import MemoryRecord


def memory(memory_id: str, text: str) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        text=text,
        timestamp=datetime(2026, 11, 11, tzinfo=UTC),
    )


def main() -> None:
    model_id = "cross-encoder/nli-deberta-v3-base"
    model = CrossEncoder(model_id, device="cpu")

    examples = [
        ("I live in Miami.", "I reside in Miami."),
        ("I do not drink coffee.", "I drink coffee every day."),
        ("I have a dog.", "I enjoy playing chess."),
    ]

    for index, (old_text, new_text) in enumerate(examples, 1):
        old = memory(f"old-{index}", old_text)
        new = memory(f"new-{index}", new_text)

        result = score_bidirectional(old, new, model)

        print(f"\n=== PAIR {index} ===")
        print("OLD:", old.text)
        print("NEW:", new.text)

        for name, scores in [
            ("FORWARD", result.forward),
            ("REVERSE", result.reverse),
        ]:
            print(
                f"{name}: {scores.predicted_label.value} | "
                f"C={scores.contradiction:.4f} "
                f"E={scores.entailment:.4f} "
                f"N={scores.neutral:.4f}"
            )


if __name__ == "__main__":
    main()
