"""Run semantic retrieval on synthetic memory examples."""

from datetime import UTC, datetime

from sentence_transformers import SentenceTransformer

from agent_memory_auditor.retrieval import retrieve_top_k
from agent_memory_auditor.schemas import MemoryRecord


def memory(memory_id: str, text: str) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        text=text,
        timestamp=datetime(2026, 11, 7, tzinfo=UTC),
    )


def main() -> None:
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_id, device="cpu")

    history = (
        memory("m001", "I live in Miami."),
        memory("m002", "I have two dogs."),
        memory("m003", "I enjoy playing basketball."),
        memory("m004", "My favorite food is pizza."),
        memory("m005", "I work as a software developer."),
        memory("m006", "I recently moved to Boston."),
    )

    incoming = memory("new001", "I currently reside in Miami.")

    candidates = retrieve_top_k(
        history=history,
        incoming=incoming,
        model=model,
        top_k=5,
    )

    print(f"Model: {model_id}")
    print(f"Incoming: {incoming.text}")
    print()

    for candidate in candidates:
        matched = next(
            item for item in history
            if item.memory_id == candidate.memory_id
        )
        print(
            f"Rank {candidate.rank}: "
            f"{candidate.memory_id} | "
            f"score={candidate.score:.4f} | "
            f"{matched.text}"
        )


if __name__ == "__main__":
    main()
