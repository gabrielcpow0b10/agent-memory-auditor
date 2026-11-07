"""Semantic memory retrieval using normalized embeddings."""

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

import numpy as np

from agent_memory_auditor.schemas import MemoryRecord


class Encoder(Protocol):
    """Interface for an embedding model."""

    def encode(
        self, sentences: list[str], *, normalize_embeddings: bool
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class Candidate:
    """A retrieved memory with its similarity and rank."""

    memory_id: str
    score: float
    rank: int


def retrieve_top_k(
    history: tuple[MemoryRecord, ...],
    incoming: MemoryRecord,
    model: Encoder,
    top_k: int = 5,
) -> list[Candidate]:
    """Retrieve related memories ranked by cosine similarity."""

    if top_k < 1:
        raise ValueError("top_k must be positive")

    if not history:
        return []

    texts = [memory.text for memory in history]
    texts.append(incoming.text)

    embeddings = np.asarray(
        model.encode(texts, normalize_embeddings=True),
        dtype=np.float64,
    )

    expected_rows = len(texts)
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
        raise ValueError("Unexpected embedding shape")

    if embeddings.shape[1] == 0 or not np.isfinite(embeddings).all():
        raise ValueError("Embeddings must be finite and nonempty")

    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms == 0):
        raise ValueError("Embeddings must have nonzero norm")

    unit_vectors = embeddings / norms[:, None]
    scores = unit_vectors[:-1] @ unit_vectors[-1]

    if not all(isfinite(float(score)) for score in scores):
        raise ValueError("Similarity scores must be finite")

    ranked_indices = sorted(
        range(len(history)),
        key=lambda index: (-float(scores[index]), index),
    )

    candidates = []
    for rank, index in enumerate(ranked_indices[:top_k], start=1):
        candidates.append(
            Candidate(
                memory_id=history[index].memory_id,
                score=float(np.clip(scores[index], -1.0, 1.0)),
                rank=rank,
            )
        )

    return candidates
