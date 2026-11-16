"""Stage 2 baseline for four-label memory auditing."""

from dataclasses import dataclass
from math import isfinite

from agent_memory_auditor.nli import NLIModel, score_bidirectional
from agent_memory_auditor.retrieval import Candidate, Encoder, retrieve_top_k
from agent_memory_auditor.schemas import AuditRequest, AuditResult, MemoryLabel
from agent_memory_auditor.temporal import detect_temporal_change


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Provisional thresholds; tune only on development data."""

    top_k: int = 5
    similarity_threshold: float = 0.35
    contradiction_threshold: float = 0.50
    entailment_threshold: float = 0.50

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if not isfinite(self.similarity_threshold) or not (
            -1.0 <= self.similarity_threshold <= 1.0
        ):
            raise ValueError("similarity_threshold must be finite and in [-1, 1]")
        for name in ("contradiction_threshold", "entailment_threshold"):
            value = getattr(self, name)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")


def audit_memory(
    request: AuditRequest,
    encoder: Encoder,
    nli_model: NLIModel,
    *,
    config: PipelineConfig | None = None,
) -> AuditResult:
    """Apply the original Stage 2 mapping; never mutate stored memories."""

    settings = config if config is not None else PipelineConfig()
    candidates = retrieve_top_k(
        request.history,
        request.incoming,
        encoder,
        top_k=settings.top_k,
    )
    history_by_id = {memory.memory_id: memory for memory in request.history}
    best_consistent: Candidate | None = None

    for candidate in candidates:
        if candidate.score <= settings.similarity_threshold:
            break  # Ranked by decreasing similarity.

        old = history_by_id[candidate.memory_id]
        nli = score_bidirectional(old, request.incoming, nli_model)
        contradiction = max(
            nli.forward.contradiction,
            nli.reverse.contradiction,
        )

        if contradiction > settings.contradiction_threshold:
            temporal = detect_temporal_change(old, request.incoming)
            return AuditResult(
                label=(
                    MemoryLabel.UPDATED
                    if temporal.candidate_update
                    else MemoryLabel.CONTRADICTORY
                ),
                incoming_id=request.incoming.memory_id,
                evidence_id=old.memory_id,
                similarity=candidate.score,
                change_signals=temporal.matched_markers,
            )

        if (
            nli.forward.entailment > settings.entailment_threshold
            and best_consistent is None
        ):
            best_consistent = candidate

    if best_consistent is not None:
        return AuditResult(
            label=MemoryLabel.CONSISTENT,
            incoming_id=request.incoming.memory_id,
            evidence_id=best_consistent.memory_id,
            similarity=best_consistent.score,
        )

    return AuditResult(
        label=MemoryLabel.NEW,
        incoming_id=request.incoming.memory_id,
    )
