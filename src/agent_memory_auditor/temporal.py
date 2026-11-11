"""Rule-based temporal evidence detection for memory auditing."""

import re
from dataclasses import dataclass

from agent_memory_auditor.schemas import MarkerType, MemoryRecord

_EXPLICIT_PATTERNS = {
    "no longer": re.compile(r"\bno\s+longer\b", re.IGNORECASE),
    "moved to": re.compile(r"\bmoved\s+to\b", re.IGNORECASE),
    "switched to": re.compile(r"\bswitched\b(?:\s+\w+){0,8}\s+to\b", re.IGNORECASE),
    "now": re.compile(r"\bnow\b", re.IGNORECASE),
    "last month": re.compile(r"\blast\s+month\b", re.IGNORECASE),
}

_EXPLICIT_CHANGE = frozenset(
    {
        "no longer",
        "moved to",
        "switched to",
    }
)

_IMPLICIT_PATTERNS = {
    "started": re.compile(r"\bstarted\b", re.IGNORECASE),
    "began": re.compile(r"\bbegan\b", re.IGNORECASE),
}


@dataclass(frozen=True, slots=True)
class TemporalEvidence:
    """Traceable temporal signals for a memory pair."""

    old_memory_id: str
    new_memory_id: str
    marker_type: MarkerType
    matched_markers: tuple[str, ...]
    later_timestamp: bool
    candidate_update: bool


def detect_temporal_change(
    old: MemoryRecord,
    new: MemoryRecord,
) -> TemporalEvidence:
    """Find temporal cues without deciding the final memory label."""

    explicit = tuple(
        name for name, pattern in _EXPLICIT_PATTERNS.items() if pattern.search(new.text)
    )

    implicit = tuple(
        name for name, pattern in _IMPLICIT_PATTERNS.items() if pattern.search(new.text)
    )

    if explicit:
        marker_type = MarkerType.EXPLICIT
    elif implicit:
        marker_type = MarkerType.IMPLICIT
    else:
        marker_type = MarkerType.NONE

    later = new.timestamp > old.timestamp

    has_change_verb = any(name in _EXPLICIT_CHANGE for name in explicit) or bool(
        implicit
    )

    return TemporalEvidence(
        old_memory_id=old.memory_id,
        new_memory_id=new.memory_id,
        marker_type=marker_type,
        matched_markers=explicit + implicit,
        later_timestamp=later,
        candidate_update=later and has_change_verb,
    )
