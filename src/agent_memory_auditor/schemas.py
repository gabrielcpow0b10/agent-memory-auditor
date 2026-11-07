"""Validated data contracts for Agent Memory Auditor."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import isfinite


class MemoryLabel(str, Enum):
    """The four labels defined in the research protocol."""

    CONSISTENT = "consistent"
    NEW = "new"
    UPDATED = "updated"
    CONTRADICTORY = "contradictory"


class MarkerType(str, Enum):
    """Types of temporal change markers."""

    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """A stored memory with an identifier and timestamp."""

    memory_id: str
    text: str
    timestamp: datetime
    source: str = "synthetic"

    def __post_init__(self) -> None:
        if not self.memory_id.strip():
            raise ValueError("memory_id cannot be empty")
        if not self.text.strip():
            raise ValueError("memory text cannot be empty")
        if not self.source.strip():
            raise ValueError("source cannot be empty")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must include timezone information")


@dataclass(frozen=True, slots=True)
class AuditRequest:
    """A memory history and an incoming memory."""

    history: tuple[MemoryRecord, ...]
    incoming: MemoryRecord

    def __post_init__(self) -> None:
        identifiers = [memory.memory_id for memory in self.history]
        identifiers.append(self.incoming.memory_id)

        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Memory identifiers must be unique")


@dataclass(frozen=True, slots=True)
class AuditResult:
    """A read-only classification and supporting evidence."""

    label: MemoryLabel
    incoming_id: str
    evidence_id: str | None = None
    similarity: float | None = None
    change_signals: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.incoming_id.strip():
            raise ValueError("incoming_id cannot be empty")

        if self.label != MemoryLabel.NEW and self.evidence_id is None:
            raise ValueError("Non-new labels require supporting evidence")

        if self.similarity is not None:
            if not isfinite(self.similarity):
                raise ValueError("similarity must be finite")
            if not -1.0 <= self.similarity <= 1.0:
                raise ValueError("similarity must be between -1 and 1")
