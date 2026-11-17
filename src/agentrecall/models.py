"""Pydantic domain models shared across memory, curation, agent, and eval layers."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    """Timezone-aware UTC now, used for every stored timestamp."""
    return datetime.now(UTC)


def new_id() -> str:
    """Return a fresh UUID string."""
    return str(uuid.uuid4())


class MemoryTier(StrEnum):
    """The three tiers of the memory hierarchy."""

    SCRATCH = "scratch"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class EntryStatus(StrEnum):
    """Lifecycle state of a semantic memory entry."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETIRED = "retired"
    STALE = "stale"


class SourceDoc(BaseModel):
    """An externally-sourced document that memories may be grounded in."""

    id: str
    uri: str = ""
    title: str = ""


    body: str = ""
    content_hash: str = ""
    updated_at: datetime = Field(default_factory=utcnow)
    tags: list[str] = Field(default_factory=list)


class MemoryRecord(BaseModel):
    """A semantic memory: one subject/claim pair with provenance and versioning."""

    id: str = Field(default_factory=new_id)
    tier: MemoryTier = MemoryTier.SEMANTIC
    subject: str
    claim: str
    body: str = ""
    embedding: list[float] | None = None
    source_doc_id: str | None = None
    session_id: str | None = None
    version: int = 1
    status: EntryStatus = EntryStatus.ACTIVE
    supersedes: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.5
    access_count: int = 0
    created_at: datetime = Field(default_factory=utcnow)
    last_used_at: datetime = Field(default_factory=utcnow)
    content_hash: str = ""


class ScoredMemory(BaseModel):
    """A semantic hit together with its score decomposition."""

    record: MemoryRecord
    similarity: float
    recency: float
    usage: float
    decay: float
    score: float
    tier: MemoryTier = MemoryTier.SEMANTIC


class EpisodicSummary(BaseModel):
    """A compact summary of a turn span from one session."""

    id: str = Field(default_factory=new_id)
    session_id: str
    start_turn: int
    end_turn: int
    summary: str
    embedding: list[float] | None = None
    score: float = 0.0
    created_at: datetime = Field(default_factory=utcnow)


class ScratchpadEntry(BaseModel):
    """One recent conversational turn held in Redis."""

    session_id: str
    role: str
    text: str
    created_at: datetime = Field(default_factory=utcnow)


class CurationDecision(BaseModel):
    """Outcome of running a candidate through the write-path pipeline."""

    action: str
    target_id: str | None = None
    rival_id: str | None = None
    reason: str = ""
    similarity: float = 0.0
    score: float = 0.0
    entry: MemoryRecord | None = None


class TraceStep(BaseModel):
    """One node execution inside the LangGraph resolution workflow."""

    node: str
    detail: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class ResolutionTrace(BaseModel):
    """Full audit trail of one intent-to-action resolution."""

    id: str = Field(default_factory=new_id)
    session_id: str
    request: str
    intent: str
    chosen_tool: str | None = None
    memory_hits: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[TraceStep] = Field(default_factory=list)
    outcome: str = "unknown"
    latency_ms: int = 0
    created_at: datetime = Field(default_factory=utcnow)


class AuditFinding(BaseModel):
    """One problem discovered by the knowledge-repository auditor."""

    kind: str
    severity: str = "warning"
    entry_id: str | None = None
    detail: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class MemoryHit(BaseModel):
    """A unified hit returned by ``MemoryStore.recall``."""

    tier: MemoryTier
    id: str
    text: str
    score: float
    provenance: dict[str, Any] = Field(default_factory=dict)
