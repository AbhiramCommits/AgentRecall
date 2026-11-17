"""Semantic memory: pgvector candidate retrieval with Python-side score re-ranking.

The SQL query only narrows candidates by cosine distance and tier/status filters. All
recency, usage, and decay arithmetic lives in :mod:`agentrecall.memory.scoring`-style pure
code here so the weights and formulas are directly testable with fixed timestamps.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any

from langchain_core.embeddings import Embeddings
from sqlalchemy import text
from sqlalchemy.engine import Engine

from agentrecall.config import Settings, get_settings
from agentrecall.db import get_engine
from agentrecall.embeddings import get_embedder, vector_literal
from agentrecall.models import (
    EntryStatus,
    MemoryRecord,
    MemoryTier,
    ScoredMemory,
    utcnow,
)

_INACTIVE = (EntryStatus.SUPERSEDED.value, EntryStatus.RETIRED.value)


def recency_component(
    last_used_at: datetime, now: datetime, half_life_days: float
) -> float:
    """Exponential recency in [0, 1]: ``exp(-ln2 * age / half_life)``."""
    age_days = max(0.0, (now - last_used_at).total_seconds() / 86400.0)
    return math.exp(-math.log(2.0) * age_days / half_life_days)


def usage_component(access_count: int, access_count_max: int) -> float:
    """Log-scaled usage in [0, 1], normalized by the most-used entry."""
    if access_count_max <= 0:
        return 0.0
    return math.log1p(max(0, access_count)) / math.log1p(access_count_max)


def decay_factor(
    last_used_at: datetime, now: datetime, stale_after_days: float, decay_floor: float
) -> float:
    """Full weight until the stale horizon, then the configured floor."""
    age_days = max(0.0, (now - last_used_at).total_seconds() / 86400.0)
    return decay_floor if age_days > stale_after_days else 1.0


def score_entry(
    similarity: float,
    last_used_at: datetime,
    access_count: int,
    access_count_max: int,
    now: datetime,
    settings: Settings,
) -> tuple[float, float, float, float, float]:
    """Return ``(final, similarity, recency, usage, decay)`` for one candidate."""
    recency = recency_component(last_used_at, now, settings.recency_half_life_days)
    usage = usage_component(access_count, access_count_max)
    decay = decay_factor(last_used_at, now, settings.stale_after_days, settings.decay_floor)
    raw = (
        settings.w_similarity * similarity
        + settings.w_recency * recency
        + settings.w_usage * usage
    )
    return raw * decay, similarity, recency, usage, decay


class SemanticMemory:
    """Long-term memory entries backed by ``memory_entries`` and pgvector."""

    def __init__(
        self,
        engine: Engine | None = None,
        embedder: Embeddings | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.engine = engine or get_engine(self.settings)
        self.embedder = embedder or get_embedder()

    # ------------------------------------------------------------------ writes
    def write(self, record: MemoryRecord) -> str:
        """Insert a raw memory row (curation happens upstream in Step 3)."""
        vector = record.embedding or self.embedder.embed_query(
            record.body or f"{record.subject} {record.claim}"
        )
        record.embedding = vector
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO memory_entries
                        (id, tier, subject, claim, body, embedding, source_doc_id, session_id,
                         version, status, supersedes, provenance, confidence, access_count,
                         created_at, last_used_at, content_hash)
                    VALUES
                        (CAST(:id AS uuid), :tier, :subject, :claim, :body,
                         CAST(:embedding AS vector),
                         :source_doc_id, :session_id, :version, :status, CAST(:supersedes AS uuid),
                         CAST(:provenance AS jsonb), :confidence, :access_count, :created_at,
                         :last_used_at, :content_hash)
                    """
                ),
                {
                    "id": record.id,
                    "tier": record.tier.value,
                    "subject": record.subject,
                    "claim": record.claim,
                    "body": record.body,
                    "embedding": vector_literal(vector),
                    "source_doc_id": record.source_doc_id,
                    "session_id": record.session_id,
                    "version": record.version,
                    "status": record.status.value,
                    "supersedes": record.supersedes,
                    "provenance": json.dumps(record.provenance, default=str),
                    "confidence": record.confidence,
                    "access_count": record.access_count,
                    "created_at": record.created_at,
                    "last_used_at": record.last_used_at,
                    "content_hash": record.content_hash,
                },
            )
        return record.id

    def touch(self, ids: list[str], now: datetime | None = None) -> int:
        """Increment usage and refresh ``last_used_at`` for the given ids."""
        unique = sorted({str(i) for i in ids})
        if not unique:
            return 0
        now = now or utcnow()
        with self.engine.begin() as conn:
            result = conn.execute(
                text(
                    """
                    UPDATE memory_entries
                    SET access_count = access_count + 1, last_used_at = :now
                    WHERE id = ANY(CAST(:ids AS uuid[]))
                    """
                ),
                {"ids": unique, "now": now},
            )
        return int(result.rowcount or 0)

    # ------------------------------------------------------------------- reads
    def search(
        self,
        query: str,
        k: int = 5,
        tiers: tuple[MemoryTier, ...] | list[MemoryTier] | None = None,
        now: datetime | None = None,
    ) -> list[ScoredMemory]:
        """Cosine-retrieve, then re-rank with recency, usage, and decay."""
        now = now or utcnow()
        tiers = tiers if tiers is not None else [MemoryTier.SEMANTIC]
        tier_values = [t.value if isinstance(t, MemoryTier) else str(t) for t in tiers]
        vector = self.embedder.embed_query(query)
        with self.engine.connect() as conn:
            max_count = int(
                conn.execute(
                    text(
                        "SELECT COALESCE(MAX(access_count), 0) FROM memory_entries "
                        "WHERE status NOT IN ('superseded', 'retired')"
                    )
                ).scalar_one()
            )
            rows = conn.execute(
                text(
                    """
                    SELECT id, tier, subject, claim, body, embedding::text AS embedding_text,
                           source_doc_id, session_id, version, status, supersedes,
                           provenance, confidence, access_count, created_at, last_used_at,
                           content_hash,
                           1 - (embedding <=> CAST(:vec AS vector)) AS similarity
                    FROM memory_entries
                    WHERE embedding IS NOT NULL
                      AND status NOT IN ('superseded', 'retired')
                      AND tier = ANY(:tiers)
                    ORDER BY embedding <=> CAST(:vec AS vector), id
                    LIMIT :k
                    """
                ),
                {"vec": vector_literal(vector), "k": k, "tiers": tier_values},
            ).mappings()
            candidates: list[MemoryRecord] = []
            sims: dict[str, float] = {}
            for row in rows:
                record = self._row_to_record(row)
                candidates.append(record)
                sims[record.id] = float(row["similarity"])

        scored: list[ScoredMemory] = []
        for record in candidates:
            final, sim, recency, usage, decay = score_entry(
                sims[record.id],
                record.last_used_at,
                record.access_count,
                max_count,
                now,
                self.settings,
            )
            scored.append(
                ScoredMemory(
                    record=record,
                    similarity=sim,
                    recency=recency,
                    usage=usage,
                    decay=decay,
                    score=final,
                    tier=record.tier,
                )
            )
        scored.sort(key=lambda s: (-s.score, s.record.id))
        return scored

    def get(self, entry_id: str) -> MemoryRecord | None:
        """Fetch one entry by id, or ``None`` when absent."""
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM memory_entries WHERE id = CAST(:id AS uuid)"),
                {"id": entry_id},
            ).mappings().first()
        return self._row_to_record(row) if row else None

    def count(self, status: EntryStatus | None = None) -> int:
        """Count entries, optionally filtered by status."""
        sql = "SELECT count(*) FROM memory_entries"
        params: dict[str, Any] = {}
        if status is not None:
            sql += " WHERE status = :status"
            params["status"] = status.value if isinstance(status, EntryStatus) else str(status)
        with self.engine.connect() as conn:
            return int(conn.execute(text(sql), params).scalar_one())

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _row_to_record(row: Any) -> MemoryRecord:
        embedding = None
        if row.get("embedding_text"):
            embedding = json.loads(row["embedding_text"])
        provenance = row.get("provenance") or {}
        if isinstance(provenance, str):
            provenance = json.loads(provenance)
        return MemoryRecord(
            id=str(row["id"]),
            tier=MemoryTier(row["tier"]),
            subject=row["subject"],
            claim=row["claim"],
            body=row["body"] or "",
            embedding=embedding,
            source_doc_id=row["source_doc_id"],
            session_id=row["session_id"],
            version=int(row["version"]),
            status=EntryStatus(row["status"]),
            supersedes=str(row["supersedes"]) if row["supersedes"] else None,
            provenance=provenance,
            confidence=float(row["confidence"] or 0.0),
            access_count=int(row["access_count"] or 0),
            created_at=row["created_at"],
            last_used_at=row["last_used_at"],
            content_hash=row["content_hash"] or "",
        )
