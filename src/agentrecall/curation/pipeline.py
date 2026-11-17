"""Write-path curation: normalize, dedup, merge, adjudicate contradictions, version."""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from sqlalchemy import text
from sqlalchemy.engine import Engine

from agentrecall.config import Settings, get_settings
from agentrecall.curation.synonyms import content_hash, normalize_text
from agentrecall.db import get_engine
from agentrecall.embeddings import get_embedder, vector_literal
from agentrecall.llm import adjudicate, llm_factory
from agentrecall.memory.semantic import SemanticMemory
from agentrecall.models import (
    CurationDecision,
    EntryStatus,
    MemoryRecord,
    new_id,
    utcnow,
)

_NUMBER_UNIT_RE = re.compile(
    r"(?P<num>-?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>mj/cm2|mj/cm\^2|nm|%|percent|kv|mm|um|µm|nm|degrees?|deg|seconds?|sec|s|a\.u\.)",
    re.IGNORECASE,
)
_NEGATIONS = frozenset(("not", "never", "no", "avoid", "without", "prohibit", "exclude"))
_ANTONYMS: tuple[tuple[str, str], ...] = (
    ("increase", "decrease"),
    ("higher", "lower"),
    ("enable", "disable"),
    ("converges", "diverges"),
    ("pass", "fail"),
    ("above", "below"),
)


def extract_number_units(text: str) -> list[tuple[float, str]]:
    """Extract ``(value, canonical_unit)`` pairs from a claim."""
    pairs: list[tuple[float, str]] = []
    for match in _NUMBER_UNIT_RE.finditer(normalize_text(text)):
        unit = match.group("unit").lower()
        if unit in ("mj/cm^2", "mj/cm²"):
            unit = "mj/cm2"
        pairs.append((float(match.group("num")), unit))
    return pairs


def _claim_text(value: MemoryRecord | str) -> str:
    return value.claim if isinstance(value, MemoryRecord) else str(value)


def detect_contradiction(a: MemoryRecord | str, b: MemoryRecord | str) -> bool:
    """Detect numeric disagreement on the same unit or a negation/antonym flip."""
    text_a = normalize_text(_claim_text(a))
    text_b = normalize_text(_claim_text(b))
    if text_a == text_b:
        return False

    a_numbers: dict[str, set[float]] = {}
    b_numbers: dict[str, set[float]] = {}
    for value, unit in extract_number_units(text_a):
        a_numbers.setdefault(unit, set()).add(value)
    for value, unit in extract_number_units(text_b):
        b_numbers.setdefault(unit, set()).add(value)
    for unit in set(a_numbers) & set(b_numbers):
        if a_numbers[unit] != b_numbers[unit]:
            return True

    tokens_a = set(text_a.split())
    tokens_b = set(text_b.split())
    if bool(tokens_a & _NEGATIONS) != bool(tokens_b & _NEGATIONS):
        return True
    for left, right in _ANTONYMS:
        a_has_left = bool(re.search(rf"\b{left}\w*", text_a))
        a_has_right = bool(re.search(rf"\b{right}\w*", text_a))
        b_has_left = bool(re.search(rf"\b{left}\w*", text_b))
        b_has_right = bool(re.search(rf"\b{right}\w*", text_b))
        if (a_has_left and b_has_right) or (a_has_right and b_has_left):
            return True
    return False


def union_provenance(
    base: dict[str, Any], incoming: dict[str, Any], winner_first: bool = True
) -> dict[str, Any]:
    """Union provenance dictionaries, concatenating list-valued fields without duplicates."""
    result = copy.deepcopy(base) if winner_first else copy.deepcopy(incoming)
    other = incoming if winner_first else base
    for key, value in other.items():
        if key not in result:
            result[key] = copy.deepcopy(value)
        elif isinstance(result[key], list) and isinstance(value, list):
            merged: list[Any] = []
            for item in [*result[key], *value]:
                if item not in merged:
                    merged.append(item)
            result[key] = merged
        elif isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = {**result[key], **value}
        elif result[key] != value:
            result[key] = value if winner_first else result[key]
    return result


class CurationPipeline:
    """Run a candidate memory through the write-path stages in a fixed order."""

    def __init__(
        self,
        engine: Engine | None = None,
        embedder: Embeddings | None = None,
        llm: BaseChatModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.engine = engine or get_engine(self.settings)
        self.embedder = embedder or get_embedder()
        self.llm = llm or llm_factory(self.settings)
        self.semantic = SemanticMemory(
            engine=self.engine, embedder=self.embedder, settings=self.settings
        )

    # -------------------------------------------------------------- stage 1
    def normalize(self, candidate: MemoryRecord) -> MemoryRecord:
        """Return a copy with normalized subject/claim/body, hash, and embedding."""
        normalized = candidate.model_copy(deep=True)
        normalized.subject = normalize_text(candidate.subject)
        normalized.claim = normalize_text(candidate.claim)
        normalized.body = normalize_text(candidate.body or f"{candidate.subject} {candidate.claim}")
        normalized.content_hash = content_hash(normalized.subject, normalized.claim)
        normalized.embedding = self.embedder.embed_query(
            f"{normalized.subject} {normalized.claim} {normalized.body}"
        )
        if normalized.source_doc_id:
            with self.engine.connect() as conn:
                source_hash = conn.execute(
                    text("SELECT content_hash FROM source_docs WHERE id = :id"),
                    {"id": normalized.source_doc_id},
                ).scalar_one_or_none()
            provenance = dict(normalized.provenance)
            provenance["source_doc_id"] = normalized.source_doc_id
            if source_hash:
                provenance["source_content_hash"] = source_hash
            normalized.provenance = provenance
        return normalized

    # ---------------------------------------------------------------- ingest
    def ingest(self, candidate: MemoryRecord) -> CurationDecision:
        """Normalize then dedup, merge, adjudicate conflict, or accept a new entry."""
        record = self.normalize(candidate)

        exact = self._find_exact(record)
        if exact is not None:
            self.semantic.touch([exact.id])
            refreshed = self.semantic.get(exact.id) or exact
            self._log(
                "skip_duplicate", exact.id, None, "identical normalized content hash", 1.0
            )
            return CurationDecision(
                action="skip_duplicate",
                target_id=exact.id,
                reason="identical normalized content hash",
                similarity=1.0,
                score=exact.confidence,
                entry=refreshed,
            )

        near = self._nearest_active(record, self.settings.dedup_threshold)
        if near is not None:
            rival, similarity = near
            if detect_contradiction(record, rival):
                return self._resolve_conflict(record, rival, similarity)
            return self._merge(record, rival, similarity)

        conflict = self._nearest_active(record, self.settings.conflict_threshold)
        if conflict is not None:
            rival, similarity = conflict
            if detect_contradiction(record, rival):
                return self._resolve_conflict(record, rival, similarity)

        return self._accept(record)

    # -------------------------------------------------------------- stage 2
    def _find_exact(self, record: MemoryRecord) -> MemoryRecord | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT id FROM memory_entries "
                    "WHERE content_hash = :hash AND status = 'active' LIMIT 1"
                ),
                {"hash": record.content_hash},
            ).first()
        return self.semantic.get(str(row[0])) if row else None

    # -------------------------------------------------------------- stage 3
    def _nearest_active(
        self, record: MemoryRecord, threshold: float
    ) -> tuple[MemoryRecord, float] | None:
        vector = record.embedding or self.embedder.embed_query(
            f"{record.subject} {record.claim}"
        )
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id, 1 - (embedding <=> CAST(:vec AS vector)) AS similarity
                    FROM memory_entries
                    WHERE status = 'active'
                      AND tier = 'semantic'
                      AND subject = :subject
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(:vec AS vector), id
                    LIMIT 5
                    """
                ),
                {"vec": vector_literal(vector), "subject": record.subject},
            ).mappings()
            candidates = [(str(row["id"]), float(row["similarity"])) for row in rows]
        for entry_id, similarity in candidates:
            if similarity >= threshold:
                rival = self.semantic.get(entry_id)
                if rival is not None:
                    return rival, similarity
        return None

    def _merge(
        self, record: MemoryRecord, rival: MemoryRecord, similarity: float
    ) -> CurationDecision:
        winner = record if record.confidence > rival.confidence else rival
        merged = rival.model_copy(deep=True)
        merged.subject = winner.subject
        merged.claim = winner.claim
        merged.body = winner.body
        merged.embedding = winner.embedding or rival.embedding
        merged.confidence = max(record.confidence, rival.confidence)
        merged.created_at = min(record.created_at, rival.created_at)
        merged.last_used_at = utcnow()
        merged.version = max(record.version, rival.version) + 1
        merged.access_count = record.access_count + rival.access_count
        merged.provenance = union_provenance(
            rival.provenance, record.provenance, winner_first=winner is rival
        )
        merged.provenance["merged_from"] = sorted(
            {*merged.provenance.get("merged_from", []), record.id, rival.id}
        )
        self._update(merged)
        self._log(
            "merge",
            merged.id,
            record.id,
            f"near-duplicate cosine {similarity:.4f}; kept higher-confidence body",
            similarity,
        )
        return CurationDecision(
            action="merge",
            target_id=merged.id,
            rival_id=record.id,
            reason=f"near-duplicate cosine {similarity:.4f}",
            similarity=similarity,
            score=merged.confidence,
            entry=merged,
        )

    # -------------------------------------------------------------- stage 4
    def _resolve_conflict(
        self, record: MemoryRecord, rival: MemoryRecord, similarity: float
    ) -> CurationDecision:
        verdict = adjudicate(
            {
                "id": record.id,
                "claim": record.claim,
                "confidence": record.confidence,
                "created_at": record.created_at.isoformat(),
                "provenance": record.provenance,
            },
            {
                "id": rival.id,
                "claim": rival.claim,
                "confidence": rival.confidence,
                "created_at": rival.created_at.isoformat(),
                "provenance": rival.provenance,
            },
            self.llm,
        )
        candidate_wins = verdict.get("winner", "A") == "A"
        reason = str(verdict.get("reason", "adjudicated"))

        if candidate_wins:
            new_version = rival.version + 1
            record.version = new_version
            record.supersedes = rival.id
            record.status = EntryStatus.ACTIVE
            record.provenance = union_provenance(record.provenance, rival.provenance, True)
            record.provenance["adjudication"] = {
                "winner": "candidate",
                "reason": reason,
                "similarity": similarity,
            }
            record.created_at = utcnow()
            record.last_used_at = record.created_at
            self.semantic.write(record)
            self._set_status(rival.id, EntryStatus.SUPERSEDED)
            self._log("supersede", record.id, rival.id, reason, similarity)
            return CurationDecision(
                action="supersede",
                target_id=record.id,
                rival_id=rival.id,
                reason=reason,
                similarity=similarity,
                score=record.confidence,
                entry=record,
            )

        record.version = rival.version + 1
        record.supersedes = rival.id
        record.status = EntryStatus.SUPERSEDED
        record.provenance = union_provenance(record.provenance, rival.provenance, False)
        record.provenance["adjudication"] = {
            "winner": "existing",
            "reason": reason,
            "similarity": similarity,
        }
        record.created_at = utcnow()
        record.last_used_at = record.created_at
        self.semantic.write(record)
        self._log("reject_conflict", rival.id, record.id, reason, similarity)
        return CurationDecision(
            action="reject_conflict",
            target_id=rival.id,
            rival_id=record.id,
            reason=f"existing version retained: {reason}",
            similarity=similarity,
            score=rival.confidence,
            entry=rival,
        )

    # -------------------------------------------------------------- stage 5
    def _accept(self, record: MemoryRecord) -> CurationDecision:
        record.version = 1
        record.status = EntryStatus.ACTIVE
        self.semantic.write(record)
        self._log("accept", record.id, None, "no duplicate or contradiction found", 0.0)
        return CurationDecision(
            action="accept",
            target_id=record.id,
            reason="no duplicate or contradiction found",
            similarity=0.0,
            score=record.confidence,
            entry=record,
        )

    # --------------------------------------------------------------- history
    def history(self, entry_id: str) -> list[MemoryRecord]:
        """Return the full supersede lineage, newest first, plus orphaned branches."""
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT id, tier, subject, claim, body, embedding::text AS embedding_text, "
                    "source_doc_id, session_id, version, status, supersedes, provenance, "
                    "confidence, access_count, created_at, last_used_at, content_hash "
                    "FROM memory_entries"
                )
            ).mappings()
            records = {str(r["id"]): SemanticMemory._row_to_record(r) for r in rows}
        if entry_id not in records:
            return []
        current = records[entry_id]
        # Walk backwards through supersede pointers.
        lineage: list[MemoryRecord] = []
        seen: set[str] = set()
        while current is not None and current.id not in seen:
            seen.add(current.id)
            lineage.append(current)
            current = records.get(current.supersedes) if current.supersedes else None
        return lineage

    # --------------------------------------------------------------- helpers
    def _update(self, record: MemoryRecord) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE memory_entries SET
                        subject = :subject, claim = :claim, body = :body,
                        embedding = CAST(:embedding AS vector),
                        version = :version, status = :status,
                        supersedes = CAST(:supersedes AS uuid),
                        provenance = CAST(:provenance AS jsonb), confidence = :confidence,
                        access_count = :access_count, created_at = :created_at,
                        last_used_at = :last_used_at, content_hash = :content_hash,
                        source_doc_id = :source_doc_id
                    WHERE id = CAST(:id AS uuid)
                    """
                ),
                {
                    "id": record.id,
                    "subject": record.subject,
                    "claim": record.claim,
                    "body": record.body,
                    "embedding": vector_literal(record.embedding or []),
                    "version": record.version,
                    "status": record.status.value,
                    "supersedes": record.supersedes,
                    "provenance": json.dumps(record.provenance, default=str),
                    "confidence": record.confidence,
                    "access_count": record.access_count,
                    "created_at": record.created_at,
                    "last_used_at": record.last_used_at,
                    "content_hash": record.content_hash,
                    "source_doc_id": record.source_doc_id,
                },
            )

    def _set_status(self, entry_id: str, status: EntryStatus, at: datetime | None = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE memory_entries SET status = :status, last_used_at = :at "
                    "WHERE id = CAST(:id AS uuid)"
                ),
                {"id": entry_id, "status": status.value, "at": at or utcnow()},
            )

    def _log(
        self,
        action: str,
        target_id: str | None,
        rival_id: str | None,
        reason: str,
        similarity: float,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO curation_log "
                    "(id, action, target_id, rival_id, reason, similarity, created_at) "
                    "VALUES (CAST(:id AS uuid), :action, CAST(:target AS uuid), "
                    "CAST(:rival AS uuid), :reason, :similarity, :created_at)"
                ),
                {
                    "id": new_id(),
                    "action": action,
                    "target": target_id,
                    "rival": rival_id,
                    "reason": reason,
                    "similarity": similarity,
                    "created_at": utcnow(),
                },
            )
