"""Idempotent knowledge-repository maintenance jobs.

All four jobs read and write the real Postgres store. They never delete memory rows:
staleness and retirement are status transitions, and superseded versions remain queryable
through the supersede chain. Each job returns a Pydantic result model for CLI/eval output.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.engine import Engine

from agentrecall.config import Settings, get_settings
from agentrecall.curation.pipeline import detect_contradiction
from agentrecall.db import get_engine
from agentrecall.embeddings import get_embedder, vector_literal
from agentrecall.models import AuditFinding, EntryStatus, utcnow


class StaleResult(BaseModel):
    """Outcome of ``detect_stale``."""

    scanned: int = 0
    marked_stale: int = 0
    reasons: dict[str, int] = Field(default_factory=dict)
    stale_total: int = 0


class ReembedResult(BaseModel):
    """Outcome of ``reembed_changed``."""

    scanned_documents: int = 0
    changed_documents: int = 0
    reembedded_entries: int = 0
    skipped_documents: int = 0


class AuditResult(BaseModel):
    """Outcome of ``audit``."""

    scanned: int = 0
    findings: list[AuditFinding] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)


class CompactResult(BaseModel):
    """Outcome of ``compact``."""

    inspected: int = 0
    retired: int = 0
    skipped: int = 0


def _provenance(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return json.loads(value)
    return dict(value or {})


def _body_hash(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def detect_stale(
    engine: Engine | None = None,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> StaleResult:
    """Mark ACTIVE entries STALE for age, changed source content, or a missing source."""
    settings = settings or get_settings()
    engine = engine or get_engine(settings)
    now = now or utcnow()
    reasons: Counter[str] = Counter()
    marked = 0

    with engine.connect() as conn:
        docs = {
            str(row["id"]): row["body"] or ""
            for row in conn.execute(text("SELECT id, body FROM source_docs")).mappings()
        }
        entries = list(
            conn.execute(
                text(
                    "SELECT id, last_used_at, source_doc_id, provenance "
                    "FROM memory_entries WHERE status = 'active'"
                )
            ).mappings()
        )

    transitions: list[tuple[str, str]] = []
    for entry in entries:
        prov = _provenance(entry["provenance"])
        causes: list[str] = []
        if entry["last_used_at"] and entry["last_used_at"] < now - timedelta(
            days=settings.stale_after_days
        ):
            causes.append("age")
        doc_ref = prov.get("source_doc_id") or entry["source_doc_id"]
        if doc_ref:
            if doc_ref not in docs:
                causes.append("missing_source")
            else:
                baseline = prov.get("source_content_hash")
                if baseline and baseline != _body_hash(docs[doc_ref]):
                    causes.append("source_changed")
        if causes:
            transitions.append((str(entry["id"]), "stale"))
            for cause in causes:
                reasons[cause] += 1

    with engine.begin() as conn:
        for entry_id, status in transitions:
            conn.execute(
                text(
                    "UPDATE memory_entries SET status = :status WHERE id = CAST(:id AS uuid)"
                ),
                {"id": entry_id, "status": status},
            )
        stale_total = int(
            conn.execute(
                text("SELECT count(*) FROM memory_entries WHERE status = 'stale'")
            ).scalar_one()
        )
    marked = len(transitions)
    return StaleResult(
        scanned=len(entries),
        marked_stale=marked,
        reasons=dict(sorted(reasons.items())),
        stale_total=stale_total,
    )


def reembed_changed(
    engine: Engine | None = None,
    settings: Settings | None = None,
) -> ReembedResult:
    """Recompute source hashes, then re-embed only entries whose source body changed."""
    settings = settings or get_settings()
    engine = engine or get_engine(settings)
    embedder = get_embedder()

    with engine.connect() as conn:
        docs = list(
            conn.execute(
                text("SELECT id, body, content_hash FROM source_docs ORDER BY id")
            ).mappings()
        )
        entries = list(
            conn.execute(
                text(
                    "SELECT id, subject, claim, body, source_doc_id, provenance "
                    "FROM memory_entries"
                )
            ).mappings()
        )

    changed_docs: dict[str, str] = {}
    skipped = 0
    for doc in docs:
        current = _body_hash(doc["body"] or "")
        if current != (doc["content_hash"] or ""):
            changed_docs[str(doc["id"])] = current
        else:
            skipped += 1

    reembedded = 0
    updates: list[dict[str, Any]] = []
    for entry in entries:
        prov = _provenance(entry["provenance"])
        doc_ref = entry["source_doc_id"] or prov.get("source_doc_id")
        if not doc_ref or doc_ref not in changed_docs:
            continue
        vector = embedder.embed_query(
            f"{entry['subject']} {entry['claim']} {entry['body'] or ''}"
        )
        prov["source_doc_id"] = doc_ref
        prov["source_content_hash"] = changed_docs[doc_ref]
        updates.append(
            {
                "id": str(entry["id"]),
                "embedding": vector_literal(vector),
                "provenance": json.dumps(prov, default=str),
            }
        )
        reembedded += 1

    with engine.begin() as conn:
        for doc_id, current_hash in changed_docs.items():
            conn.execute(
                text("UPDATE source_docs SET content_hash = :hash WHERE id = :id"),
                {"hash": current_hash, "id": doc_id},
            )
        for update in updates:
            conn.execute(
                text(
                    "UPDATE memory_entries SET embedding = CAST(:embedding AS vector), "
                    "provenance = CAST(:provenance AS jsonb) WHERE id = CAST(:id AS uuid)"
                ),
                update,
            )
    return ReembedResult(
        scanned_documents=len(docs),
        changed_documents=len(changed_docs),
        reembedded_entries=reembedded,
        skipped_documents=skipped,
    )


def audit(
    engine: Engine | None = None,
    settings: Settings | None = None,
    now: datetime | None = None,
    zero_usage_days: int = 30,
) -> AuditResult:
    """Find orphaned, contradictory, multiply-headed, dangling, and unused entries."""
    settings = settings or get_settings()
    engine = engine or get_engine(settings)
    now = now or utcnow()
    findings: list[AuditFinding] = []

    with engine.connect() as conn:
        docs = {
            str(row[0]) for row in conn.execute(text("SELECT id FROM source_docs"))
        }
        entries = list(
            conn.execute(
                text(
                    "SELECT id, subject, claim, embedding::text AS embedding_text, "
                    "source_doc_id, status, supersedes, provenance, access_count, created_at "
                    "FROM memory_entries ORDER BY id"
                )
            ).mappings()
        )

    by_id = {str(entry["id"]): entry for entry in entries}
    for entry in entries:
        prov = _provenance(entry["provenance"])
        doc_ref = prov.get("source_doc_id") or entry["source_doc_id"]
        if (doc_ref and doc_ref not in docs) or (
            entry["source_doc_id"] and entry["source_doc_id"] not in docs
        ):
            findings.append(
                AuditFinding(
                    kind="orphaned_entry",
                    severity="error",
                    entry_id=str(entry["id"]),
                    detail=f"source document {doc_ref!r} is missing",
                )
            )
        if entry["supersedes"] and str(entry["supersedes"]) not in by_id:
            findings.append(
                AuditFinding(
                    kind="dangling_supersede",
                    severity="error",
                    entry_id=str(entry["id"]),
                    detail=f"supersedes {entry['supersedes']} does not exist",
                )
            )
        if entry["access_count"] == 0 and entry["created_at"] < now - timedelta(
            days=zero_usage_days
        ):
            findings.append(
                AuditFinding(
                    kind="zero_usage_old",
                    severity="info",
                    entry_id=str(entry["id"]),
                    detail=f"never used and older than {zero_usage_days} days",
                )
            )

    active = [entry for entry in entries if entry["status"] == EntryStatus.ACTIVE.value]
    by_subject: dict[str, list[Any]] = {}
    for entry in active:
        by_subject.setdefault(entry["subject"], []).append(entry)

    for subject, group in sorted(by_subject.items()):
        for left, right in combinations(sorted(group, key=lambda e: str(e["id"])), 2):
            if detect_contradiction(left["claim"], right["claim"]):
                findings.append(
                    AuditFinding(
                        kind="contradictory_active_pair",
                        severity="error",
                        entry_id=str(left["id"]),
                        detail=(
                            f"subject {subject!r} has contradictory claims: {left['claim']!r} "
                            f"vs {right['claim']!r} (other id {right['id']})"
                        ),
                    )
                )

    heads: dict[tuple[str, str], list[str]] = {}
    for entry in active:
        heads.setdefault((entry["subject"], entry["claim"]), []).append(str(entry["id"]))
    for (subject, claim), ids in sorted(heads.items()):
        if len(ids) > 1:
            findings.append(
                AuditFinding(
                    kind="multiple_active_heads",
                    severity="error",
                    entry_id=ids[0],
                    detail=(
                        f"{len(ids)} ACTIVE heads for subject={subject!r} claim={claim!r}: "
                        + ", ".join(ids)
                    ),
                )
            )

    counts = dict(sorted(Counter(f.kind for f in findings).items()))
    return AuditResult(scanned=len(entries), findings=findings, counts=counts)


def compact(
    engine: Engine | None = None,
    settings: Settings | None = None,
) -> CompactResult:
    """Retire STALE, unused, non-head entries that have a surviving superseding version."""
    settings = settings or get_settings()
    engine = engine or get_engine(settings)
    with engine.connect() as conn:
        stale = list(
            conn.execute(
                text(
                    "SELECT id, access_count FROM memory_entries "
                    "WHERE status = 'stale' ORDER BY id"
                )
            ).mappings()
        )
        superseding = {
            str(row[0])
            for row in conn.execute(
                text(
                    "SELECT DISTINCT supersedes FROM memory_entries "
                    "WHERE supersedes IS NOT NULL AND status = 'active'"
                )
            )
        }
        active_heads = {
            str(row[0])
            for row in conn.execute(
                text("SELECT id FROM memory_entries WHERE status = 'active'")
            )
        }

    retire = [
        str(entry["id"])
        for entry in stale
        if entry["access_count"] == 0
        and str(entry["id"]) in superseding
        and str(entry["id"]) not in active_heads
    ]
    with engine.begin() as conn:
        for entry_id in retire:
            conn.execute(
                text(
                    "UPDATE memory_entries SET status = 'retired' "
                    "WHERE id = CAST(:id AS uuid) AND status = 'stale'"
                ),
                {"id": entry_id},
            )
    return CompactResult(
        inspected=len(stale), retired=len(retire), skipped=len(stale) - len(retire)
    )
