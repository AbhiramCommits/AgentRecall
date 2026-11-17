"""Seeded synthetic computational-lithography corpus with ground-truth curation labels.

The generator returns roughly sixty source documents and a candidate stream containing a
known mix of unique facts, exact duplicates, near-duplicate paraphrases, numeric
contradictions, and negation contradictions. Every candidate carries its label and, where
relevant, the index of the base fact it relates to, so curation precision/recall is
measurable without hand inspection.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, Field

from agentrecall.models import MemoryRecord, MemoryTier, SourceDoc

_NAMESPACE = uuid.UUID("6f6b1f5e-9e6a-4e5a-9b6f-2a1d9c7e3b10")
_STACKS = tuple("ABCDEFGHIJKLMNOPQRST")
_BASE_TIME = datetime(2025, 9, 1, 8, 0, tzinfo=UTC)


class CorpusCandidate(BaseModel):
    """A candidate memory plus the ground-truth label used by evaluation."""

    candidate: MemoryRecord
    label: str
    related_index: int | None = None
    doc_id: str = ""
    tags: list[str] = Field(default_factory=list)


def _doc_id(prefix: str, index: int) -> str:
    return f"doc-{prefix}-{index:02d}"


def _hash(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _dose_claim(stack: str, value: int, connecting_word: str = "for") -> str:
    return (
        f"use dose {value} mj/cm2 {connecting_word} resist stack {stack} at 193 nm "
        f"with numerical aperture 1.35 and depth of focus 80 nm"
    )


def _base_memory(index: int, seed: int) -> MemoryRecord:
    stack = _STACKS[index % len(_STACKS)]
    value = 30 + (seed + index) % 7
    doc_id = _doc_id("dose", index)
    claim = _dose_claim(stack, value)
    return MemoryRecord(
        id=str(uuid.uuid5(_NAMESPACE, f"candidate/{seed}/{index}")),
        tier=MemoryTier.SEMANTIC,
        subject=f"resist stack {stack} dose",
        claim=claim,
        body=f"{claim}; qualified on the 193i immersion scanner.",
        source_doc_id=doc_id,
        session_id="session-corpus",
        confidence=0.70 + (index % 3) * 0.05,
        created_at=_BASE_TIME + timedelta(hours=index),
        last_used_at=_BASE_TIME + timedelta(hours=index),
        provenance={"sources": [doc_id], "origin": "synthetic-corpus", "seed": seed},
    )


def build_source_docs(seed: int = 7) -> list[SourceDoc]:
    """Create 60 lithography source documents (20 dose, 20 OPC, 20 job-deck notes)."""
    docs: list[SourceDoc] = []
    for index in range(20):
        stack = _STACKS[index % len(_STACKS)]
        value = 30 + (seed + index) % 7
        doc_id = _doc_id("dose", index)
        body = (
            f"Resist stack {stack} dose qualification: exposure dose {value} mJ/cm^2 at "
            f"193 nm, numerical aperture 1.35, depth of focus 80 nm."
        )
        docs.append(
            SourceDoc(
                id=doc_id,
                uri=f"https://litho.example/recipes/{doc_id}",
                title=f"Resist stack {stack} dose qualification",
                body=body,
                content_hash=_hash(body),
                updated_at=_BASE_TIME - timedelta(days=30) + timedelta(hours=index),
                tags=["dose", "resist", f"stack-{stack.lower()}"],
            )
        )
    for index in range(20):
        doc_id = _doc_id("opc", index)
        body = (
            f"OPC recipe note {index}: model calibration converged after "
            f"{4 + index % 5} iterations with edge placement error {1.2 + index * 0.1:.1f} nm."
        )
        docs.append(
            SourceDoc(
                id=doc_id,
                uri=f"https://litho.example/opc/{doc_id}",
                title=f"OPC calibration note {index}",
                body=body,
                content_hash=_hash(body),
                updated_at=_BASE_TIME - timedelta(days=20) + timedelta(hours=index),
                tags=["opc", "calibration"],
            )
        )
    for index in range(20):
        doc_id = _doc_id("deck", index)
        body = (
            f"Job deck note {index}: layer M{index + 1} uses {2 + index % 4} exposure passes, "
            f"reticle field 26 x 33 mm, and dose monitor every {5 + index % 6} lots."
        )
        docs.append(
            SourceDoc(
                id=doc_id,
                uri=f"https://litho.example/decks/{doc_id}",
                title=f"Job deck note {index}",
                body=body,
                content_hash=_hash(body),
                updated_at=_BASE_TIME - timedelta(days=10) + timedelta(hours=index),
                tags=["job-deck", f"layer-M{index + 1}"],
            )
        )
    return docs


def build_candidates(seed: int = 7) -> list[CorpusCandidate]:
    """Create 60 labeled candidates: 20 unique, 10 exact, 10 near, 20 contradictory."""
    bases = [_base_memory(i, seed) for i in range(20)]
    candidates: list[CorpusCandidate] = []

    for base in bases:
        candidates.append(
            CorpusCandidate(
                candidate=base.model_copy(deep=True),
                label="unique_fact",
                doc_id=base.source_doc_id or "",
                tags=["unique"],
            )
        )

    def clone(base: MemoryRecord, suffix: str, **updates) -> MemoryRecord:
        copied = base.model_copy(deep=True)
        copied.id = str(uuid.uuid5(_NAMESPACE, f"candidate/{seed}/{suffix}"))
        copied.created_at = _BASE_TIME + timedelta(hours=20 + len(candidates))
        copied.last_used_at = copied.created_at
        copied.provenance = {
            **base.provenance,
            "sources": [*base.provenance.get("sources", [])],
            "submission": suffix,
        }
        for key, value in updates.items():
            setattr(copied, key, value)
        return copied

    for offset in range(10):
        base_index = offset
        base = bases[base_index]
        # Exact duplicate: only whitespace and case change; normalized hash is identical.
        text = base.claim.replace("use dose", "Use  Dose")
        candidates.append(
            CorpusCandidate(
                candidate=clone(
                    base,
                    f"exact-{offset}",
                    claim=text.replace("  ", "   "),
                    confidence=0.55,
                    provenance={**base.provenance, "submission": f"exact-{offset}"},
                ),
                label="exact_duplicate",
                related_index=base_index,
                doc_id=base.source_doc_id or "",
                tags=["duplicate"],
            )
        )

    for offset in range(10):
        base_index = 10 + offset
        base = bases[base_index]
        # Near duplicate: a stopword-level paraphrase that leaves the token stream identical.
        text = base.claim.replace(" for ", " on ").replace("mj/cm2", "mJ/cm^2")
        candidates.append(
            CorpusCandidate(
                candidate=clone(base, f"near-{offset}", claim=text, confidence=0.65),
                label="near_duplicate",
                related_index=base_index,
                doc_id=base.source_doc_id or "",
                tags=["duplicate", "paraphrase"],
            )
        )

    for offset in range(10):
        base_index = offset
        base = bases[base_index]
        stack = _STACKS[base_index % len(_STACKS)]
        original = 30 + (seed + base_index) % 7
        contradicted = _dose_claim(stack, original + 4)
        candidates.append(
            CorpusCandidate(
                candidate=clone(
                    base,
                    f"numeric-{offset}",
                    claim=contradicted,
                    confidence=0.9,
                    provenance={
                        **base.provenance,
                        "sources": [*base.provenance.get("sources", []), "doc-bench-01"],
                        "submission": f"numeric-{offset}",
                        "measured_on": "scanner-B",
                    },
                ),
                label="numeric_contradiction",
                related_index=base_index,
                doc_id=base.source_doc_id or "",
                tags=["contradiction", "numeric"],
            )
        )

    for offset in range(10):
        base_index = 10 + offset
        base = bases[base_index]
        negated = base.claim.replace("use dose", "must not use dose", 1)
        candidates.append(
            CorpusCandidate(
                candidate=clone(base, f"negation-{offset}", claim=negated, confidence=0.92),
                label="negation_contradiction",
                related_index=base_index,
                doc_id=base.source_doc_id or "",
                tags=["contradiction", "negation"],
            )
        )

    return candidates


def generate_corpus(seed: int = 7) -> tuple[list[SourceDoc], list[CorpusCandidate]]:
    """Return source documents and labeled candidates for a fixed seed."""
    return build_source_docs(seed), build_candidates(seed)


MUTATION_MARKER = " Revised measurement note: dose drift observed during re-qualification."


def mutate_source_docs(engine, limit: int = 3) -> list[str]:
    """Deterministically edit the first N dose docs, leaving content_hash stale on purpose."""
    from sqlalchemy import text

    from agentrecall.models import utcnow

    with engine.connect() as conn:
        ids = [
            str(row[0])
            for row in conn.execute(
                text(
                    "SELECT id FROM source_docs WHERE id LIKE 'doc-dose-%' "
                    "ORDER BY id LIMIT :limit"
                ),
                {"limit": limit},
            )
        ]
    touched: list[str] = []
    with engine.begin() as conn:
        for doc_id in ids:
            body = conn.execute(
                text("SELECT body FROM source_docs WHERE id = :id"), {"id": doc_id}
            ).scalar_one()
            if MUTATION_MARKER.strip() in body:
                continue
            conn.execute(
                text(
                    "UPDATE source_docs SET body = :body, updated_at = :updated_at WHERE id = :id"
                ),
                {"body": body + MUTATION_MARKER, "updated_at": utcnow(), "id": doc_id},
            )
            touched.append(doc_id)
    return touched


def insert_source_docs(engine, docs: list[SourceDoc]) -> int:
    """Upsert source documents and return the number written."""
    from sqlalchemy import text

    with engine.begin() as conn:
        for doc in docs:
            conn.execute(
                text(
                    """
                    INSERT INTO source_docs (id, uri, title, body, content_hash, updated_at, tags)
                    VALUES (:id, :uri, :title, :body, :content_hash, :updated_at, :tags)
                    ON CONFLICT (id) DO UPDATE SET
                        uri = EXCLUDED.uri, title = EXCLUDED.title, body = EXCLUDED.body,
                        content_hash = EXCLUDED.content_hash, updated_at = EXCLUDED.updated_at,
                        tags = EXCLUDED.tags
                    """
                ),
                {
                    "id": doc.id,
                    "uri": doc.uri,
                    "title": doc.title,
                    "body": doc.body,
                    "content_hash": doc.content_hash,
                    "updated_at": doc.updated_at,
                    "tags": doc.tags,
                },
            )
    return len(docs)
