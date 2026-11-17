"""Curation tests: each stage, lineage integrity, provenance, and measured accuracy."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from agentrecall.curation import (
    CurationPipeline,
    canonicalize,
    content_hash,
    detect_contradiction,
    normalize_text,
    union_provenance,
)
from agentrecall.data.corpus import build_candidates, generate_corpus, insert_source_docs
from agentrecall.models import EntryStatus, MemoryRecord, MemoryTier, SourceDoc

BASE = datetime(2025, 10, 1, 9, 0, tzinfo=UTC)
LONG_BODY = (
    "qualified on the 193i immersion scanner with numerical aperture 1.35 and depth of "
    "focus 80 nm for the production lithography recipe"
)


@pytest.fixture(autouse=True)
def _seed_sources(engine, clean_db):
    docs = []
    for doc_id in ("doc-a", "doc-low", "doc-high", "doc-bench-01"):
        body = f"fixture source body for {doc_id}"
        docs.append(
            SourceDoc(
                id=doc_id,
                uri=f"https://litho.example/{doc_id}",
                title=doc_id,
                body=body,
                content_hash=hashlib.sha256(body.encode()).hexdigest(),
                updated_at=BASE,
                tags=["fixture"],
            )
        )
    insert_source_docs(engine, docs)
    yield


def _candidate(
    subject: str,
    claim: str,
    *,
    body: str | None = None,
    confidence: float = 0.7,
    created_at: datetime = BASE,
    source: str = "doc-a",
    sources: list[str] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        tier=MemoryTier.SEMANTIC,
        subject=subject,
        claim=claim,
        body=body or f"{subject} {claim}",
        confidence=confidence,
        created_at=created_at,
        last_used_at=created_at,
        source_doc_id=source,
        session_id="session-curation",
        provenance={"sources": sources or [source], "origin": "test"},
    )


# ---------------------------------------------------------------- normalization
def test_canonicalize_units_and_parameters():
    assert canonicalize("NA 1.35 with exposure dose 32 mJ/cm^2") == (
        "numerical aperture 1.35 with dose 32 mj/cm2"
    )
    assert normalize_text("  Dose   32   mJ/cm²  ") == "dose 32 mj/cm2"
    assert content_hash("NA", "dose 32 mJ/cm2") == content_hash(
        "numerical aperture", "exposure dose 32 mJ/cm^2"
    )


def test_detect_contradiction_numeric_negation_and_antonym():
    assert detect_contradiction("dose is 32 mJ/cm2", "dose is 36 mJ/cm2")
    assert detect_contradiction("use dose 32 mJ/cm2", "must not use dose 32 mJ/cm2")
    assert detect_contradiction("edge placement increases", "edge placement decreases")
    assert not detect_contradiction("dose is 32 mJ/cm2", "dose is 32.0 mJ/cm2")
    assert not detect_contradiction("dose is 32 mJ/cm2", "wavelength 193 nm")


# ------------------------------------------------------------------- stages
def test_accept_plain_insert(engine, settings, embedder):
    pipeline = CurationPipeline(engine=engine, embedder=embedder, settings=settings)
    decision = pipeline.ingest(_candidate("resist stack A dose", "use dose 32 mj/cm2"))
    assert decision.action == "accept"
    assert decision.entry.version == 1
    assert decision.entry.status == EntryStatus.ACTIVE
    assert pipeline.semantic.count() == 1


def test_exact_dedup_skips_and_bumps_usage(engine, settings, embedder):
    pipeline = CurationPipeline(engine=engine, embedder=embedder, settings=settings)
    first = pipeline.ingest(_candidate("NA", "numerical aperture 1.35"))
    duplicate = _candidate("na", "NA 1.35  ")
    decision = pipeline.ingest(duplicate)
    assert decision.action == "skip_duplicate"
    assert decision.target_id == first.target_id
    assert pipeline.semantic.count() == 1
    stored = pipeline.semantic.get(first.target_id)
    assert stored.access_count == 1
    assert stored.last_used_at > stored.created_at


def test_near_dedup_merges_body_provenance_and_version(engine, settings, embedder):
    pipeline = CurationPipeline(engine=engine, embedder=embedder, settings=settings)
    base = pipeline.ingest(
        _candidate(
            "resist stack A dose",
            "use dose 32 mj/cm2 for resist stack A at 193 nm",
            body=LONG_BODY,
            confidence=0.6,
            sources=["doc-low"],
        )
    )
    paraphrase = _candidate(
        "resist stack A dose",
        "use dose 32 mj/cm2 on resist stack A at 193 nm",
        body=LONG_BODY + " verified twice",
        confidence=0.9,
        created_at=BASE + timedelta(hours=5),
        sources=["doc-high"],
    )
    decision = pipeline.ingest(paraphrase)
    assert decision.action == "merge"
    assert decision.similarity >= settings.dedup_threshold
    assert pipeline.semantic.count() == 1
    merged = pipeline.semantic.get(base.target_id)
    assert merged.version == 2
    assert merged.confidence == 0.9
    assert "verified twice" in merged.body
    assert merged.created_at == BASE
    assert set(merged.provenance["sources"]) == {"doc-low", "doc-high"}
    assert set(merged.provenance["merged_from"]) == {base.target_id, paraphrase.id}


def test_numeric_conflict_supersedes_never_deletes(engine, settings, embedder):
    pipeline = CurationPipeline(engine=engine, embedder=embedder, settings=settings)
    base = pipeline.ingest(
        _candidate(
            "resist stack A dose",
            "use dose 32 mj/cm2 for stack A",
            body=LONG_BODY,
            confidence=0.6,
        )
    )
    challenger = _candidate(
        "resist stack A dose",
        "use dose 36 mj/cm2 for stack A",
        body=LONG_BODY,
        confidence=0.9,
        created_at=BASE + timedelta(days=1),
    )
    decision = pipeline.ingest(challenger)
    assert decision.action == "supersede"
    assert decision.entry.version == 2
    assert decision.entry.supersedes == base.target_id
    old = pipeline.semantic.get(base.target_id)
    new = pipeline.semantic.get(decision.target_id)
    assert old.status == EntryStatus.SUPERSEDED
    assert new.status == EntryStatus.ACTIVE
    assert pipeline.semantic.count() == 2
    hits = pipeline.semantic.search("dose for resist stack A", k=5)
    assert [h.record.id for h in hits] == [decision.target_id]


def test_negation_conflict_supersedes(engine, settings, embedder):
    pipeline = CurationPipeline(engine=engine, embedder=embedder, settings=settings)
    pipeline.ingest(
        _candidate("dose", "use dose 32 mj/cm2", body=LONG_BODY, confidence=0.6)
    )
    decision = pipeline.ingest(
        _candidate("dose", "must not use dose 32 mj/cm2", body=LONG_BODY, confidence=0.92)
    )
    assert decision.action == "supersede"
    assert "not" in pipeline.semantic.get(decision.target_id).claim


def test_conflict_adjudication_can_keep_existing_head(engine, settings, embedder):
    pipeline = CurationPipeline(engine=engine, embedder=embedder, settings=settings)
    base = pipeline.ingest(
        _candidate(
            "dose", "use dose 32 mj/cm2", body=LONG_BODY, confidence=0.95, created_at=BASE
        )
    )
    weak = pipeline.ingest(
        _candidate(
            "dose",
            "use dose 36 mj/cm2",
            body=LONG_BODY,
            confidence=0.2,
            created_at=BASE + timedelta(days=2),
        )
    )
    assert weak.action == "reject_conflict"
    assert weak.target_id == base.target_id
    active = pipeline.semantic.get(base.target_id)
    rejected = pipeline.semantic.get(weak.rival_id)
    assert active.status == EntryStatus.ACTIVE
    assert rejected.status == EntryStatus.SUPERSEDED
    assert pipeline.semantic.count() == 2


def test_supersede_chain_integrity_and_history(engine, settings, embedder):
    pipeline = CurationPipeline(engine=engine, embedder=embedder, settings=settings)
    v1 = pipeline.ingest(
        _candidate(
            "resist stack B dose", "use dose 30 mj/cm2", body=LONG_BODY, confidence=0.6
        )
    )
    v2 = pipeline.ingest(
        _candidate(
            "resist stack B dose",
            "use dose 34 mj/cm2",
            body=LONG_BODY,
            confidence=0.8,
            created_at=BASE + timedelta(days=1),
        )
    )
    v3 = pipeline.ingest(
        _candidate(
            "resist stack B dose",
            "use dose 38 mj/cm2",
            body=LONG_BODY,
            confidence=0.97,
            created_at=BASE + timedelta(days=2),
        )
    )
    assert [v1.action, v2.action, v3.action] == ["accept", "supersede", "supersede"]
    history = pipeline.history(v3.target_id)
    assert [h.version for h in history] == [3, 2, 1]
    assert len({h.id for h in history}) == 3
    with engine.connect() as conn:
        active = conn.execute(
            text(
                "SELECT count(*) FROM memory_entries "
                "WHERE subject='resist stack b dose' AND status='active'"
            )
        ).scalar_one()
        dangling = conn.execute(
            text(
                "SELECT count(*) FROM memory_entries m WHERE m.supersedes IS NOT NULL "
                "AND NOT EXISTS (SELECT 1 FROM memory_entries p WHERE p.id=m.supersedes)"
            )
        ).scalar_one()
    assert active == 1
    assert dangling == 0


def test_union_provenance_is_deduplicated():
    merged = union_provenance(
        {"sources": ["a", "b"], "meta": {"x": 1}},
        {"sources": ["b", "c"], "meta": {"y": 2}},
    )
    assert merged["sources"] == ["a", "b", "c"]
    assert merged["meta"] == {"x": 1, "y": 2}


def test_curation_log_records_each_decision(engine, settings, embedder):
    pipeline = CurationPipeline(engine=engine, embedder=embedder, settings=settings)
    pipeline.ingest(_candidate("subject", "claim one"))
    pipeline.ingest(_candidate("subject", "claim one"))
    with engine.connect() as conn:
        actions = [
            row[0]
            for row in conn.execute(text("SELECT action FROM curation_log ORDER BY created_at"))
        ]
    assert actions == ["accept", "skip_duplicate"]


# ------------------------------------------------------------- end-to-end
def test_end_to_end_labeled_stream_accuracy(engine, settings, embedder):
    docs, candidates = generate_corpus(seed=7)
    assert len(docs) == 60
    assert len(candidates) == 60
    insert_source_docs(engine, docs)
    pipeline = CurationPipeline(engine=engine, embedder=embedder, settings=settings)
    decisions = [pipeline.ingest(item.candidate) for item in candidates]

    predicted_dedup = 0
    dedup_tp = 0
    actual_conflicts = 0
    conflict_tp = 0
    for item, decision in zip(candidates, decisions, strict=True):
        is_dup_label = item.label in {"exact_duplicate", "near_duplicate"}
        is_dup_pred = decision.action in {"skip_duplicate", "merge"}
        predicted_dedup += is_dup_pred
        dedup_tp += is_dup_pred and is_dup_label
        is_conflict_label = item.label.endswith("contradiction")
        is_conflict_pred = decision.action in {"supersede", "reject_conflict"}
        actual_conflicts += is_conflict_label
        conflict_tp += is_conflict_pred and is_conflict_label

    dedup_precision = dedup_tp / predicted_dedup
    contradiction_recall = conflict_tp / actual_conflicts
    actions = {d.action for d in decisions}
    assert predicted_dedup == 20
    assert actual_conflicts == 20
    assert dedup_precision >= 0.9
    assert contradiction_recall >= 0.9
    assert actions == {"accept", "merge", "skip_duplicate", "supersede"}
    assert pipeline.semantic.count() == 40
    with engine.connect() as conn:
        heads = conn.execute(
            text("SELECT count(*) FROM memory_entries WHERE status='active'")
        ).scalar_one()
    assert heads == 20


def test_corpus_is_deterministic():
    docs_a, candidates_a = generate_corpus(seed=11)
    docs_b, candidates_b = generate_corpus(seed=11)
    assert [d.model_dump() for d in docs_a] == [d.model_dump() for d in docs_b]
    assert [c.candidate.model_dump() for c in candidates_a] == [
        c.candidate.model_dump() for c in candidates_b
    ]
    labels = [c.label for c in build_candidates(11)]
    assert labels.count("unique_fact") == 20
    assert labels.count("exact_duplicate") == 10
    assert labels.count("near_duplicate") == 10
    assert labels.count("numeric_contradiction") == 10
    assert labels.count("negation_contradiction") == 10


@pytest.mark.parametrize("seed", [3, 7])
def test_contradictions_all_detected_for_seeds(engine, settings, embedder, seed):
    docs, candidates = generate_corpus(seed=seed)
    insert_source_docs(engine, docs)
    pipeline = CurationPipeline(engine=engine, embedder=embedder, settings=settings)
    decisions = [pipeline.ingest(item.candidate) for item in candidates]
    conflicts = [
        d.action
        for item, d in zip(candidates, decisions, strict=True)
        if item.label.endswith("contradiction")
    ]
    assert conflicts.count("supersede") == 20
