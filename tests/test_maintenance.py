"""Maintenance tests: all stale causes, re-embed idempotence, audit kinds, compaction."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from agentrecall.data.corpus import insert_source_docs
from agentrecall.maintenance import (
    audit,
    compact,
    detect_stale,
    reembed_changed,
    write_report,
)
from agentrecall.memory.semantic import SemanticMemory
from agentrecall.models import EntryStatus, MemoryRecord, MemoryTier, SourceDoc, new_id

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
BODIES = {
    "doc-maint-1": (
        "Resist stack M dose qualification: exposure dose 31 mJ/cm^2 at 193 nm with "
        "numerical aperture 1.35 and depth of focus 80 nm on the immersion scanner."
    ),
    "doc-maint-2": (
        "Resist stack N dose qualification: exposure dose 33 mJ/cm^2 at 193 nm with "
        "numerical aperture 1.35 and depth of focus 80 nm on the immersion scanner."
    ),
    "doc-maint-3": (
        "OPC model note: calibration converged after six iterations with edge placement "
        "error 1.4 nm on the production layout."
    ),
    "doc-maint-4": (
        "Job deck note: layer M7 uses three exposure passes and a dose monitor every "
        "five lots on the 193i scanner."
    ),
}


def _hash(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()


@pytest.fixture(autouse=True)
def _seed_sources(engine, clean_db):
    docs = [
        SourceDoc(
            id=doc_id,
            uri=f"https://litho.example/{doc_id}",
            title=doc_id,
            body=body,
            content_hash=_hash(body),
            updated_at=NOW - timedelta(days=40),
            tags=["maintenance"],
        )
        for doc_id, body in BODIES.items()
    ]
    insert_source_docs(engine, docs)
    yield


def _add_entry(
    engine,
    doc_id: str | None,
    subject: str,
    claim: str,
    *,
    body: str | None = None,
    status: EntryStatus = EntryStatus.ACTIVE,
    last_used_at: datetime = NOW,
    created_at: datetime | None = None,
    access_count: int = 0,
    baseline_hash: bool = True,
    supersedes: str | None = None,
) -> str:
    provenance = {"sources": [doc_id] if doc_id else [], "origin": "maintenance-test"}
    if doc_id:
        provenance["source_doc_id"] = doc_id
        if baseline_hash:
            provenance["source_content_hash"] = _hash(BODIES.get(doc_id, body or ""))
    record = MemoryRecord(
        tier=MemoryTier.SEMANTIC,
        subject=subject,
        claim=claim,
        body=body or f"{subject} {claim}",
        source_doc_id=doc_id,
        status=status,
        last_used_at=last_used_at,
        created_at=created_at or last_used_at,
        access_count=access_count,
        supersedes=supersedes,
        provenance=provenance,
        confidence=0.8,
        session_id="session-maint",
    )
    return SemanticMemory(engine=engine).write(record)


def _status(engine, entry_id: str) -> str:
    with engine.connect() as conn:
        return conn.execute(
            text("SELECT status FROM memory_entries WHERE id = CAST(:id AS uuid)"),
            {"id": entry_id},
        ).scalar_one()


# ------------------------------------------------------------------ staleness
def test_detect_stale_by_age(engine, settings):
    entry_id = _add_entry(
        engine,
        "doc-maint-1",
        "resist stack M dose",
        "use dose 31 mj/cm2",
        last_used_at=NOW - timedelta(days=200),
    )
    result = detect_stale(engine=engine, settings=settings, now=NOW)
    assert result.marked_stale == 1
    assert result.reasons == {"age": 1}
    assert _status(engine, entry_id) == "stale"
    again = detect_stale(engine=engine, settings=settings, now=NOW)
    assert again.marked_stale == 0


def test_detect_stale_by_changed_source(engine, settings):
    entry_id = _add_entry(
        engine, "doc-maint-1", "resist stack M dose", "use dose 31 mj/cm2"
    )
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE source_docs SET body = :body WHERE id = 'doc-maint-1'"),
            {"body": BODIES["doc-maint-1"] + " New re-qualification measurement."},
        )
    result = detect_stale(engine=engine, settings=settings, now=NOW)
    assert result.reasons.get("source_changed") == 1
    assert _status(engine, entry_id) == "stale"


def test_detect_stale_by_missing_source(engine, settings):
    entry_id = _add_entry(
        engine, "doc-maint-4", "job deck M7", "use three exposure passes"
    )
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM source_docs WHERE id = 'doc-maint-4'"))
    result = detect_stale(engine=engine, settings=settings, now=NOW)
    assert result.reasons.get("missing_source") == 1
    assert _status(engine, entry_id) == "stale"
    with engine.connect() as conn:
        assert conn.execute(
            text("SELECT source_doc_id FROM memory_entries WHERE id = CAST(:id AS uuid)"),
            {"id": entry_id},
        ).scalar_one() is None


# ------------------------------------------------------------------ re-embedding
def test_reembed_changed_is_idempotent(engine, settings):
    changed_entry = _add_entry(
        engine, "doc-maint-1", "resist stack M dose", "use dose 31 mj/cm2"
    )
    unchanged_entry = _add_entry(
        engine, "doc-maint-3", "opc calibration", "converged after six iterations"
    )
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE source_docs SET body = :body WHERE id = 'doc-maint-1'"),
            {"body": BODIES["doc-maint-1"] + " Updated source measurement."},
        )

    first = reembed_changed(engine=engine, settings=settings)
    assert first.scanned_documents == 4
    assert first.changed_documents == 1
    assert first.reembedded_entries == 1
    assert first.skipped_documents == 3
    with engine.connect() as conn:
        prov = conn.execute(
            text(
                "SELECT provenance FROM memory_entries WHERE id = CAST(:id AS uuid)"
            ),
            {"id": changed_entry},
        ).scalar_one()
        unchanged = conn.execute(
            text(
                "SELECT provenance FROM memory_entries WHERE id = CAST(:id AS uuid)"
            ),
            {"id": unchanged_entry},
        ).scalar_one()
    expected_hash = _hash(BODIES["doc-maint-1"] + " Updated source measurement.")
    assert json.loads(prov if isinstance(prov, str) else json.dumps(prov))[
        "source_content_hash"
    ] == expected_hash
    assert json.loads(unchanged if isinstance(unchanged, str) else json.dumps(unchanged))[
        "source_content_hash"
    ] == _hash(BODIES["doc-maint-3"])

    second = reembed_changed(engine=engine, settings=settings)
    assert second.changed_documents == 0
    assert second.reembedded_entries == 0
    assert second.skipped_documents == 4


# ------------------------------------------------------------------------ audit
def test_audit_detects_orphaned_entry(engine, settings):
    _add_entry(engine, "doc-maint-4", "job deck M7", "use three exposure passes")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM source_docs WHERE id = 'doc-maint-4'"))
    result = audit(engine=engine, settings=settings, now=NOW)
    assert result.counts.get("orphaned_entry", 0) >= 1


def test_audit_detects_contradictory_active_pair(engine, settings):
    body = BODIES["doc-maint-1"]
    _add_entry(
        engine, "doc-maint-1", "resist stack M dose",
        "use dose 31 mj/cm2 for resist stack M", body=body,
    )
    _add_entry(
        engine, "doc-maint-1", "resist stack M dose",
        "use dose 35 mj/cm2 for resist stack M", body=body,
    )
    result = audit(engine=engine, settings=settings, now=NOW)
    assert result.counts.get("contradictory_active_pair") == 1


def test_audit_detects_multiple_active_heads(engine, settings):
    for _ in range(2):
        _add_entry(
            engine,
            "doc-maint-2",
            "resist stack N dose",
            "use dose 33 mj/cm2 for resist stack N",
        )
    result = audit(engine=engine, settings=settings, now=NOW)
    assert result.counts.get("multiple_active_heads") == 1


def test_audit_detects_dangling_supersede(engine, settings):
    entry_id = new_id()
    with engine.begin() as conn:
        conn.execute(text("SET LOCAL session_replication_role = replica"))
        conn.execute(
            text(
                "INSERT INTO memory_entries "
                "(id, tier, subject, claim, body, version, status, supersedes, provenance, "
                "confidence, access_count, created_at, last_used_at, content_hash) "
                "VALUES (CAST(:id AS uuid), 'semantic', 'dangling', 'dangling claim', "
                "'body', 1, 'active', CAST(:sup AS uuid), '{}', 0.5, 0, :now, :now, '')"
            ),
            {"id": entry_id, "sup": new_id(), "now": NOW},
        )
    result = audit(engine=engine, settings=settings, now=NOW)
    assert result.counts.get("dangling_supersede") == 1


def test_audit_detects_zero_usage_old(engine, settings):
    _add_entry(
        engine,
        "doc-maint-1",
        "unused fact",
        "use dose 31 mj/cm2",
        access_count=0,
        last_used_at=NOW - timedelta(days=90),
        created_at=NOW - timedelta(days=90),
    )
    result = audit(engine=engine, settings=settings, now=NOW)
    assert result.counts.get("zero_usage_old", 0) >= 1


# --------------------------------------------------------------------- compact
def test_compact_retires_only_stale_unused_superseded(engine, settings):
    ancestor = _add_entry(
        engine,
        "doc-maint-1",
        "resist stack M dose",
        "use dose 31 mj/cm2",
        status=EntryStatus.STALE,
        access_count=0,
    )
    head = _add_entry(
        engine,
        "doc-maint-1",
        "resist stack M dose",
        "use dose 35 mj/cm2",
        supersedes=ancestor,
    )
    stale_head = _add_entry(
        engine,
        "doc-maint-2",
        "resist stack N dose",
        "use dose 33 mj/cm2",
        status=EntryStatus.STALE,
        access_count=0,
    )
    result = compact(engine=engine, settings=settings)
    assert result.inspected == 2
    assert result.retired == 1
    assert _status(engine, ancestor) == "retired"
    assert _status(engine, head) == "active"
    assert _status(engine, stale_head) == "stale"
    assert compact(engine=engine, settings=settings).retired == 0


def test_compact_never_retires_active_heads(engine, settings):
    active = _add_entry(engine, "doc-maint-1", "active fact", "use dose 31 mj/cm2")
    active_superseding = _add_entry(
        engine,
        "doc-maint-1",
        "active fact",
        "use dose 36 mj/cm2",
        supersedes=active,
    )
    result = compact(engine=engine, settings=settings)
    assert result.retired == 0
    assert _status(engine, active) == "active"
    assert _status(engine, active_superseding) == "active"


# ---------------------------------------------------------------------- report
def test_audit_report_files_are_written(engine, settings, tmp_path):
    _add_entry(
        engine,
        "doc-maint-1",
        "orphan source",
        "use dose 31 mj/cm2",
        created_at=NOW - timedelta(days=90),
        last_used_at=NOW - timedelta(days=90),
    )
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM source_docs WHERE id = 'doc-maint-1'"))
    result = audit(engine=engine, settings=settings, now=NOW)
    json_path, md_path = write_report(result, out_dir=tmp_path)
    assert json_path.exists() and md_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["total_findings"] == len(result.findings)
    assert payload["counts"] == result.counts
    markdown = md_path.read_text()
    assert "Knowledge-Repository Audit" in markdown
    assert "Findings by type" in markdown
    assert markdown.count("|") > 10
