"""Tiered memory tests: scratchpad bounds/TTL, episodic round-trip, scoring, recall merge."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from agentrecall.embeddings import cosine_similarity
from agentrecall.memory import EpisodicMemory, MemoryStore, Scratchpad, SemanticMemory
from agentrecall.memory.semantic import (
    decay_factor,
    recency_component,
    score_entry,
    usage_component,
)
from agentrecall.models import EntryStatus, MemoryRecord, MemoryTier

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)


def _record(
    subject: str,
    claim: str,
    *,
    tier: MemoryTier = MemoryTier.SEMANTIC,
    last_used_at: datetime = NOW,
    access_count: int = 0,
    created_at: datetime = NOW,
    status: EntryStatus = EntryStatus.ACTIVE,
    confidence: float = 0.8,
    session_id: str | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        tier=tier,
        subject=subject,
        claim=claim,
        body=f"{subject} {claim}",
        last_used_at=last_used_at,
        created_at=created_at,
        access_count=access_count,
        status=status,
        confidence=confidence,
        session_id=session_id,
    )


# ------------------------------------------------------------------ scratchpad
def test_scratchpad_append_and_window(redis_client):
    pad = Scratchpad(client=redis_client, ttl_seconds=3600, max_turns=50)
    pad.append("s1", "user", "first")
    pad.append("s1", "assistant", "second")
    pad.append("s1", "user", "third")
    window = pad.window("s1", 2)
    assert [e.text for e in window] == ["second", "third"]
    assert [e.role for e in window] == ["assistant", "user"]
    assert pad.length("s1") == 3


def test_scratchpad_trim_to_max_length(redis_client):
    pad = Scratchpad(client=redis_client, ttl_seconds=3600, max_turns=3)
    for i in range(6):
        pad.append("s2", "user", f"turn-{i}")
    assert pad.length("s2") == 3
    assert [e.text for e in pad.window("s2", 10)] == ["turn-3", "turn-4", "turn-5"]


def test_scratchpad_ttl_and_clear(redis_client):
    pad = Scratchpad(client=redis_client, ttl_seconds=120, max_turns=50)
    pad.append("s3", "user", "hello")
    assert 0 < pad.ttl("s3") <= 120
    pad.clear("s3")
    assert pad.length("s3") == 0
    assert pad.window("s3", 5) == []


# --------------------------------------------------------------------- episodic
def test_episodic_summarize_and_search_roundtrip(engine, settings, embedder):
    epi = EpisodicMemory(engine=engine, embedder=embedder, settings=settings)
    turns = [
        {"role": "user", "text": "Use exposure dose 32 mJ/cm2 for resist stack A."},
        {"role": "assistant", "text": "Recorded the dose constraint for stack A."},
    ]
    summary = epi.summarize_session("session-1", turns, start_turn=0, end_turn=1)
    assert summary.start_turn == 0 and summary.end_turn == 1
    assert "32" in summary.summary
    assert epi.count() == 1
    hits = epi.search("exposure dose for resist stack A", k=3)
    assert len(hits) == 1
    assert hits[0].id == summary.id
    assert hits[0].score > 0.2


def test_episodic_search_orders_by_similarity(engine, settings, embedder):
    epi = EpisodicMemory(engine=engine, embedder=embedder, settings=settings)
    epi.summarize_session("a", [{"role": "user", "text": "Numerical aperture is 1.35."}])
    epi.summarize_session("b", [{"role": "user", "text": "The reticle stage needs a new belt."}])
    hits = epi.search("numerical aperture value", k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


# --------------------------------------------------------------------- semantic
def test_semantic_write_and_search(engine, settings, embedder):
    sem = SemanticMemory(engine=engine, embedder=embedder, settings=settings)
    entry_id = sem.write(_record("na", "numerical aperture is 1.35"))
    assert sem.count() == 1
    hits = sem.search("numerical aperture", k=5, now=NOW)
    assert hits[0].record.id == entry_id
    assert hits[0].similarity > 0.5


def test_scoring_components_hand_computed(settings):
    # 28 days at a 14-day half-life is exactly two half-lives -> 0.25.
    assert math.isclose(recency_component(NOW - timedelta(days=28), NOW, 14.0), 0.25, rel_tol=1e-12)
    assert math.isclose(recency_component(NOW, NOW, 14.0), 1.0, rel_tol=1e-12)
    assert abs(recency_component(NOW - timedelta(days=200), NOW, 14.0)) < 1e-4
    assert math.isclose(usage_component(9, 9), 1.0, rel_tol=1e-12)
    assert math.isclose(usage_component(0, 9), 0.0, rel_tol=1e-12)
    assert decay_factor(NOW - timedelta(days=89), NOW, 90.0, 0.05) == 1.0
    assert decay_factor(NOW - timedelta(days=91), NOW, 90.0, 0.05) == 0.05


def test_score_entry_matches_hand_computation(settings):
    similarity = 0.8
    final, sim, recency, usage, decay = score_entry(
        similarity,
        NOW - timedelta(days=28),
        access_count=3,
        access_count_max=3,
        now=NOW,
        settings=settings,
    )
    expected = 0.6 * 0.8 + 0.25 * 0.25 + 0.15 * 1.0
    assert (sim, recency, usage, decay) == (0.8, 0.25, 1.0, 1.0)
    assert math.isclose(final, expected, rel_tol=1e-12)


def test_semantic_search_matches_hand_computed_scores(engine, settings, embedder):
    sem = SemanticMemory(engine=engine, embedder=embedder, settings=settings)
    sem.write(_record("aperture", "aperture 1.35", last_used_at=NOW - timedelta(days=28),
                      access_count=3))
    sem.write(_record("resist", "resist dose 32", last_used_at=NOW, access_count=0))
    hits = sem.search("aperture 1.35", k=2, now=NOW)
    by_subject = {h.record.subject: h for h in hits}
    a = by_subject["aperture"]
    b = by_subject["resist"]
    expected_sim = cosine_similarity(
        embedder.embed_query("aperture 1.35"), embedder.embed_query("aperture aperture 1.35")
    )
    expected_a = 0.6 * expected_sim + 0.25 * 0.25 + 0.15 * 1.0
    assert math.isclose(a.similarity, expected_sim, rel_tol=1e-6)
    assert math.isclose(a.score, expected_a, rel_tol=1e-6)
    assert math.isclose(b.recency, 1.0, rel_tol=1e-12)
    assert a.score > b.score


def test_touch_lifts_decayed_entry_above_fresh_irrelevant(engine, settings, embedder):
    sem = SemanticMemory(engine=engine, embedder=embedder, settings=settings)
    old = _record("dose", "dose 32", last_used_at=NOW - timedelta(days=120))
    fresh = _record("stage", "stage belt tension 4 N", last_used_at=NOW)
    old_id = sem.write(old)
    sem.write(fresh)
    before = sem.search("dose 32", k=5, now=NOW)
    old_before = next(h for h in before if h.record.id == old_id)
    assert old_before.decay == settings.decay_floor
    assert old_before.score < 0.1

    assert sem.touch([old_id], now=NOW) == 1
    after = sem.search("dose 32", k=5, now=NOW)
    old_after = next(h for h in after if h.record.id == old_id)
    fresh_after = next(h for h in after if h.record.subject == "stage")
    assert old_after.decay == 1.0
    assert old_after.score > fresh_after.score
    assert old_after.record.access_count == 1


def test_touch_is_idempotent_per_call_and_deduplicates(engine, settings, embedder):
    sem = SemanticMemory(engine=engine, embedder=embedder, settings=settings)
    entry_id = sem.write(_record("na", "numerical aperture 1.35"))
    assert sem.touch([entry_id, entry_id]) == 1
    assert sem.get(entry_id).access_count == 1


def test_tier_filtering(engine, settings, embedder):
    sem = SemanticMemory(engine=engine, embedder=embedder, settings=settings)
    sem.write(_record("semantic fact", "semantic body", tier=MemoryTier.SEMANTIC))
    sem.write(_record("episodic fact", "episodic body", tier=MemoryTier.EPISODIC))
    default = sem.search("fact", k=5, now=NOW)
    assert {h.tier for h in default} == {MemoryTier.SEMANTIC}
    only_episodic = sem.search("fact", k=5, tiers=[MemoryTier.EPISODIC], now=NOW)
    assert {h.tier for h in only_episodic} == {MemoryTier.EPISODIC}
    assert len(only_episodic) == 1


def test_inactive_entries_excluded(engine, settings, embedder):
    sem = SemanticMemory(engine=engine, embedder=embedder, settings=settings)
    active = sem.write(_record("dose", "dose 32"))
    superseded = sem.write(_record("dose", "dose 30"))
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE memory_entries SET status='superseded' WHERE id=CAST(:id AS uuid)"),
            {"id": superseded},
        )
    hits = sem.search("dose", k=5, now=NOW)
    assert [h.record.id for h in hits] == [active]


# ------------------------------------------------------------------------ store
def test_recall_merges_all_three_tiers_deterministically(engine, settings, embedder, redis_client):
    store = MemoryStore(engine=engine, redis_client=redis_client, embedder=embedder,
                        settings=settings)
    store.scratchpad.append("session-x", "user", "numeric aperture is 1.35")
    store.episodic.summarize_session(
        "session-0", [{"role": "user", "text": "Exposure dose 32 mJ/cm2 for stack A."}]
    )
    store.semantic.write(_record("resist stack A", "dose is 32 mJ/cm2", session_id="session-old"))
    hits = store.recall("dose for resist stack A", "session-x", k=5)
    assert {h.tier for h in hits} == {
        MemoryTier.SCRATCH,
        MemoryTier.EPISODIC,
        MemoryTier.SEMANTIC,
    }
    again = store.recall("dose for resist stack A", "session-x", k=5)
    assert [h.id for h in hits] == [h.id for h in again]
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    block = store.context_block(hits)
    assert "semantic" in block and "episodic" in block and "scratch" in block


def test_recall_empty_store_returns_empty(engine, settings, embedder, redis_client):
    store = MemoryStore(engine=engine, redis_client=redis_client, embedder=embedder,
                        settings=settings)
    assert store.recall("anything", "empty-session") == []
    assert store.context_block([]) == "(no relevant memory)"


@pytest.mark.parametrize("k", [1, 3, 5])
def test_recall_respects_k(engine, settings, embedder, redis_client, k):
    store = MemoryStore(engine=engine, redis_client=redis_client, embedder=embedder,
                        settings=settings)
    for i in range(6):
        store.semantic.write(_record(f"subject-{i}", f"claim-{i}", session_id="s"))
    hits = store.recall("claim", "s", k=k)
    assert len(hits) <= 3 * k
