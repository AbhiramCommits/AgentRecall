"""MemoryStore facade: merges scratchpad, episodic, and semantic hits into one context."""

from __future__ import annotations

import contextlib

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from sqlalchemy.engine import Engine

from agentrecall.config import Settings, get_settings
from agentrecall.db import get_engine
from agentrecall.embeddings import get_embedder, tokenize
from agentrecall.llm import llm_factory
from agentrecall.memory.episodic import EpisodicMemory
from agentrecall.memory.scratchpad import Scratchpad
from agentrecall.memory.semantic import SemanticMemory
from agentrecall.models import MemoryHit, MemoryTier

_TIER_PRIORITY = {MemoryTier.SEMANTIC: 0, MemoryTier.EPISODIC: 1, MemoryTier.SCRATCH: 2}


def lexical_overlap(query: str, text: str) -> float:
    """Symmetric token overlap used to rank scratch turns that have no embedding."""
    q = set(tokenize(query))
    t = set(tokenize(text))
    if not q or not t:
        return 0.0
    return len(q & t) / len(q | t)


class MemoryStore:
    """Wires the three tiers and produces one ranked, provenance-labelled context."""

    def __init__(
        self,
        engine: Engine | None = None,
        redis_client=None,
        embedder: Embeddings | None = None,
        llm: BaseChatModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.engine = engine or get_engine(self.settings)
        self.embedder = embedder or get_embedder()
        self.llm = llm or llm_factory(self.settings)
        self.scratchpad = Scratchpad(client=redis_client, settings=self.settings)
        self.episodic = EpisodicMemory(
            engine=self.engine, embedder=self.embedder, llm=self.llm, settings=self.settings
        )
        self.semantic = SemanticMemory(
            engine=self.engine, embedder=self.embedder, settings=self.settings
        )

    def recall(
        self, query: str, session_id: str, k: int = 5, scratch_window: int = 6
    ) -> list[MemoryHit]:
        """Merge recent scratch turns, episodic summaries, and semantic memories."""
        hits: list[MemoryHit] = []

        for entry in self.scratchpad.window(session_id, scratch_window):
            hits.append(
                MemoryHit(
                    tier=MemoryTier.SCRATCH,
                    id=f"scratch:{session_id}:{entry.created_at.isoformat()}",
                    text=f"{entry.role}: {entry.text}",
                    score=lexical_overlap(query, entry.text),
                    provenance={"session_id": session_id, "role": entry.role},
                )
            )

        for summary in self.episodic.search(query, k=k):
            hits.append(
                MemoryHit(
                    tier=MemoryTier.EPISODIC,
                    id=summary.id,
                    text=summary.summary,
                    score=max(0.0, summary.score),
                    provenance={
                        "session_id": summary.session_id,
                        "turns": [summary.start_turn, summary.end_turn],
                        "created_at": summary.created_at.isoformat(),
                    },
                )
            )

        for scored in self.semantic.search(query, k=k):
            record = scored.record
            provenance = {
                "source_doc_id": record.source_doc_id,
                "session_id": record.session_id,
                "version": record.version,
                "status": record.status.value,
                "confidence": record.confidence,
                **record.provenance,
            }
            hits.append(
                MemoryHit(
                    tier=MemoryTier.SEMANTIC,
                    id=record.id,
                    text=f"{record.subject}: {record.claim}",
                    score=scored.score,
                    provenance=provenance,
                )
            )

        hits.sort(key=lambda h: (-h.score, _TIER_PRIORITY[h.tier], h.id))
        return hits[: max(k, k * 3)]

    def context_block(self, hits: list[MemoryHit]) -> str:
        """Render hits as a compact context block with tier labels and provenance."""
        if not hits:
            return "(no relevant memory)"
        lines = []
        for hit in hits:
            source = hit.provenance.get("source_doc_id") or hit.provenance.get("session_id") or "-"
            lines.append(
                f"[{hit.tier.value} score={hit.score:.3f} source={source}] {hit.text}"
            )
        return "\n".join(lines)

    def close(self) -> None:
        """Close resources held by the facade (Redis connection only)."""
        with contextlib.suppress(Exception):
            self.scratchpad.client.close()
