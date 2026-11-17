"""Episodic memory: LLM-summarized session spans stored with embeddings in Postgres."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from sqlalchemy import text
from sqlalchemy.engine import Engine

from agentrecall.config import Settings, get_settings
from agentrecall.db import get_engine
from agentrecall.embeddings import get_embedder, vector_literal
from agentrecall.llm import llm_factory, summarize_turns
from agentrecall.models import EpisodicSummary


def _normalize_turns(turns: list[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for turn in turns:
        if isinstance(turn, dict):
            normalized.append(
                {"role": str(turn.get("role", "user")), "text": str(turn.get("text", ""))}
            )
        elif hasattr(turn, "role") and hasattr(turn, "text"):
            normalized.append({"role": str(turn.role), "text": str(turn.text)})
        else:
            normalized.append({"role": "user", "text": str(turn)})
    return normalized


class EpisodicMemory:
    """Stores and searches compact summaries of past sessions."""

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

    def summarize_session(
        self,
        session_id: str,
        turns: list[Any],
        start_turn: int | None = None,
        end_turn: int | None = None,
    ) -> EpisodicSummary:
        """Summarize a turn span with the LLM, embed it, and persist it."""
        normalized = _normalize_turns(turns)
        summary_text = summarize_turns(normalized, self.llm)
        vector = self.embedder.embed_query(summary_text)
        summary = EpisodicSummary(
            session_id=session_id,
            start_turn=0 if start_turn is None else start_turn,
            end_turn=len(normalized) - 1 if end_turn is None else end_turn,
            summary=summary_text,
            embedding=vector,
        )
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO episodic_summaries
                        (id, session_id, start_turn, end_turn, summary, embedding, created_at)
                    VALUES
                        (CAST(:id AS uuid), :session_id, :start_turn, :end_turn, :summary,
                         CAST(:embedding AS vector), :created_at)
                    """
                ),
                {
                    "id": summary.id,
                    "session_id": summary.session_id,
                    "start_turn": summary.start_turn,
                    "end_turn": summary.end_turn,
                    "summary": summary.summary,
                    "embedding": vector_literal(vector),
                    "created_at": summary.created_at,
                },
            )
        return summary

    def search(self, query: str, k: int = 5) -> list[EpisodicSummary]:
        """Cosine-search summaries, returning highest similarity first."""
        vector = self.embedder.embed_query(query)
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id, session_id, start_turn, end_turn, summary,
                           embedding::text AS embedding_text, created_at,
                           1 - (embedding <=> CAST(:vec AS vector)) AS similarity
                    FROM episodic_summaries
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(:vec AS vector), id
                    LIMIT :k
                    """
                ),
                {"vec": vector_literal(vector), "k": k},
            ).mappings()
            results: list[EpisodicSummary] = []
            for row in rows:
                emb = json.loads(row["embedding_text"]) if row["embedding_text"] else None
                results.append(
                    EpisodicSummary(
                        id=str(row["id"]),
                        session_id=row["session_id"],
                        start_turn=row["start_turn"],
                        end_turn=row["end_turn"],
                        summary=row["summary"],
                        embedding=emb,
                        score=float(row["similarity"]),
                        created_at=row["created_at"],
                    )
                )
        return results

    def count(self) -> int:
        with self.engine.connect() as conn:
            return int(conn.execute(text("SELECT count(*) FROM episodic_summaries")).scalar_one())
