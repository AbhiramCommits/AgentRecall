"""AgentSession: a stateful conversation facade over the LangGraph resolution workflow."""

from __future__ import annotations

import time

from langchain_core.language_models.chat_models import BaseChatModel
from sqlalchemy import text

from agentrecall.agent.graph import AgentGraph, AgentState
from agentrecall.config import Settings, get_settings
from agentrecall.llm import llm_factory
from agentrecall.memory.store import MemoryStore


class AgentSession:
    """One conversation, optionally ablated to disable all memory reads and writes."""

    def __init__(
        self,
        session_id: str,
        memory_enabled: bool = True,
        store: MemoryStore | None = None,
        llm: BaseChatModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.session_id = session_id
        self.memory_enabled = memory_enabled
        self.settings = settings or get_settings()
        self.store = store or MemoryStore(
            settings=self.settings,
            llm=llm or llm_factory(self.settings),
            embedder=None,
        )
        self.agent = AgentGraph(self.store, llm=llm, settings=self.settings)
        self.turn_count = 0
        self.answer = ""
        self.last_state: AgentState | None = None
        self.last_trace_id: str | None = None
        self._closed = False

    def ask(self, text_value: str) -> str:
        """Resolve one natural-language request and return the answer."""
        self.turn_count += 1
        state: AgentState = {
            "request": text_value,
            "session_id": self.session_id,
            "memory_enabled": self.memory_enabled,
            "turn_count": self.turn_count,
            "steps": [],
            "recall_loops": 0,
            "started_at": time.perf_counter(),
        }
        result = self.agent.invoke(state)
        self.last_state = result
        self.last_trace_id = result.get("trace_id")
        self.answer = result.get("answer", "")
        return self.answer

    def close(self) -> None:
        """Write a final episodic summary of the session when memory is enabled."""
        if self._closed:
            return
        self._closed = True
        if not self.memory_enabled or self.turn_count == 0:
            return
        turns = self.store.scratchpad.window(self.session_id, self.settings.scratch_max_turns)
        if turns:
            self.store.episodic.summarize_session(
                self.session_id, turns, 0, len(turns) - 1
            )

    def get_trace(self) -> dict | None:
        """Return the last persisted resolution trace as a mapping."""
        if not self.last_trace_id:
            return None
        with self.store.engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM resolution_traces WHERE id = CAST(:id AS uuid)"),
                {"id": self.last_trace_id},
            ).mappings().first()
        return dict(row) if row else None


def session_traces(engine, session_id: str) -> list[dict]:
    """Return every persisted trace for a session in creation order."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, session_id, request, intent, chosen_tool, memory_hits, steps, "
                "outcome, latency_ms, created_at FROM resolution_traces "
                "WHERE session_id = :session ORDER BY created_at, id"
            ),
            {"session": session_id},
        ).mappings()
        return [dict(row) for row in rows]
