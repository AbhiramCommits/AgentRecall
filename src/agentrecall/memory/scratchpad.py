"""Short-term memory: a bounded, expiring Redis list of recent turns per session."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import redis as redis_lib

from agentrecall.config import Settings, get_settings
from agentrecall.models import ScratchpadEntry


class Scratchpad:
    """Redis-backed scratchpad with a per-session TTL and a hard turn cap."""

    def __init__(
        self,
        client: redis_lib.Redis | None = None,
        settings: Settings | None = None,
        ttl_seconds: int | None = None,
        max_turns: int | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or redis_lib.Redis.from_url(
            self.settings.redis_url, decode_responses=True
        )
        self.ttl_seconds = ttl_seconds or self.settings.scratch_ttl_seconds
        self.max_turns = max_turns or self.settings.scratch_max_turns

    @staticmethod
    def _key(session_id: str) -> str:
        return f"agentrecall:scratch:{session_id}"

    def append(self, session_id: str, role: str, text: str) -> ScratchpadEntry:
        """Append one turn, trim to the newest ``max_turns``, and refresh the TTL."""
        entry = ScratchpadEntry(session_id=session_id, role=role, text=text)
        payload = json.dumps(
            {
                "session_id": session_id,
                "role": role,
                "text": text,
                "created_at": entry.created_at.isoformat(),
            }
        )
        key = self._key(session_id)
        pipe = self.client.pipeline()
        pipe.lpush(key, payload)
        pipe.ltrim(key, 0, self.max_turns - 1)
        pipe.expire(key, self.ttl_seconds)
        pipe.execute()
        return entry

    def window(self, session_id: str, n: int = 6) -> list[ScratchpadEntry]:
        """Return up to ``n`` most recent turns in chronological order."""
        if n <= 0:
            return []
        raw = self.client.lrange(self._key(session_id), 0, n - 1)
        entries: list[ScratchpadEntry] = []
        for item in raw:
            if isinstance(item, bytes):
                item = item.decode("utf-8")
            data = json.loads(item)
            created = data.get("created_at")
            entries.append(
                ScratchpadEntry(
                    session_id=data.get("session_id", session_id),
                    role=data.get("role", "user"),
                    text=data.get("text", ""),
                    created_at=datetime.fromisoformat(created) if created else datetime.now(UTC),
                )
            )
        entries.reverse()
        return entries

    def clear(self, session_id: str) -> None:
        """Delete the session scratchpad immediately."""
        self.client.delete(self._key(session_id))

    def ttl(self, session_id: str) -> int:
        """Remaining TTL in seconds (-2 when the key is absent)."""
        return int(self.client.ttl(self._key(session_id)))

    def length(self, session_id: str) -> int:
        """Number of stored turns (before windowing)."""
        return int(self.client.llen(self._key(session_id)))
