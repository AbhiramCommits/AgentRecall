"""Database engine, schema lifecycle, and service health checks.

PostgreSQL 16 with pgvector is the only persistence layer for semantic, episodic, trace,
and curation data. Nothing here falls back to SQLite or in-process dictionaries.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from agentrecall.config import Settings, get_settings

_SCHEMA_PATH = Path(__file__).parent / "sql" / "schema.sql"
_engine: Engine | None = None


def get_engine(settings: Settings | None = None, refresh: bool = False) -> Engine:
    """Return a process-wide SQLAlchemy engine for the configured DSN."""
    global _engine
    if _engine is None or refresh:
        settings = settings or get_settings()
        _engine = create_engine(settings.pg_dsn, pool_pre_ping=True, future=True)
    return _engine


def set_engine(engine: Engine | None) -> None:
    """Install a custom engine (tests) or reset the singleton with ``None``."""
    global _engine
    _engine = engine


def schema_sql() -> str:
    """Return the full DDL script."""
    return _SCHEMA_PATH.read_text(encoding="utf-8")


def init_schema(engine: Engine | None = None) -> None:
    """Create the pgvector extension and all tables/indexes idempotently."""
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql(schema_sql())


def reset_schema(engine: Engine | None = None) -> None:
    """Drop and recreate the AgentRecall schema (destructive, for tests and demos)."""
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                "DROP TABLE IF EXISTS curation_log, resolution_traces, episodic_summaries, "
                "memory_entries, source_docs CASCADE"
            )
        )
    init_schema(engine)


def healthcheck(engine: Engine | None = None, settings: Settings | None = None) -> dict[str, bool]:
    """Ping PostgreSQL and Redis; return a boolean per service."""
    settings = settings or get_settings()
    engine = engine or get_engine(settings)
    result = {"postgres": False, "redis": False}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        result["postgres"] = True
    except Exception:
        result["postgres"] = False
    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2)
        result["redis"] = bool(client.ping())
        client.close()
    except Exception:
        result["redis"] = False
    return result
