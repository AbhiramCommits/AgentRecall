"""Shared pytest fixtures: real Postgres + Redis, schema reset, deterministic embedder/LLM."""

from __future__ import annotations

import os

import pytest
import redis as redis_lib
from sqlalchemy import text

from agentrecall.config import Settings, configure
from agentrecall.db import get_engine, healthcheck, reset_schema, set_engine
from agentrecall.embeddings import HashingEmbedder, set_embedder

PG_DSN = os.environ.get(
    "AGENTRECALL_TEST_PG_DSN",
    "postgresql+psycopg://agentrecall:agentrecall@localhost:5433/agentrecall",
)
REDIS_URL = os.environ.get("AGENTRECALL_TEST_REDIS_URL", "redis://localhost:6380/1")

TABLES = "curation_log, resolution_traces, episodic_summaries, memory_entries, source_docs"


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Settings pinned to the docker-compose ports, with Redis db 1 for tests."""
    s = Settings(pg_dsn=PG_DSN, redis_url=REDIS_URL)
    configure(s)
    set_embedder(HashingEmbedder(dim=s.embed_dim))
    return s


@pytest.fixture(scope="session")
def engine(settings: Settings):
    """Session engine; skips the whole suite with a clear message if services are down."""
    engine = get_engine(settings, refresh=True)
    status = healthcheck(engine, settings)
    if not status["postgres"] or not status["redis"]:
        pytest.skip(
            "AgentRecall integration tests need PostgreSQL+pgvector and Redis. "
            "Start them with `docker compose up -d` (ports 5433/6380). "
            f"Current status: {status}"
        )
    yield engine
    engine.dispose()
    set_engine(None)


@pytest.fixture(scope="module", autouse=True)
def _schema_per_module(engine):
    reset_schema(engine)
    yield


@pytest.fixture(autouse=True)
def clean_db(engine, _schema_per_module):
    """Truncate all tables before each test for isolation."""
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {TABLES} CASCADE"))
    yield


@pytest.fixture()
def redis_client(settings: Settings):
    """A flushed Redis client on db 1."""
    client = redis_lib.Redis.from_url(settings.redis_url, decode_responses=True)
    client.flushdb()
    yield client
    client.flushdb()
    client.close()


@pytest.fixture(scope="session")
def embedder(settings: Settings) -> HashingEmbedder:
    return HashingEmbedder(dim=settings.embed_dim)
