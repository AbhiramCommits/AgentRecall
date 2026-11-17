"""Scaffold tests: settings, deterministic models, schema idempotence, service health."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

from sqlalchemy import text

from agentrecall.config import Settings
from agentrecall.db import healthcheck, init_schema
from agentrecall.embeddings import HashingEmbedder
from agentrecall.llm import StubChatModel

SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("AGENTRECALL_W_SIMILARITY", "0.71")
    monkeypatch.setenv("AGENTRECALL_STALE_AFTER_DAYS", "45")
    monkeypatch.setenv("AGENTRECALL_REDIS_URL", "redis://localhost:6380/7")
    loaded = Settings(_env_file=None)
    assert loaded.w_similarity == 0.71
    assert loaded.stale_after_days == 45.0
    assert loaded.redis_url == "redis://localhost:6380/7"
    assert loaded.embed_dim == 384


def test_embedder_shape_norm_and_determinism():
    emb = HashingEmbedder(dim=384)
    text = "OPC recipe for 193i immersion: numerical aperture 1.35"
    first = emb.embed_query(text)
    second = emb.embed_query(text)
    assert len(first) == 384
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0, rel_tol=1e-9)
    assert first == second
    assert emb.embed_query("a different sentence") != first


def test_embedder_is_process_independent():
    emb = HashingEmbedder(dim=384)
    local = emb.embed_query("dose 32 mJ/cm2 resist A")
    script = (
        "import json;"
        "from agentrecall.embeddings import HashingEmbedder;"
        "print(json.dumps(HashingEmbedder(dim=384).embed_query('dose 32 mJ/cm2 resist A')[:8]))"
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"},
    )
    remote = json.loads(out.stdout)
    assert remote == [round(v, 12) for v in local[:8]] or all(
        math.isclose(a, b, abs_tol=1e-9) for a, b in zip(remote, local[:8], strict=True)
    )


def test_stub_chat_model_deterministic():
    model = StubChatModel()
    first = model.invoke("[TASK:CLASSIFY]\nRequest: what is the best dose for resist A?")
    second = model.invoke("[TASK:CLASSIFY]\nRequest: what is the best dose for resist A?")
    assert first.content == second.content == "lookup_recipe_parameter"
    summary_one = model.invoke("[TASK:SUMMARIZE]\nuser: Use dose 32 mJ/cm2 for stack A.")
    summary_two = model.invoke("[TASK:SUMMARIZE]\nuser: Use dose 32 mJ/cm2 for stack A.")
    assert summary_one.content == summary_two.content
    assert "32" in summary_one.content


def test_init_schema_idempotent(engine):
    init_schema(engine)
    init_schema(engine)
    with engine.connect() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
            )
        }
    expected = {
        "source_docs",
        "memory_entries",
        "episodic_summaries",
        "resolution_traces",
        "curation_log",
    }
    assert expected <= tables
    with engine.connect() as conn:
        dims = conn.execute(
            text(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid='memory_entries'::regclass AND attname='embedding'"
            )
        ).scalar_one()
    assert dims == 384


def test_healthcheck_reports_both_services(engine, settings):
    status = healthcheck(engine, settings)
    assert status == {"postgres": True, "redis": True}
