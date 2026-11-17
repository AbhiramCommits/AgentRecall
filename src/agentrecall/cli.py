"""AgentRecall command-line interface."""

from __future__ import annotations

from collections import Counter

import typer
from sqlalchemy import text

from agentrecall.config import get_settings
from agentrecall.curation.pipeline import CurationPipeline
from agentrecall.data.corpus import generate_corpus, insert_source_docs
from agentrecall.db import get_engine, healthcheck, init_schema
from agentrecall.embeddings import get_embedder
from agentrecall.memory.semantic import SemanticMemory

app = typer.Typer(
    name="agentrecall",
    help="Persistent memory and knowledge curation for lithography engineering agents.",
    no_args_is_help=True,
)


@app.command("init-db")
def init_db() -> None:
    """Create the pgvector extension, tables, and indexes (idempotent)."""
    init_schema(get_engine())
    typer.echo("schema initialized")


@app.command()
def health() -> None:
    """Check that PostgreSQL and Redis are reachable."""
    status = healthcheck()
    for service, ok in status.items():
        typer.echo(f"{service}: {'OK' if ok else 'DOWN'}")
    if not all(status.values()):
        raise typer.Exit(code=1)


@app.command("ingest-corpus")
def ingest_corpus(
    seed: int = typer.Option(7, help="Deterministic corpus seed."),
    curate: bool = typer.Option(True, "--curate/--no-curate", help="Run the curation pipeline."),
    reset: bool = typer.Option(False, "--reset", help="Truncate memory and sources first."),
) -> None:
    """Generate the synthetic lithography corpus and ingest its candidate stream."""
    settings = get_settings()
    engine = get_engine(settings)
    init_schema(engine)
    if reset:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "TRUNCATE curation_log, resolution_traces, episodic_summaries, "
                    "memory_entries, source_docs CASCADE"
                )
            )

    docs, candidates = generate_corpus(seed)
    written = insert_source_docs(engine, docs)
    typer.echo(f"source documents: {written}")

    semantic = SemanticMemory(engine=engine, embedder=get_embedder(), settings=settings)
    pipeline = CurationPipeline(
        engine=engine, embedder=get_embedder(), settings=settings
    )
    counts: Counter[str] = Counter()
    for item in candidates:
        if curate:
            decision = pipeline.ingest(item.candidate)
        else:
            semantic.write(item.candidate)
            decision = None
        counts[decision.action if decision else "raw_insert"] += 1

    typer.echo(f"candidates: {len(candidates)}")
    typer.echo(f"{'action':<18} {'count':>6}")
    typer.echo(f"{'-' * 18} {'-' * 6}")
    for action, count in sorted(counts.items()):
        typer.echo(f"{action:<18} {count:>6}")
    typer.echo(f"{'total entries':<18} {semantic.count():>6}")


if __name__ == "__main__":
    app()
