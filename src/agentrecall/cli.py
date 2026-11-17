"""AgentRecall command-line interface."""

from __future__ import annotations

from collections import Counter

import typer
from sqlalchemy import text

from agentrecall.agent.session import AgentSession, session_traces
from agentrecall.config import get_settings
from agentrecall.curation.pipeline import CurationPipeline
from agentrecall.data.corpus import generate_corpus, insert_source_docs
from agentrecall.db import get_engine, healthcheck, init_schema
from agentrecall.embeddings import get_embedder
from agentrecall.maintenance.jobs import (
    audit,
    compact,
    detect_stale,
    reembed_changed,
)
from agentrecall.maintenance.report import write_report
from agentrecall.memory.semantic import SemanticMemory
from agentrecall.models import utcnow

app = typer.Typer(
    name="agentrecall",
    help="Persistent memory and knowledge curation for lithography engineering agents.",
    no_args_is_help=True,
)
maintain_app = typer.Typer(help="Knowledge-repository maintenance jobs.")
app.add_typer(maintain_app, name="maintain")

MUTATION_MARKER = " Revised measurement note: dose drift observed during re-qualification."


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
    pipeline = CurationPipeline(engine=engine, embedder=get_embedder(), settings=settings)
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


@app.command("mutate-sources")
def mutate_sources() -> None:
    """Deterministically edit a fixed subset of source docs without refreshing their hashes."""
    engine = get_engine(get_settings())
    init_schema(engine)
    with engine.connect() as conn:
        ids = [
            str(row[0])
            for row in conn.execute(
                text(
                    "SELECT id FROM source_docs WHERE id LIKE 'doc-dose-%' "
                    "ORDER BY id LIMIT 3"
                )
            )
        ]
    if not ids:
        typer.echo("no source documents found; run ingest-corpus first")
        raise typer.Exit(code=1)
    touched: list[str] = []
    with engine.begin() as conn:
        for doc_id in ids:
            body = conn.execute(
                text("SELECT body FROM source_docs WHERE id = :id"), {"id": doc_id}
            ).scalar_one()
            if MUTATION_MARKER.strip() in body:
                continue
            conn.execute(
                text(
                    "UPDATE source_docs SET body = :body, updated_at = :updated_at WHERE id = :id"
                ),
                {"body": body + MUTATION_MARKER, "updated_at": utcnow(), "id": doc_id},
            )
            touched.append(doc_id)
    typer.echo(f"mutated source documents: {len(touched)}")
    for doc_id in touched:
        typer.echo(f"  {doc_id}")


@app.command()
def ask(
    request: str = typer.Argument(..., help="Natural-language engineering request."),
    session: str = typer.Option("S1", help="Session identifier."),
    memory: bool = typer.Option(True, "--memory/--no-memory", help="Enable memory recall/write."),
) -> None:
    """Ask the agent a request and print the answer plus its resolution trace."""
    agent = AgentSession(session, memory_enabled=memory)
    answer = agent.ask(request)
    agent.close()
    typer.echo(answer)
    typer.echo("")
    trace = agent.get_trace()
    if trace:
        _print_trace(trace)


@app.command()
def trace(session: str = typer.Option("S1", help="Session identifier to replay.")) -> None:
    """Replay all stored resolution traces for a session."""
    rows = session_traces(get_engine(), session)
    if not rows:
        typer.echo(f"no traces for session {session}")
        raise typer.Exit(code=1)
    for row in rows:
        _print_trace(row)
        typer.echo("")


def _print_trace(trace: dict) -> None:
    hits = trace.get("memory_hits") or []
    if isinstance(hits, str):
        import json

        hits = json.loads(hits)
    steps = trace.get("steps") or []
    if isinstance(steps, str):
        import json

        steps = json.loads(steps)
    typer.echo(
        f"trace {trace['id']} | intent={trace['intent']} | tool={trace['chosen_tool']} | "
        f"outcome={trace['outcome']} | {trace['latency_ms']} ms"
    )
    for index, step in enumerate(steps, start=1):
        typer.echo(f"  {index}. {step.get('node')}: {step.get('detail')}")
    if hits:
        typer.echo("  memory hits:")
        for hit in hits:
            typer.echo(
                f"    [{hit['tier']} {hit['score']:.3f}] {hit.get('text', '')[:120]}"
            )
    else:
        typer.echo("  memory hits: none")


@maintain_app.command("stale")
def maintain_stale() -> None:
    """Mark entries STALE by age, changed source content, or a missing source."""
    result = detect_stale()
    typer.echo(
        f"scanned={result.scanned} marked_stale={result.marked_stale} "
        f"stale_total={result.stale_total} reasons={result.reasons}"
    )


@maintain_app.command("reembed")
def maintain_reembed() -> None:
    """Re-embed only entries whose source document body changed."""
    result = reembed_changed()
    typer.echo(
        f"scanned_documents={result.scanned_documents} "
        f"changed_documents={result.changed_documents} "
        f"reembedded_entries={result.reembedded_entries} "
        f"skipped_documents={result.skipped_documents}"
    )


@maintain_app.command("audit")
def maintain_audit() -> None:
    """Audit the repository and write results/audit_report.{json,md}."""
    result = audit()
    json_path, md_path = write_report(result)
    typer.echo(f"entries_scanned={result.scanned} findings={len(result.findings)}")
    for kind, count in result.counts.items():
        typer.echo(f"  {kind:<28} {count:>5}")
    typer.echo(f"wrote {json_path} and {md_path}")


@maintain_app.command("compact")
def maintain_compact() -> None:
    """Retire stale, unused entries that have a surviving superseding version."""
    result = compact()
    typer.echo(
        f"inspected={result.inspected} retired={result.retired} skipped={result.skipped}"
    )


@maintain_app.command("all")
def maintain_all() -> None:
    """Run staleness, re-embedding, compaction, and audit in order."""
    stale = detect_stale()
    reembed = reembed_changed()
    compacted = compact()
    result = audit()
    json_path, md_path = write_report(result)
    typer.echo(
        f"stale: scanned={stale.scanned} marked={stale.marked_stale} "
        f"reasons={stale.reasons}"
    )
    typer.echo(
        f"reembed: scanned={reembed.scanned_documents} "
        f"changed={reembed.changed_documents} entries={reembed.reembedded_entries}"
    )
    typer.echo(f"compact: inspected={compacted.inspected} retired={compacted.retired}")
    typer.echo(f"audit: entries={result.scanned} findings={len(result.findings)}")
    for kind, count in result.counts.items():
        typer.echo(f"  {kind:<28} {count:>5}")
    typer.echo(f"wrote {json_path} and {md_path}")


if __name__ == "__main__":
    app()
