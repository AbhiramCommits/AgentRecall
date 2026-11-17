"""AgentRecall command-line interface."""

from __future__ import annotations

import typer

from agentrecall.db import get_engine, healthcheck, init_schema

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


if __name__ == "__main__":
    app()
