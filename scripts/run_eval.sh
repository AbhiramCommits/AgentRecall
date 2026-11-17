#!/usr/bin/env bash
# Reproduce the AgentRecall demo and measured evaluation from a clean checkout.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
CLI=("$PYTHON_BIN" -m agentrecall.cli)

echo "==> starting postgres + redis"
docker compose up -d --wait

echo "==> initializing schema"
"${CLI[@]}" init-db

echo "==> ingesting curated corpus"
"${CLI[@]}" ingest-corpus --curate --reset

echo "==> mutating source documents"
"${CLI[@]}" mutate-sources

echo "==> running maintenance jobs"
"${CLI[@]}" maintain all

echo "==> running evaluation"
"${CLI[@]}" eval run --seed 7
