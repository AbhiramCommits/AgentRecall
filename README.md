# AgentRecall

[![ci](https://github.com/abhiramkasireddi/AgentRecall/actions/workflows/ci.yml/badge.svg)](https://github.com/abhiramkasireddi/AgentRecall/actions/workflows/ci.yml)

AgentRecall is a long-term memory and knowledge-curation layer for engineering AI agents,
themed around computational-lithography support. It gives an agent persistent memory across
sessions instead of forcing it to re-retrieve the same know-how from a static corpus: a
tiered store keeps recent scratchpad turns, LLM-written episodic summaries, and embedded
semantic facts; a write-path curation pipeline deduplicates, merges, and version-resolves
contradictions with provenance; maintenance jobs detect stale knowledge, re-embed changed
sources, audit repository health, and compact dead versions; and a LangGraph intent-to-action
workflow resolves requests against memory, picks a lithography tool, executes it, and writes
a full resolution trace. The whole stack runs offline and deterministically with local hashed
embeddings and a stub chat model, while remaining pluggable for real models.

## Architecture

```
                        ┌──────────────────────────── LangGraph ────────────────────────────┐
 request ──────────────▶│ classify_intent ─▶ recall_memory ─▶ plan ──(missing slots, once)──┐│
                        │        ▲                                  │                       ││
                        │        └──────────────────────────────────┘                       ││
                        │                                           ▼                       ││
                        │              route_tool ─▶ execute ─▶ reflect_and_write ─▶ summarize_turn ─▶ END
                        └───────────────────────────────────────────────────────────────────┘
                                                   │                    │
                                                   │                    └── CurationPipeline.ingest (closes the loop)
                                                   ▼
      ┌────────────────────────────── three memory tiers ──────────────────────────────┐
      │ SCRATCH      Redis list per session, last 50 turns, 24 h TTL                   │
      │ EPISODIC     LLM session summaries, embedded, cosine search over summaries     │
      │ SEMANTIC     memory_entries + pgvector HNSW, recency/relevance/usage re-rank    │
      └────────────────────────────────────────────────────────────────────────────────┘
```

| Module | Responsibility |
| --- | --- |
| `memory/` | Redis scratchpad, episodic summary store, pgvector semantic store, and `MemoryStore.recall()` merge across tiers. |
| `curation/` | Normalize/canonicalize, exact and near dedup, contradiction adjudication, provenance union, versioned supersede lineage, and history. |
| `maintenance/` | Idempotent stale detection, selective re-embedding, repository audit, and compaction plus JSON/Markdown reports. |
| `agent/` | Five deterministic tools, the seven-node LangGraph workflow, write-back reflection, and stateful sessions with resolution traces. |
| `eval/` | Seeded 40-task three-session benchmark, labeled retrieval set, curation/maintenance/latency metrics, and the A–E harness. |

### Retrieval scoring

Every semantic candidate is retrieved in SQL with pgvector cosine distance, then re-ranked in
Python so the arithmetic is testable with fixed timestamps:

```
recency = exp(-ln2 * age_days / 14.0)                 # age since last_used_at
usage   = log1p(access_count) / log1p(max_access_count)
score   = (0.60 * cosine_similarity
         + 0.25 * recency
         + 0.15 * usage) * decay

decay   = 1.0      when age_days <= 90
        = 0.05     when age_days  > 90
```

`touch(ids)` increments `access_count` and sets `last_used_at = now`, which is how frequently
used knowledge escapes age-based decay. Weights, half-life (`14` days), decay floor (`0.05`),
and the stale horizon (`90` days) are all configurable through `AGENTRECALL_*` settings.
Retrieval excludes `SUPERSEDED` and `RETIRED` entries by default; `STALE` entries remain
searchable but are visibly marked by maintenance.

### Supersede, merge, and decay rules

- `ACTIVE` is the default status. Nothing is ever hard-deleted.
- A near-duplicate (`cosine >= 0.94`, same canonical subject, non-contradictory) is merged in
  place: the higher-confidence body wins, provenance lists are unioned, the earliest
  `created_at` is kept, and `version` increments.
- A contradiction (`cosine >= 0.86` plus a value or negation flip) is adjudicated by
  confidence first, then provenance recency. The winner becomes a new row with
  `version = loser.version + 1`, `supersedes = loser.id`; the loser becomes `SUPERSEDED`.
- A plain new fact is accepted as version 1.
- `history(entry_id)` walks the `supersedes` chain newest-to-oldest so every version and its
  provenance remain inspectable.
- Compaction only retires entries that are `STALE`, have zero usage, and already have a
  surviving superseding version. It never touches an `ACTIVE` head.

## Data model

`memory_entries` is the center of the design:

```
id uuid pk · tier · subject · claim · body · embedding vector(384)
source_doc_id → source_docs(id) · session_id · version · status
supersedes → memory_entries(id) · provenance jsonb · confidence
access_count · created_at · last_used_at · content_hash
```

`source_docs`, `episodic_summaries`, `resolution_traces`, and `curation_log` sit alongside it.
An HNSW index (`vector_cosine_ops`) supports cosine candidate retrieval, with b-tree indexes on
`status`, `subject`, and `session_id`. Every write decision is also logged to `curation_log`.

Provenance is a JSON object. Session-derived memories record `session_id`, source-grounded
memories record `source_doc_id` plus the source's `source_content_hash` at ingest time, and
merges/conflicts union the contributing lists. That source hash is what lets maintenance
detect that a document changed out from under a memory.

### Worked example: resolving a numeric contradiction

Seed 7's corpus contains two statements about resist stack A:

1. Version 1 — `"use dose 30 mj/cm2 for resist stack a at 193 nm ..."`, confidence `0.70`,
   source `doc-dose-00`.
2. Challenger — `"use dose 34 mj/cm2 for resist stack a ..."`, confidence `0.90`, provenance
   adds `measured_on: scanner-B`. The normalized claim cosine is `0.986`.

`detect_contradiction` extracts `30 mj/cm2` and `34 mj/cm2`, sees the same unit with different
values, and routes the pair to the adjudicator. The adjudicator prefers the higher-confidence
0.90 row. That row is inserted as version 2 with `supersedes` pointing at version 1, and
version 1 is marked `SUPERSEDED`. `history(version_2)` returns `[v2, v1]`; semantic search sees
only the active v2. The store grew by one row (history preserved) rather than overwriting or
deleting the old measurement.

## How to run

Prerequisites: Docker (Compose v2), Python 3.11, and a local checkout.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
docker compose up -d --wait
agentrecall init-db
agentrecall ingest-corpus --curate --reset
```

Ask the agent and inspect the resolution trace:

```bash
agentrecall ask "What is the qualified dose for resist stack A?" --session S1
agentrecall trace --session S1
```

Teach it something in one session and recall it in another:

```bash
agentrecall ask "Remember: pilot stack P99 uses a qualified dose of 44 mJ/cm2." --session demo-1
agentrecall ask "What is the qualified dose for pilot stack P99?" --session demo-2
```

Run maintenance and the evaluation:

```bash
agentrecall mutate-sources
agentrecall maintain all
agentrecall eval run --seed 7
```

Or run the complete demo end to end (starts Compose, initializes the schema, ingests the
corpus, mutates sources, runs maintenance, and evaluates):

```bash
bash scripts/run_eval.sh
```

### Swapping in a real embedder or LLM

The default is fully offline. `HashingEmbedder` implements LangChain's `Embeddings` interface
and produces deterministic 384-dimensional L2-normalized vectors; `StubChatModel` implements
`BaseChatModel` for summarization, adjudication, and intent classification.

To use an OpenAI-compatible chat endpoint, install the optional extra and set:

```bash
pip install -e ".[llm]"
export AGENTRECALL_LLM_BASE_URL="https://your-endpoint.example/v1"
export AGENTRECALL_LLM_MODEL="your-model"
export AGENTRECALL_LLM_API_KEY="..."
```

`llm_factory()` then returns `ChatOpenAI` instead of the stub. To replace embeddings, implement
the LangChain `Embeddings` interface and install it once at process start:

```python
from agentrecall.embeddings import set_embedder

set_embedder(MyRealEmbedder())  # must produce AGENTRECALL_EMBED_DIM (default 384) floats
```

A real model with a different vector width also requires changing `embed_dim` and migrating the
`vector(384)` columns/indexes; the hashed 384-dim space is the default schema contract.

## Results

All numbers below were measured by actually running this repository, seed `7`, on the build
machine. The canonical artifacts are `results/eval_results.json` and `results/eval_report.md`;
test results come from the full `pytest` run. Reproduce with:

```bash
bash scripts/run_eval.sh
# or, against an already-initialized database:
agentrecall eval run --seed 7
```

### A. Multi-session task success (memory ON vs OFF)

| Arm | Tasks | Successes | Success rate |
| --- | ---: | ---: | ---: |
| Memory ON | 40 | 40 | 1.000 |
| Memory OFF | 40 | 0 | 0.000 |
| Delta | 40 | - | +1.000 |

| Intent | Tasks | ON rate | OFF rate | Delta |
| --- | ---: | ---: | ---: | ---: |
| compute_process_window | 8 | 1.000 | 0.000 | +1.000 |
| diagnose_opc_convergence | 4 | 1.000 | 0.000 | +1.000 |
| draft_job_deck | 8 | 1.000 | 0.000 | +1.000 |
| lookup_recipe_parameter | 16 | 1.000 | 0.000 | +1.000 |
| summarize_prior_session | 4 | 1.000 | 0.000 | +1.000 |

### B. Retrieval metrics before vs after curation

| Store | Queries | Precision@5 | Recall@5 | MRR | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw ingest | 4 | 0.350 | 0.375 | 0.333 | 0.279 |
| After curation | 4 | 0.950 | 1.000 | 1.000 | 1.000 |

### C. Curation quality

| Metric | Value |
| --- | ---: |
| Dedup precision | 1.000 |
| Dedup recall | 1.000 |
| Contradiction precision | 1.000 |
| Contradiction recall | 1.000 |
| Contradiction F1 | 1.000 |
| Entries before curation | 60 |
| Entries after curation | 40 |
| Store reduction | 33.3% |

Decision counts: `accept`=20, `merge`=10, `skip_duplicate`=10, `supersede`=20

### D. Maintenance effects

| Metric | Value |
| --- | ---: |
| Source documents mutated | 3 |
| Stale entries detected | 3 |
| Stale reasons | {'source_changed': 3} |
| Re-embeds run 1 | 6 |
| Re-embeds run 2 | 0 |
| Documents changed run 1 | 3 |
| Documents changed run 2 | 0 |
| Compacted (retired) | 0 |
| Audit findings | 10 |

| Audit finding type | Count |
| --- | ---: |
| zero_usage_old | 10 |

### E. Latency

| Metric | Memory ON | Memory OFF |
| --- | ---: | ---: |
| ask p50 (ms) | 13.00 | 2.00 |
| ask p95 (ms) | 39.10 | 4.00 |
| asks measured | 120 | 120 |

Mean direct memory recall latency: **8.60 ms** (12 samples).

### Test suite

- **69 tests passed** in the full suite.
- **88.93% line coverage** over `src/agentrecall` (reported by
  `pytest --cov=src/agentrecall --cov-report=term-missing`).
- `ruff check .` is clean.

The evaluated timestamp is recorded in `results/eval_report.md`; latency figures naturally vary
between runs, while task success, retrieval, curation, and maintenance counts are deterministic
for seed `7`.

## Limitations and next steps

- **The LLM is a stub.** `StubChatModel` is a deterministic rule-based model for summarization,
  conflict adjudication, and intent classification. It cannot reason about novel phrasing.
  Swapping in a real chat model would broaden intent coverage and summary quality, at the cost
  of network access, cost, and non-determinism; the evaluation would need a pinned model and
  seeded/near-deterministic decoding to stay reproducible.
- **Embeddings are lexical.** `HashingEmbedder` uses signed feature hashing over tokens and
  bigrams with lithography-aware synonym canonicalization. It is deterministic and offline, but
  it does not capture true semantic paraphrase. A real embedding model would improve recall for
  paraphrased queries and likely shift every retrieval and task-success figure above.
- **The corpus is synthetic.** The 60 source documents and 60 labeled candidates are generated
  from a fixed seed. They exercise numeric contradictions, negation flips, exact duplicates,
  and near-duplicate paraphrases, but they are not a substitute for a real fab knowledge base.
- **Metrics are small-sample.** The retrieval set has four grouped queries and the benchmark has
  40 tasks; treat the perfect curation scores as a regression signal on this corpus, not a claim
  about production accuracy.
- **Latency is machine-specific.** The measured p50/p95 timings include local Postgres, Redis,
  embedding, and graph overhead. They are useful for ON-vs-OFF comparison on this machine, not
  as a service-level guarantee.
- **Single-tenant by design.** There is no authentication or tenant isolation yet; the next
  steps are per-agent namespaces, background maintenance scheduling, a real embedding/LLM
  backend, and a larger human-labeled evaluation set.

## Development

```bash
make up      # docker compose up -d --wait
make test    # pytest with coverage gate
make lint    # ruff check .
make eval    # bash scripts/run_eval.sh
make down    # docker compose down
```
