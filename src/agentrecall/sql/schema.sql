-- AgentRecall schema: PostgreSQL 16 + pgvector.
-- All DDL is idempotent so `init_schema()` can run on every boot.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS source_docs (
    id            text PRIMARY KEY,
    uri           text,
    title         text,
    body          text,
    content_hash  text,
    updated_at    timestamptz,
    tags          text[]
);

CREATE TABLE IF NOT EXISTS memory_entries (
    id             uuid PRIMARY KEY,
    tier           text NOT NULL DEFAULT 'semantic',
    subject        text NOT NULL,
    claim          text NOT NULL,
    body           text,
    embedding      vector(384),
    source_doc_id  text NULL REFERENCES source_docs(id),
    session_id     text NULL,
    version        int NOT NULL DEFAULT 1,
    status         text NOT NULL DEFAULT 'active',
    supersedes     uuid NULL REFERENCES memory_entries(id),
    provenance     jsonb NOT NULL DEFAULT '{}'::jsonb,
    confidence     real NOT NULL DEFAULT 0.5,
    access_count   int NOT NULL DEFAULT 0,
    created_at     timestamptz NOT NULL DEFAULT now(),
    last_used_at   timestamptz NOT NULL DEFAULT now(),
    content_hash   text NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS episodic_summaries (
    id          uuid PRIMARY KEY,
    session_id  text NOT NULL,
    start_turn  int NOT NULL,
    end_turn    int NOT NULL,
    summary     text NOT NULL,
    embedding   vector(384),
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS resolution_traces (
    id           uuid PRIMARY KEY,
    session_id   text NOT NULL,
    request      text NOT NULL,
    intent       text NOT NULL,
    chosen_tool  text,
    memory_hits  jsonb NOT NULL DEFAULT '[]'::jsonb,
    steps        jsonb NOT NULL DEFAULT '[]'::jsonb,
    outcome      text NOT NULL DEFAULT 'unknown',
    latency_ms   int NOT NULL DEFAULT 0,
    created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS curation_log (
    id          uuid PRIMARY KEY,
    action      text NOT NULL,
    target_id   uuid NULL,
    rival_id    uuid NULL,
    reason      text,
    similarity  real,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- Vector index for cosine nearest-neighbour search.
CREATE INDEX IF NOT EXISTS idx_memory_embedding
    ON memory_entries USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_memory_status ON memory_entries (status);
CREATE INDEX IF NOT EXISTS idx_memory_subject ON memory_entries (subject);
CREATE INDEX IF NOT EXISTS idx_memory_session ON memory_entries (session_id);
CREATE INDEX IF NOT EXISTS idx_memory_source ON memory_entries (source_doc_id);
CREATE INDEX IF NOT EXISTS idx_episodic_session ON episodic_summaries (session_id);
CREATE INDEX IF NOT EXISTS idx_traces_session ON resolution_traces (session_id);
CREATE INDEX IF NOT EXISTS idx_curation_target ON curation_log (target_id);
