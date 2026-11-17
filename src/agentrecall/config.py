"""Application settings loaded from ``AGENTRECALL_*`` environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for every AgentRecall subsystem."""

    model_config = SettingsConfigDict(
        env_prefix="AGENTRECALL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    pg_dsn: str = "postgresql+psycopg://agentrecall:agentrecall@localhost:5433/agentrecall"
    redis_url: str = "redis://localhost:6380/0"

    # Embedding space. The local HashingEmbedder always produces this dimension.
    embed_dim: int = 384

    # Retrieval score weights.
    w_similarity: float = 0.6
    w_recency: float = 0.25
    w_usage: float = 0.15
    recency_half_life_days: float = 14.0
    decay_floor: float = 0.05
    stale_after_days: float = 90.0

    # Write-path thresholds.
    dedup_threshold: float = 0.94
    conflict_threshold: float = 0.86

    # Scratchpad behavior.
    scratch_ttl_seconds: int = 24 * 60 * 60
    scratch_max_turns: int = 50

    # Agent behavior.
    summarize_every_turns: int = 4
    max_recall_loops: int = 1

    # Optional real LLM backend; the stub is used when this is unset.
    llm_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = "not-needed"

    @property
    def weights_sum(self) -> float:
        """Return the sum of the three retrieval weights."""
        return self.w_similarity + self.w_recency + self.w_usage

    @property
    def use_stub_llm(self) -> bool:
        """True when no OpenAI-compatible endpoint has been configured."""
        return not bool(self.llm_base_url)


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Return a process-wide settings singleton (or reload it)."""
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings


def configure(settings: Settings) -> None:
    """Override the process-wide settings (used by tests and tools)."""
    global _settings
    _settings = settings
