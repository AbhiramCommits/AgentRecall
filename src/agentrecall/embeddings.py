"""Deterministic local embeddings implementing the LangChain ``Embeddings`` interface.

``HashingEmbedder`` is a dependency-free feature-hashing encoder: tokens are hashed with
BLAKE2b (never Python's randomized ``hash``), accumulated with a signed projection into a
fixed 384-dimensional space, and L2-normalized. Identical text therefore yields identical
vectors in every process, which keeps the whole test suite offline and reproducible.
Swap in a real sentence-transformer by setting a different embedder via ``get_embedder``.
"""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

from agentrecall.config import get_settings

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*")
_STOPWORDS = frozenset(
    (
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "with",
        "this",
        "these",
        "those",
        "we",
        "you",
        "i",
    )
)


def tokenize(text: str) -> list[str]:
    """Lowercase tokenize text, dropping stopwords but keeping parameter-like tokens."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class HashingEmbedder(Embeddings):
    """Signed feature hashing into a stable, L2-normalized vector space."""

    def __init__(self, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError("embedding dimension must be positive")
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % self.dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return index, sign

    def embed_text(self, text: str) -> list[float]:
        """Embed one string, including adjacent token bigrams for a little word order."""
        vec = [0.0] * self.dim
        tokens = tokenize(text)
        for token in tokens:
            idx, sign = self._bucket(token)
            vec[idx] += sign
        for left, right in zip(tokens, tokens[1:], strict=False):
            idx, sign = self._bucket(f"{left}~{right}")
            vec[idx] += 0.5 * sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # Never return a zero vector: cosine distance is undefined.
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)


_embedder: Embeddings | None = None


def get_embedder(dim: int | None = None) -> Embeddings:
    """Return the configured embedder (a process-wide hashing embedder by default)."""
    global _embedder
    if dim is not None:
        return HashingEmbedder(dim=dim)
    if _embedder is None:
        _embedder = HashingEmbedder(dim=get_settings().embed_dim)
    return _embedder


def set_embedder(embedder: Embeddings | None) -> None:
    """Install a custom embedder, or reset to the default with ``None``."""
    global _embedder
    _embedder = embedder


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors, clamped to [-1, 1]."""
    if len(a) != len(b):
        raise ValueError("vectors must share a dimension")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def vector_literal(vec: list[float]) -> str:
    """Render a float list as a pgvector text literal."""
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"
