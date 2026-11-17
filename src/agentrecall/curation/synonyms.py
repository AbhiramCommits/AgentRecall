"""Small deterministic synonym/canonicalization map for lithography memory text.

Canonicalization happens before hashing and embedding so that "NA 1.35", "numerical
aperture 1.35", and "exposure dose 32 mJ/cm^2" collapse onto the same representation as
their canonical counterparts. This makes exact dedup robust to surface form and keeps
near-duplicate cosine similarity high without pretending to do semantic NLP.
"""

from __future__ import annotations

import hashlib
import re

# Ordered longest-first so multi-word phrases win before their component words.
SYNONYMS: tuple[tuple[str, str], ...] = (
    ("numerical aperture", "numerical aperture"),
    ("millijoules per square centimeter", "mj/cm2"),
    ("millijoule per square centimeter", "mj/cm2"),
    ("mj/cm^2", "mj/cm2"),
    ("mj/cm²", "mj/cm2"),
    ("mj cm-2", "mj/cm2"),
    ("exposure dose", "dose"),
    ("exposure latitude", "dose latitude"),
    ("depth of focus", "depth of focus"),
    ("critical dimension", "critical dimension"),
    ("optical proximity correction", "opc"),
    ("process window", "process window"),
    ("job deck", "job deck"),
    ("numerical-aperture", "numerical aperture"),
    ("dof", "depth of focus"),
    ("cd", "critical dimension"),
    ("na", "numerical aperture"),
)

_PARAMETER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(p) for p, _ in sorted(SYNONYMS, key=lambda x: -len(x[0]))) + r")\b",
    re.IGNORECASE,
)
_SYNONYM_MAP = {source.lower(): canonical for source, canonical in SYNONYMS}


def canonicalize(text: str) -> str:
    """Lowercase and replace known lithography synonyms with their canonical form."""
    lowered = text.lower().replace("²", "2")

    def replace(match: re.Match[str]) -> str:
        return _SYNONYM_MAP.get(match.group(1).lower(), match.group(1).lower())

    return _PARAMETER_PATTERN.sub(replace, lowered)


def normalize_text(text: str) -> str:
    """Trim, collapse whitespace, canonicalize units/parameter names, strip punctuation."""
    canonical = canonicalize(text.replace("\u00a0", " "))
    collapsed = " ".join(canonical.split())
    return collapsed.strip(" .,;:")


def content_hash(subject: str, claim: str) -> str:
    """Stable SHA-256 over the normalized subject and claim."""
    payload = f"{normalize_text(subject)}|{normalize_text(claim)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
