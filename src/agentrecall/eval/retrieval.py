"""Labeled retrieval set and precision/recall/MRR/nDCG@5 metric implementations.

Queries group five distinct stacks so a query has several genuinely relevant entries.
Before curation the raw stream contains duplicate and contradictory rows that crowd the
top-5; after curation each subject has one ACTIVE head, which is what the metrics measure.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from pydantic import BaseModel

from agentrecall.curation.synonyms import normalize_text
from agentrecall.data.corpus import CorpusCandidate
from agentrecall.memory.semantic import SemanticMemory

_GROUPS: tuple[tuple[int, int], ...] = ((1, 6), (6, 11), (11, 16), (16, 20))


class RetrievalQuery(BaseModel):
    """One labeled query: expected relevant entry ids for the retrieval store."""

    query: str
    relevant_ids: list[str]
    subject: str


class RetrievalMetrics(BaseModel):
    """Aggregate retrieval metrics at cutoff k."""

    k: int = 5
    queries: int = 0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0


def _group_query(candidates: list[CorpusCandidate]) -> list[tuple[str, list[str], str]]:
    queries: list[tuple[str, list[str], str]] = []
    for start, end in _GROUPS:
        subjects = [candidates[i].candidate.subject for i in range(start, end)]
        stacks = [subject.split()[-2].upper() for subject in subjects]
        queries.append(
            (
                "qualified dose recipes for resist stacks " + " ".join(stacks) + " at 193 nm",
                subjects,
                "stacks-" + "-".join(stacks),
            )
        )
    return queries


def build_raw_retrieval_set(candidates: list[CorpusCandidate]) -> list[RetrievalQuery]:
    """Queries whose relevant ids are the distinct base facts ingested without curation."""
    queries: list[RetrievalQuery] = []
    for query, subjects, label in _group_query(candidates):
        relevant = [
            item.candidate.id
            for item in candidates[:20]
            if item.label == "unique_fact" and item.candidate.subject in subjects
        ]
        queries.append(
            RetrievalQuery(query=query, relevant_ids=sorted(relevant), subject=label)
        )
    return queries


def build_curated_retrieval_set(
    candidates: list[CorpusCandidate], active_by_subject: dict[str, str]
) -> list[RetrievalQuery]:
    """Queries whose relevant ids are the surviving ACTIVE heads for each grouped stack."""
    queries: list[RetrievalQuery] = []
    for query, subjects, label in _group_query(candidates):
        relevant = [
            active_by_subject[normalize_text(subject)]
            for subject in subjects
            if normalize_text(subject) in active_by_subject
        ]
        queries.append(
            RetrievalQuery(query=query, relevant_ids=sorted(set(relevant)), subject=label)
        )
    return queries


def ranking_metrics(
    rankings: Iterable[list[str]], relevants: Iterable[set[str]], k: int = 5
) -> RetrievalMetrics:
    """Compute precision@k, recall@k, MRR, and binary-gain nDCG@k."""
    rankings = list(rankings)
    relevants = list(relevants)
    if len(rankings) != len(relevants):
        raise ValueError("rankings and relevants must have equal length")
    if not rankings:
        return RetrievalMetrics(k=k)

    precision_sum = 0.0
    recall_sum = 0.0
    reciprocal_rank_sum = 0.0
    ndcg_sum = 0.0
    for ranked, relevant in zip(rankings, relevants, strict=True):
        top = ranked[:k]
        hits = [1 if entry_id in relevant else 0 for entry_id in top]
        precision_sum += sum(hits) / k
        recall_sum += (sum(hits) / len(relevant)) if relevant else 0.0
        reciprocal_rank_sum += next(
            (1.0 / (index + 1) for index, hit in enumerate(hits) if hit), 0.0
        )
        dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
        ndcg_sum += dcg / idcg if idcg else 0.0

    count = len(rankings)
    return RetrievalMetrics(
        k=k,
        queries=count,
        precision_at_k=precision_sum / count,
        recall_at_k=recall_sum / count,
        mrr=reciprocal_rank_sum / count,
        ndcg_at_k=ndcg_sum / count,
    )


def evaluate_retrieval(
    store: SemanticMemory, queries: list[RetrievalQuery], k: int = 5
) -> RetrievalMetrics:
    """Run each query against the semantic store and aggregate the metrics."""
    rankings: list[list[str]] = []
    relevants: list[set[str]] = []
    for item in queries:
        hits = store.search(item.query, k=k)
        rankings.append([hit.record.id for hit in hits])
        relevants.append(set(item.relevant_ids))
    return ranking_metrics(rankings, relevants, k=k)
