"""Evaluation harness: memory ablation, retrieval, curation, maintenance, and latency."""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import redis as redis_lib
from sqlalchemy import text
from sqlalchemy.engine import Engine

from agentrecall.agent.session import AgentSession
from agentrecall.config import Settings, get_settings
from agentrecall.curation.pipeline import CurationPipeline
from agentrecall.data.corpus import (
    CorpusCandidate,
    generate_corpus,
    insert_source_docs,
    mutate_source_docs,
)
from agentrecall.db import get_engine, reset_schema
from agentrecall.eval.retrieval import (
    build_curated_retrieval_set,
    build_raw_retrieval_set,
    evaluate_retrieval,
)
from agentrecall.eval.tasks import EvalTask, build_tasks
from agentrecall.maintenance.jobs import (
    audit,
    compact,
    detect_stale,
    reembed_changed,
)
from agentrecall.memory.semantic import SemanticMemory
from agentrecall.memory.store import MemoryStore
from agentrecall.models import utcnow


def percentile(values: list[float], quantile: float) -> float:
    """Linear-interpolated percentile (quantile in [0, 1])."""
    if not values:
        return 0.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def grade_answer(answer: str, expected: str) -> bool:
    """Exact normalized-fragment grading: the expected key must appear as a whole token."""
    normalized_answer = " ".join(answer.lower().split())
    normalized_expected = " ".join(expected.lower().split())
    if not normalized_expected:
        return False
    pattern = rf"(?<!\w){re.escape(normalized_expected)}(?!\w)"
    return re.search(pattern, normalized_answer) is not None


def _truncate_runtime(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE curation_log, resolution_traces, episodic_summaries, "
                "memory_entries CASCADE"
            )
        )


def _reset_corpus(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE curation_log, resolution_traces, episodic_summaries, "
                "memory_entries, source_docs CASCADE"
            )
        )


def _ask_with_latency(session: AgentSession, request: str) -> tuple[str, int]:
    answer = session.ask(request)
    trace = session.get_trace() or {}
    return answer, int(trace.get("latency_ms", 0))


def run_task_arm(
    task: EvalTask,
    memory_enabled: bool,
    engine: Engine,
    settings: Settings,
    redis_client: redis_lib.Redis,
) -> dict[str, Any]:
    """Run one 3-session task with memory on or off, returning answer and latencies."""
    _truncate_runtime(engine)
    redis_client.flushdb()
    arm = "on" if memory_enabled else "off"
    latencies: list[int] = []

    session1 = AgentSession(f"{task.id}-{arm}-s1", memory_enabled, settings=settings)
    _, latency1 = _ask_with_latency(session1, task.session1_fact)
    latencies.append(latency1)
    session1.close()

    session2 = AgentSession(f"{task.id}-{arm}-s2", memory_enabled, settings=settings)
    _, latency2 = _ask_with_latency(session2, task.distractor)
    latencies.append(latency2)
    session2.close()

    session3 = AgentSession(f"{task.id}-{arm}-s3", memory_enabled, settings=settings)
    answer, latency3 = _ask_with_latency(session3, task.question)
    latencies.append(latency3)
    session3.close()

    return {
        "task_id": task.id,
        "intent": task.intent,
        "group": task.group,
        "memory_enabled": memory_enabled,
        "answer": answer,
        "expected": task.expected_answer,
        "success": grade_answer(answer, task.expected_answer),
        "latencies_ms": latencies,
    }


def run_multi_session_tasks(
    engine: Engine,
    settings: Settings,
    redis_client: redis_lib.Redis,
    seed: int,
) -> dict[str, Any]:
    """Run every task in both memory arms and aggregate success by intent."""
    tasks = build_tasks(seed)
    records: list[dict[str, Any]] = []
    all_latencies: dict[str, list[int]] = {"on": [], "off": []}
    for memory_enabled in (True, False):
        for task in tasks:
            record = run_task_arm(task, memory_enabled, engine, settings, redis_client)
            record["memory_enabled"] = memory_enabled
            records.append(record)
            all_latencies["on" if memory_enabled else "off"].extend(record["latencies_ms"])

    on_records = [r for r in records if r["memory_enabled"]]
    off_records = [r for r in records if not r["memory_enabled"]]
    on_success = sum(r["success"] for r in on_records) / len(on_records)
    off_success = sum(r["success"] for r in off_records) / len(off_records)

    by_intent: dict[str, dict[str, Any]] = {}
    for intent in sorted({r["intent"] for r in records}):
        on_subset = [r for r in on_records if r["intent"] == intent]
        off_subset = [r for r in off_records if r["intent"] == intent]
        on_rate = sum(r["success"] for r in on_subset) / len(on_subset)
        off_rate = sum(r["success"] for r in off_subset) / len(off_subset)
        by_intent[intent] = {
            "tasks": len(on_subset),
            "on_rate": on_rate,
            "off_rate": off_rate,
            "delta": on_rate - off_rate,
        }

    return {
        "tasks": len(tasks),
        "sessions_per_task": 3,
        "on_success_rate": on_success,
        "off_success_rate": off_success,
        "delta": on_success - off_success,
        "by_intent": by_intent,
        "latencies_ms": all_latencies,
        "details": [
            {
                "task_id": r["task_id"],
                "intent": r["intent"],
                "group": r["group"],
                "memory_enabled": r["memory_enabled"],
                "success": r["success"],
                "answer": r["answer"],
                "expected": r["expected"],
            }
            for r in records
        ],
    }


def _active_by_subject(engine: Engine) -> dict[str, str]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT subject, id FROM memory_entries WHERE status='active' ORDER BY id")
        ).all()
    result: dict[str, str] = {}
    for subject, entry_id in rows:
        result.setdefault(subject, str(entry_id))
    return result


def run_retrieval_and_curation(
    engine: Engine,
    settings: Settings,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], SemanticMemory, list[CorpusCandidate]]:
    """Measure retrieval before/after curation and curation quality on the labeled stream."""
    _reset_corpus(engine)
    docs, candidates = generate_corpus(seed)
    insert_source_docs(engine, docs)

    raw_store = SemanticMemory(engine=engine, settings=settings)
    for item in candidates:
        raw_store.write(item.candidate)
    raw_entries = raw_store.count()
    raw_queries = build_raw_retrieval_set(candidates)
    raw_metrics = evaluate_retrieval(raw_store, raw_queries)

    _truncate_runtime(engine)
    pipeline = CurationPipeline(engine=engine, settings=settings)
    decisions = [pipeline.ingest(item.candidate) for item in candidates]
    curated_store = SemanticMemory(engine=engine, settings=settings)
    curated_entries = curated_store.count()
    active = _active_by_subject(engine)
    curated_queries = build_curated_retrieval_set(candidates, active)
    curated_metrics = evaluate_retrieval(curated_store, curated_queries)

    predicted_dedup = sum(d.action in {"skip_duplicate", "merge"} for d in decisions)
    dedup_tp = sum(
        d.action in {"skip_duplicate", "merge"}
        and item.label in {"exact_duplicate", "near_duplicate"}
        for d, item in zip(decisions, candidates, strict=True)
    )
    actual_dedup = sum(
        item.label in {"exact_duplicate", "near_duplicate"} for item in candidates
    )
    predicted_conflict = sum(
        d.action in {"supersede", "reject_conflict"} for d in decisions
    )
    conflict_tp = sum(
        d.action in {"supersede", "reject_conflict"}
        and item.label.endswith("contradiction")
        for d, item in zip(decisions, candidates, strict=True)
    )
    actual_conflict = sum(item.label.endswith("contradiction") for item in candidates)

    dedup_precision = dedup_tp / predicted_dedup if predicted_dedup else 0.0
    dedup_recall = dedup_tp / actual_dedup if actual_dedup else 0.0
    conflict_precision = conflict_tp / predicted_conflict if predicted_conflict else 0.0
    conflict_recall = conflict_tp / actual_conflict if actual_conflict else 0.0
    conflict_f1 = (
        2 * conflict_precision * conflict_recall / (conflict_precision + conflict_recall)
        if (conflict_precision + conflict_recall)
        else 0.0
    )
    reduction = (
        (raw_entries - curated_entries) / raw_entries * 100.0 if raw_entries else 0.0
    )

    retrieval = {
        "k": 5,
        "raw": raw_metrics.model_dump(),
        "curated": curated_metrics.model_dump(),
        "queries": [q.model_dump() for q in curated_queries],
    }
    curation = {
        "decisions": dict(sorted(Counter(d.action for d in decisions).items())),
        "dedup": {
            "predicted": predicted_dedup,
            "actual": actual_dedup,
            "true_positives": dedup_tp,
            "precision": dedup_precision,
            "recall": dedup_recall,
        },
        "contradiction": {
            "predicted": predicted_conflict,
            "actual": actual_conflict,
            "true_positives": conflict_tp,
            "precision": conflict_precision,
            "recall": conflict_recall,
            "f1": conflict_f1,
        },
        "entries_before": raw_entries,
        "entries_after": curated_entries,
        "reduction_pct": reduction,
    }
    return retrieval, curation, curated_store, candidates


def run_maintenance(
    engine: Engine, settings: Settings
) -> dict[str, Any]:
    """Mutate sources, run staleness/re-embedding twice, compact, and audit."""
    mutated = mutate_source_docs(engine, limit=3)
    stale = detect_stale(engine=engine, settings=settings)
    reembed_one = reembed_changed(engine=engine, settings=settings)
    reembed_two = reembed_changed(engine=engine, settings=settings)
    compaction = compact(engine=engine, settings=settings)
    audit_result = audit(engine=engine, settings=settings)
    return {
        "mutated_documents": len(mutated),
        "stale": stale.model_dump(),
        "reembed_run_1": reembed_one.model_dump(),
        "reembed_run_2": reembed_two.model_dump(),
        "compact": compaction.model_dump(),
        "audit": {
            "scanned": audit_result.scanned,
            "total_findings": len(audit_result.findings),
            "counts": audit_result.counts,
        },
    }


def run_latency(
    task_section: dict[str, Any],
    store: SemanticMemory,
    candidates: list[CorpusCandidate],
    settings: Settings,
) -> dict[str, Any]:
    """Summarize ask latency by arm and measure mean direct memory-recall latency."""
    on_latencies = task_section["latencies_ms"]["on"]
    off_latencies = task_section["latencies_ms"]["off"]
    active_queries = build_curated_retrieval_set(candidates, _active_by_subject(store.engine))
    memory_store = MemoryStore(engine=store.engine, settings=settings)
    recall_samples: list[float] = []
    for item in active_queries:
        for _ in range(3):
            started = time.perf_counter()
            memory_store.recall(item.query, "eval-latency", k=5)
            recall_samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "asks_on": len(on_latencies),
        "asks_off": len(off_latencies),
        "p50_on_ms": percentile([float(v) for v in on_latencies], 0.50),
        "p95_on_ms": percentile([float(v) for v in on_latencies], 0.95),
        "p50_off_ms": percentile([float(v) for v in off_latencies], 0.50),
        "p95_off_ms": percentile([float(v) for v in off_latencies], 0.95),
        "mean_recall_ms": sum(recall_samples) / len(recall_samples) if recall_samples else 0.0,
        "recall_samples": len(recall_samples),
    }


def run_all(
    seed: int = 7,
    write: bool = True,
    results_dir: str | Path = "results",
    engine: Engine | None = None,
    settings: Settings | None = None,
    redis_client: redis_lib.Redis | None = None,
) -> dict[str, Any]:
    """Execute the full A-E evaluation and optionally write JSON + Markdown reports."""
    settings = settings or get_settings()
    engine = engine or get_engine(settings)
    reset_schema(engine)
    client = redis_client or redis_lib.Redis.from_url(settings.redis_url, decode_responses=True)

    task_section = run_multi_session_tasks(engine, settings, client, seed)
    retrieval, curation, curated_store, candidates = run_retrieval_and_curation(
        engine, settings, seed
    )
    maintenance = run_maintenance(engine, settings)
    latency = run_latency(task_section, curated_store, candidates, settings)

    results = {
        "seed": seed,
        "timestamp": utcnow().isoformat(),
        "config": {
            "embed_dim": settings.embed_dim,
            "w_similarity": settings.w_similarity,
            "w_recency": settings.w_recency,
            "w_usage": settings.w_usage,
            "recency_half_life_days": settings.recency_half_life_days,
            "decay_floor": settings.decay_floor,
            "dedup_threshold": settings.dedup_threshold,
            "conflict_threshold": settings.conflict_threshold,
            "stale_after_days": settings.stale_after_days,
        },
        "task_success": {
            key: value
            for key, value in task_section.items()
            if key not in {"latencies_ms", "details"}
        },
        "task_details": task_section["details"],
        "retrieval": retrieval,
        "curation": curation,
        "maintenance": maintenance,
        "latency": latency,
    }

    if write:
        directory = Path(results_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "eval_results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (directory / "eval_report.md").write_text(render_report(results), encoding="utf-8")
        results["_paths"] = {
            "json": str(directory / "eval_results.json"),
            "markdown": str(directory / "eval_report.md"),
        }
    return results


def render_report(results: dict[str, Any]) -> str:
    """Render the evaluation results as Markdown tables."""
    tasks = results["task_success"]
    retrieval = results["retrieval"]
    curation = results["curation"]
    maintenance = results["maintenance"]
    latency = results["latency"]
    lines = [
        "# AgentRecall Evaluation Report",
        "",
        f"Seed: `{results['seed']}`",
        f"Generated: {results['timestamp']}",
        "",
        "## A. Multi-session task success (memory ON vs OFF)",
        "",
        "| Arm | Tasks | Successes | Success rate |",
        "| --- | ---: | ---: | ---: |",
        f"| Memory ON | {tasks['tasks']} | {round(tasks['on_success_rate'] * tasks['tasks'])} | "
        f"{tasks['on_success_rate']:.3f} |",
        f"| Memory OFF | {tasks['tasks']} | {round(tasks['off_success_rate'] * tasks['tasks'])} | "
        f"{tasks['off_success_rate']:.3f} |",
        f"| Delta | {tasks['tasks']} | - | {tasks['delta']:+.3f} |",
        "",
        "| Intent | Tasks | ON rate | OFF rate | Delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for intent, data in tasks["by_intent"].items():
        lines.append(
            f"| {intent} | {data['tasks']} | {data['on_rate']:.3f} | "
            f"{data['off_rate']:.3f} | {data['delta']:+.3f} |"
        )

    lines += [
        "",
        "## B. Retrieval metrics before vs after curation",
        "",
        "| Store | Queries | Precision@5 | Recall@5 | MRR | nDCG@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (("raw", "Raw ingest"), ("curated", "After curation")):
        metrics = retrieval[key]
        lines.append(
            f"| {label} | {metrics['queries']} | {metrics['precision_at_k']:.3f} | "
            f"{metrics['recall_at_k']:.3f} | {metrics['mrr']:.3f} | "
            f"{metrics['ndcg_at_k']:.3f} |"
        )

    lines += [
        "",
        "## C. Curation quality",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Dedup precision | {curation['dedup']['precision']:.3f} |",
        f"| Dedup recall | {curation['dedup']['recall']:.3f} |",
        f"| Contradiction precision | {curation['contradiction']['precision']:.3f} |",
        f"| Contradiction recall | {curation['contradiction']['recall']:.3f} |",
        f"| Contradiction F1 | {curation['contradiction']['f1']:.3f} |",
        f"| Entries before curation | {curation['entries_before']} |",
        f"| Entries after curation | {curation['entries_after']} |",
        f"| Store reduction | {curation['reduction_pct']:.1f}% |",
        "",
        "Decision counts: "
        + ", ".join(f"`{k}`={v}" for k, v in curation["decisions"].items()),
        "",
        "## D. Maintenance effects",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Source documents mutated | {maintenance['mutated_documents']} |",
        f"| Stale entries detected | {maintenance['stale']['marked_stale']} |",
        f"| Stale reasons | {maintenance['stale']['reasons']} |",
        f"| Re-embeds run 1 | {maintenance['reembed_run_1']['reembedded_entries']} |",
        f"| Re-embeds run 2 | {maintenance['reembed_run_2']['reembedded_entries']} |",
        f"| Documents changed run 1 | {maintenance['reembed_run_1']['changed_documents']} |",
        f"| Documents changed run 2 | {maintenance['reembed_run_2']['changed_documents']} |",
        f"| Compacted (retired) | {maintenance['compact']['retired']} |",
        f"| Audit findings | {maintenance['audit']['total_findings']} |",
        "",
        "| Audit finding type | Count |",
        "| --- | ---: |",
    ]
    for kind, count in maintenance["audit"]["counts"].items():
        lines.append(f"| {kind} | {count} |")

    lines += [
        "",
        "## E. Latency",
        "",
        "| Metric | Memory ON | Memory OFF |",
        "| --- | ---: | ---: |",
        f"| ask p50 (ms) | {latency['p50_on_ms']:.2f} | {latency['p50_off_ms']:.2f} |",
        f"| ask p95 (ms) | {latency['p95_on_ms']:.2f} | {latency['p95_off_ms']:.2f} |",
        f"| asks measured | {latency['asks_on']} | {latency['asks_off']} |",
        "",
        f"Mean direct memory recall latency: **{latency['mean_recall_ms']:.2f} ms** "
        f"({latency['recall_samples']} samples).",
        "",
        "Reproduce with:",
        "",
        "```bash",
        f"agentrecall eval run --seed {results['seed']}",
        "```",
        "",
    ]
    return "\n".join(lines)
