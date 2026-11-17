"""Eval tests: exact grading, hand-computed metrics, benchmark shape, reproducibility."""

from __future__ import annotations

import json
import math

import pytest

from agentrecall.data.corpus import build_candidates
from agentrecall.eval.harness import grade_answer, percentile, run_all
from agentrecall.eval.retrieval import (
    build_curated_retrieval_set,
    build_raw_retrieval_set,
    ranking_metrics,
)
from agentrecall.eval.tasks import build_tasks


def test_grade_answer_is_exact_fragment_match():
    assert grade_answer("The dose is 32 mJ/cm2.", "32 mj/cm2")
    assert grade_answer("verdict: slow_convergence after 3 iterations", "slow_convergence")
    assert not grade_answer("The dose is 132 mJ/cm2.", "32 mj/cm2")
    assert not grade_answer("The dose is 32 mJ/cm3.", "32 mj/cm2")
    assert not grade_answer("no data", "")


def test_percentile_hand_computed():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 0.0) == 10.0
    assert percentile(values, 0.5) == 30.0
    assert percentile(values, 1.0) == 50.0
    assert math.isclose(percentile(values, 0.95), 48.0)
    assert percentile([], 0.5) == 0.0
    with pytest.raises(ValueError):
        percentile(values, 1.5)


def test_ranking_metrics_hand_computed():
    metrics = ranking_metrics(
        rankings=[["a", "b", "c", "d", "e"], ["x", "c", "b"]],
        relevants=[{"a", "c"}, {"a", "c"}],
        k=5,
    )
    dcg_1 = 1.0 + 1.0 / math.log2(4)
    idcg = 1.0 + 1.0 / math.log2(3)
    dcg_2 = 1.0 / math.log2(3)
    expected_precision = (2 / 5 + 1 / 5) / 2
    expected_recall = (1.0 + 0.5) / 2
    expected_mrr = (1.0 + 1 / 2) / 2
    expected_ndcg = (dcg_1 / idcg + dcg_2 / idcg) / 2
    assert math.isclose(metrics.precision_at_k, expected_precision)
    assert math.isclose(metrics.recall_at_k, expected_recall)
    assert math.isclose(metrics.mrr, expected_mrr)
    assert math.isclose(metrics.ndcg_at_k, expected_ndcg)
    assert metrics.queries == 2 and metrics.k == 5


def test_task_benchmark_shape_and_determinism():
    tasks = build_tasks(seed=7)
    assert len(tasks) == 40
    assert {t.intent for t in tasks} == {
        "lookup_recipe_parameter",
        "compute_process_window",
        "draft_job_deck",
        "diagnose_opc_convergence",
        "summarize_prior_session",
    }
    assert sum(t.intent == "lookup_recipe_parameter" for t in tasks) == 16
    for task in tasks:
        assert task.session1_fact and task.distractor and task.question
        assert task.expected_answer
        assert "pilot stack" in task.session1_fact
    again = build_tasks(seed=7)
    assert [t.model_dump() for t in tasks] == [t.model_dump() for t in again]
    assert [t.expected_answer for t in build_tasks(seed=9)] != [
        t.expected_answer for t in tasks
    ]


def test_retrieval_sets_are_labeled_and_nonempty():
    candidates = build_candidates(seed=7)
    raw = build_raw_retrieval_set(candidates)
    assert len(raw) == 4
    assert all(query.relevant_ids for query in raw)
    active = {c.candidate.subject.lower(): c.candidate.id for c in candidates[:20]}
    curated = build_curated_retrieval_set(candidates, active)
    assert len(curated) == 4
    assert all(len(query.relevant_ids) in {4, 5} for query in curated)
    ids = {entry_id for query in curated for entry_id in query.relevant_ids}
    assert len(ids) == 19


def test_run_all_is_seed_reproducible_for_deterministic_metrics(engine, settings, tmp_path):
    first = run_all(seed=7, write=False, engine=engine, settings=settings)
    second = run_all(
        seed=7, write=True, results_dir=tmp_path, engine=engine, settings=settings
    )

    assert first["task_success"] == second["task_success"]
    assert first["retrieval"]["raw"] == second["retrieval"]["raw"]
    assert first["retrieval"]["curated"] == second["retrieval"]["curated"]
    assert first["curation"] == second["curation"]
    assert first["maintenance"]["stale"] == second["maintenance"]["stale"]
    assert first["maintenance"]["reembed_run_1"] == second["maintenance"]["reembed_run_1"]
    assert first["maintenance"]["reembed_run_2"] == second["maintenance"]["reembed_run_2"]
    assert first["maintenance"]["audit"] == second["maintenance"]["audit"]
    assert first["task_success"]["on_success_rate"] == 1.0
    assert first["task_success"]["off_success_rate"] == 0.0
    assert first["retrieval"]["curated"]["precision_at_k"] > first["retrieval"]["raw"][
        "precision_at_k"
    ]
    assert first["curation"]["dedup"]["precision"] >= 0.9
    assert first["curation"]["contradiction"]["f1"] >= 0.9

    payload = json.loads((tmp_path / "eval_results.json").read_text())
    assert payload["seed"] == 7
    assert payload["task_success"]["tasks"] == 40
    report = (tmp_path / "eval_report.md").read_text()
    assert "Multi-session task success" in report
    assert "Retrieval metrics" in report
    assert "Curation quality" in report
    assert "Maintenance effects" in report
    assert "Latency" in report
