"""Agent graph tests: intent routing, bounded recall, traces, write-back, ablation."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from agentrecall.agent import AgentSession, extract_durable_fact, extract_slots, session_traces
from agentrecall.agent.tools import TOOLS
from agentrecall.llm import classify_intent
from agentrecall.models import MemoryTier


@pytest.fixture(autouse=True)
def _flush_redis(redis_client):
    yield


def _session(session_id: str, memory: bool = True, settings=None) -> AgentSession:
    return AgentSession(session_id, memory_enabled=memory, settings=settings)


def _semantic_count(session: AgentSession) -> int:
    return session.store.semantic.count()


def test_classify_all_five_intents_and_unknown():
    assert classify_intent("What is the numerical aperture for stack A?") == (
        "lookup_recipe_parameter"
    )
    assert classify_intent("OPC convergence is failing after 6 iterations") == (
        "diagnose_opc_convergence"
    )
    assert classify_intent(
        "Compute the process window for dose latitude 8 percent and depth of focus 90 nm"
    ) == "compute_process_window"
    assert classify_intent("Summarize the prior session for stack A") == (
        "summarize_prior_session"
    )
    assert classify_intent("Draft a job deck for layer M7") == "draft_job_deck"
    assert classify_intent("Tell me a joke about masks") == "unknown"


def test_slot_extraction():
    slots = extract_slots(
        "For resist stack A at dose 32 mJ/cm2 with NA 1.35 and depth of focus 80 nm, "
        "after 6 iterations with edge placement error 1.2 nm, layer M7 uses 2 exposure passes."
    )
    assert slots["dose"] == 32.0
    assert slots["na"] == 1.35
    assert slots["dof"] == 80.0
    assert slots["iterations"] == 6
    assert slots["epe"] == 1.2
    assert slots["passes"] == 2
    assert slots["layer"] == "M7"
    assert slots["stack"] == "A"


def test_tool_routing_for_each_intent(settings, engine, redis_client):
    cases = [
        ("What is the numerical aperture for stack A?", "lookup_recipe_parameter"),
        ("OPC convergence is failing after 6 iterations with EPE 2.4 nm",
         "diagnose_opc_convergence"),
        ("Compute the process window for dose latitude 8 percent and depth of focus 90 nm",
         "compute_process_window"),
        ("Summarize the prior session for stack A", "summarize_prior_session"),
        ("Draft a job deck for layer M7 with 2 exposure passes", "draft_job_deck"),
    ]
    for index, (request, expected_tool) in enumerate(cases):
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE memory_entries, episodic_summaries CASCADE"))
        redis_client.flushdb()
        session = _session(f"route-{index}", settings=settings)
        session.ask(request)
        trace = session.get_trace()
        assert trace["intent"] == expected_tool
        assert trace["chosen_tool"] == expected_tool
        assert trace["steps"]
        session.close()


def test_unknown_intent_is_clarification(settings):
    session = _session("unknown-session", settings=settings)
    answer = session.ask("Tell me a joke about masks")
    trace = session.get_trace()
    assert trace["intent"] == "unknown"
    assert trace["chosen_tool"] is None
    assert trace["outcome"] == "clarification"
    assert "could not map" in answer


def test_bounded_recall_loop_terminates(settings):
    session = _session("loop-session", settings=settings)
    session.ask("OPC calibration is not converging and the job deck fails.")
    trace = session.get_trace()
    recall_steps = [step for step in trace["steps"] if step["node"] == "recall_memory"]
    assert 1 <= len(recall_steps) <= 2
    assert session.last_state["recall_loops"] <= settings.max_recall_loops + 1
    assert trace["chosen_tool"] == "diagnose_opc_convergence"


def test_trace_persisted_with_steps_and_hits(settings):
    s1 = _session("trace-s1", settings=settings)
    s1.ask("Remember: for resist stack A, the qualified dose is 32 mJ/cm2 at 193 nm.")
    s1.close()

    s2 = _session("trace-s2", settings=settings)
    answer = s2.ask("What is the qualified dose for resist stack A?")
    assert "32" in answer
    trace = s2.get_trace()
    assert trace["outcome"] == "answered_from_memory"
    assert trace["memory_hits"]
    assert trace["memory_hits"][0]["tier"] == MemoryTier.SEMANTIC.value
    assert trace["memory_hits"][0]["score"] > 0
    assert len(trace["steps"]) >= 6
    assert {step["node"] for step in trace["steps"]} >= {
        "classify_intent",
        "recall_memory",
        "plan",
        "route_tool",
        "execute",
        "reflect_and_write",
        "summarize_turn",
    }
    rows = session_traces(s2.store.engine, "trace-s2")
    assert len(rows) == 1 and rows[0]["latency_ms"] >= 0


def test_reflect_and_write_creates_retrievable_memory(settings):
    s1 = _session("write-s1", settings=settings)
    assert _semantic_count(s1) == 0
    s1.ask("Remember: for resist stack B, the qualified dose is 29 mJ/cm2 at 193 nm.")
    s1.close()
    assert _semantic_count(s1) == 1
    hits = s1.store.semantic.search("dose for resist stack B", k=5)
    assert hits and "29" in hits[0].record.claim

    s2 = _session("write-s2", settings=settings)
    answer = s2.ask("What dose did we qualify for resist stack B?")
    assert "29" in answer
    assert s2.last_state["outcome"] == "answered_from_memory"

    with s1.store.engine.connect() as conn:
        actions = [
            row[0]
            for row in conn.execute(text("SELECT action FROM curation_log ORDER BY created_at"))
        ]
    assert "accept" in actions


def test_tool_result_write_back(settings):
    session = _session("tool-write", settings=settings)
    session.ask(
        "Compute the process window for dose 32 mJ/cm2 with dose latitude 8 percent "
        "and depth of focus 90 nm."
    )
    state = session.last_state
    assert state["chosen_tool"] == "compute_process_window"
    assert state["tool_result"]["verdict"] == "robust"
    assert _semantic_count(session) == 1
    assert any(write["subject"] == "process window" for write in state["memory_writes"])


def test_memory_disabled_performs_no_memory_io(settings):
    baseline = _session("ablation-baseline", settings=settings)
    before = baseline.store.semantic.count()

    session = _session("ablation-off", memory=False, settings=settings)
    session.ask("Remember: for resist stack C the dose is 27 mJ/cm2.")
    session.ask("What is the dose for resist stack C?")
    session.close()

    trace = session.get_trace()
    assert session.memory_enabled is False
    assert session.store.semantic.count() == before
    assert session.store.episodic.count() == 0
    assert session.store.scratchpad.length("ablation-off") == 0
    assert trace["memory_hits"] == []
    assert all(step["node"] != "recall_memory" for step in trace["steps"])
    assert trace["steps"][0]["node"] == "classify_intent"
    assert trace["outcome"] == "tool_error"  # lookup tool had no recalled value


def test_tools_declare_schemas_and_required_facts():
    assert set(TOOLS) == {
        "lookup_recipe_parameter",
        "diagnose_opc_convergence",
        "compute_process_window",
        "summarize_prior_session",
        "draft_job_deck",
    }
    for spec in TOOLS.values():
        assert spec.required_facts
        assert spec.description
        assert spec.args_model.__doc__
        assert spec.func.__doc__
    result = TOOLS["diagnose_opc_convergence"].run(
        {"iterations": 4, "epe_nm": 3.5, "epe_threshold_nm": 1.5}
    )
    assert result["diagnosis"] == "unstable"
    result = TOOLS["compute_process_window"].run(
        {"dose_latitude_pct": 8, "depth_of_focus_nm": 90}
    )
    assert result["verdict"] == "robust"


def test_extract_durable_fact_ignores_questions():
    assert extract_durable_fact("What is the dose for stack A?") is None
    fact = extract_durable_fact(
        "Remember: for resist stack D the dose is 31 mJ/cm2 at 193 nm."
    )
    assert fact is not None
    assert fact[0] == "resist stack d dose"
    assert "31" in fact[1]


def test_two_session_demo(settings):
    """Session 2 answers using a fact only session 1 supplied."""
    first = _session("demo-1", settings=settings)
    first.ask("Remember: the qualified dose for resist stack E is 28 mJ/cm2 at 193 nm.")
    first.close()

    second = _session("demo-2", settings=settings)
    answer = second.ask("What is the qualified dose for resist stack E?")
    assert "28" in answer
    trace = second.get_trace()
    assert trace["outcome"] == "answered_from_memory"
