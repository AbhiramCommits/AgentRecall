"""LangGraph intent-to-action workflow with memory recall, write-back, and traces."""

from __future__ import annotations

import json
import re
import time
from typing import Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from sqlalchemy import text

from agentrecall.agent.tools import INTENT_TO_TOOL, TOOLS, ToolSpec
from agentrecall.config import Settings, get_settings
from agentrecall.curation.pipeline import CurationPipeline
from agentrecall.curation.synonyms import normalize_text
from agentrecall.llm import classify_intent, llm_factory
from agentrecall.memory.store import MemoryStore
from agentrecall.models import (
    EntryStatus,
    MemoryHit,
    MemoryRecord,
    MemoryTier,
    new_id,
    utcnow,
)


class AgentState(TypedDict, total=False):
    """Mutable state passed between LangGraph nodes."""

    request: str
    session_id: str
    memory_enabled: bool
    turn_count: int
    intent: str
    memory_hits: list[dict[str, Any]]
    context: str
    slots: dict[str, Any]
    missing_slots: list[str]
    plan: str
    chosen_tool: str | None
    tool_args: dict[str, Any]
    tool_result: dict[str, Any] | None
    answer: str
    outcome: str
    error: str
    steps: list[dict[str, Any]]
    recall_loops: int
    started_at: float
    trace_id: str
    memory_writes: list[dict[str, Any]]


_DOSE_RE = re.compile(
    r"(?:dose|exposure dose)?\s*(?:is|of|=|to|at)?\s*(\d+(?:\.\d+)?)\s*mj/cm2", re.I
)
_NA_RE = re.compile(r"(?:numerical aperture|\bna\b)\s*(?:is|of|=|to)?\s*(\d+(?:\.\d+)?)", re.I)
_DOF_RE = re.compile(r"depth of focus\s*(?:is|of|=|to)?\s*(\d+(?:\.\d+)?)\s*nm", re.I)
_LATITUDE_RE = re.compile(
    r"dose latitude\s*(?:is|of|=|to)?\s*(\d+(?:\.\d+)?)\s*(?:%|percent)", re.I
)
_ITER_RE = re.compile(r"(?:after|in)\s*(\d+)\s*iterations?|(\d+)\s*iterations?", re.I)
_EPE_RE = re.compile(r"edge placement error\s*(?:is|of|=|to)?\s*(\d+(?:\.\d+)?)\s*nm", re.I)
_PASSES_RE = re.compile(r"(\d+)\s*(?:exposure\s+)?passes?", re.I)
_LAYER_RE = re.compile(r"layer\s+([A-Za-z]*\d+[A-Za-z0-9]*)", re.I)
_STACK_RE = re.compile(r"(?:resist\s+)?stack\s+([A-Za-z0-9]+)", re.I)
_DECLARATIVE_RE = re.compile(
    r"\b(remember|use dose|dose is|dose of|set dose|na is|numerical aperture is|"
    r"depth of focus is|uses? \d+\s*(?:exposure\s+)?passes?)\b",
    re.I,
)


def _first(match: re.Match[str] | None) -> str | None:
    if not match:
        return None
    for group in match.groups():
        if group is not None:
            return group
    return None


def extract_slots(text_value: str) -> dict[str, Any]:
    """Extract numeric lithography parameters from a request or memory snippet."""
    text_value = text_value or ""
    slots: dict[str, Any] = {}
    dose = _first(_DOSE_RE.search(text_value))
    if dose is not None:
        slots["dose"] = float(dose)
    na = _first(_NA_RE.search(text_value))
    if na is not None:
        slots["na"] = float(na)
    dof = _first(_DOF_RE.search(text_value))
    if dof is not None:
        slots["dof"] = float(dof)
    latitude = _first(_LATITUDE_RE.search(text_value))
    if latitude is not None:
        slots["dose_latitude"] = float(latitude)
    iterations = _first(_ITER_RE.search(text_value))
    if iterations is not None:
        slots["iterations"] = int(iterations)
    epe = _first(_EPE_RE.search(text_value))
    if epe is not None:
        slots["epe"] = float(epe)
    passes = _first(_PASSES_RE.search(text_value))
    if passes is not None:
        slots["passes"] = int(passes)
    layer = _first(_LAYER_RE.search(text_value))
    if layer is not None:
        slots["layer"] = layer.upper()
    stack = _first(_STACK_RE.search(text_value))
    if stack is not None:
        slots["stack"] = stack.upper()

    for key in ("dose_latitude", "depth_of_focus", "dof", "dose", "na"):
        if key in slots:
            if key == "dose":
                slots["parameter"] = "dose"
            elif key == "na":
                slots["parameter"] = "numerical aperture"
            elif key == "dof":
                slots["parameter"] = "depth of focus"
            else:
                slots["parameter"] = key
            break
    return slots


def _value_in_text(value: Any, text_value: str) -> bool:
    if value is None:
        return False
    rendered = f"{value:g}" if isinstance(value, float) else str(value)
    return rendered.lower() in text_value.lower()


def _best_memory_answer(
    intent: str, hits: list[MemoryHit], slots: dict[str, Any]
) -> MemoryHit | None:
    if intent == "summarize_prior_session":
        stack = str(slots.get("stack", "")).lower()
        for hit in hits:
            if (
                hit.tier == MemoryTier.EPISODIC
                and hit.score > 0
                and (not stack or stack in hit.text.lower())
            ):
                return hit
        for hit in hits:
            if (
                hit.tier == MemoryTier.SEMANTIC
                and hit.score >= 0.2
                and (not stack or stack in hit.text.lower())
            ):
                return hit
        return None
    if intent != "lookup_recipe_parameter":
        return None
    parameter = slots.get("parameter", "parameter")
    value = slots.get(parameter if parameter != "numerical aperture" else "na")
    if parameter == "dose":
        value = slots.get("dose")
    elif parameter == "depth of focus":
        value = slots.get("dof")
    for hit in hits:
        if hit.tier != MemoryTier.SEMANTIC or hit.score < 0.2:
            continue
        text_lower = hit.text.lower()
        parameter_tokens = ["dose"] if "dose" in parameter else parameter.split()
        if not all(token in text_lower for token in parameter_tokens):
            continue
        if value is not None and not _value_in_text(value, hit.text):
            continue
        return hit
    return None


def extract_durable_fact(request: str) -> tuple[str, str] | None:
    """Recognize an explicit factual statement in a request, ignoring questions."""
    lowered = request.lower()
    if not _DECLARATIVE_RE.search(request):
        return None
    if any(lowered.lstrip().startswith(word) for word in ("what", "which", "how", "why", "does")):
        return None
    slots = extract_slots(request)
    claim = normalize_text(request)
    if not claim:
        return None
    if slots.get("stack") and slots.get("dose"):
        subject = f"resist stack {str(slots['stack']).lower()} dose"
    elif slots.get("na"):
        subject = "numerical aperture"
    elif slots.get("dof"):
        subject = "depth of focus"
    elif slots.get("layer"):
        subject = f"job deck layer {str(slots['layer']).lower()}"
    elif "remember" in lowered:
        subject = "session note"
    else:
        return None
    return subject, claim


class AgentGraph:
    """Builds and runs the seven-node LangGraph resolution workflow."""

    def __init__(
        self,
        store: MemoryStore,
        llm: BaseChatModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.store = store
        self.settings = settings or get_settings()
        self.llm = llm or llm_factory(self.settings)
        self.pipeline = CurationPipeline(
            engine=store.engine, embedder=store.embedder, llm=self.llm, settings=self.settings
        )
        self.graph = self._build()

    # ------------------------------------------------------------------ nodes
    def classify_intent_node(self, state: AgentState) -> AgentState:
        intent = classify_intent(state["request"], self.llm)
        step = {"node": "classify_intent", "detail": f"intent={intent}", "data": {}}
        return {"intent": intent, "steps": [*state.get("steps", []), step]}

    def recall_memory_node(self, state: AgentState) -> AgentState:
        loops = state.get("recall_loops", 0) + 1
        hits = self.store.recall(state["request"], state["session_id"], k=5)
        context = self.store.context_block(hits)
        hit_dicts = [
            {
                "id": hit.id,
                "tier": hit.tier.value,
                "score": round(hit.score, 6),
                "text": hit.text,
                "provenance": hit.provenance,
            }
            for hit in hits
        ]
        step = {
            "node": "recall_memory",
            "detail": f"{len(hits)} hits (loop {loops})",
            "data": {"tiers": sorted({h.tier.value for h in hits})},
        }
        return {
            "memory_hits": hit_dicts,
            "context": context,
            "recall_loops": loops,
            "steps": [*state.get("steps", []), step],
        }

    def plan_node(self, state: AgentState) -> AgentState:
        intent = state["intent"]
        request_slots = extract_slots(state["request"])
        context_slots = extract_slots(state.get("context", ""))
        slots = {**context_slots, **request_slots}
        tool_name = INTENT_TO_TOOL.get(intent)

        hits = [
            MemoryHit(
                tier=MemoryTier(hit["tier"]),
                id=hit["id"],
                text=hit.get("text", ""),
                score=float(hit.get("score", 0.0)),
                provenance=hit.get("provenance", {}),
            )
            for hit in state.get("memory_hits", [])
        ]

        if tool_name is None:
            plan = "clarify"
            missing: list[str] = []
            memory_answer = None
        else:
            memory_answer = (
                _best_memory_answer(intent, hits, slots)
                if state.get("memory_enabled", True)
                else None
            )
            if memory_answer is not None:
                plan = "answer_from_memory"
                missing = []
            else:
                required = TOOLS[tool_name].required_facts
                missing = [
                    fact for fact in required if fact != "parameter" and fact not in slots
                ]
                if fact_is_missing_prior_facts(required, hits):
                    missing.append("prior_facts")
                if (
                    missing
                    and state.get("memory_enabled", True)
                    and state.get("recall_loops", 0) <= self.settings.max_recall_loops
                ):
                    plan = "recall_again"
                else:
                    plan = "tool"
        step = {
            "node": "plan",
            "detail": f"plan={plan} tool={tool_name} missing={missing}",
            "data": {"slots": slots, "missing_slots": missing},
        }
        updates: AgentState = {
            "slots": slots,
            "missing_slots": missing,
            "plan": plan,
            "steps": [*state.get("steps", []), step],
        }
        if memory_answer is not None:
            updates["answer"] = memory_answer.text
            updates["chosen_tool"] = None
            updates["outcome"] = "answered_from_memory"
        elif plan == "clarify":
            updates["answer"] = (
                "I could not map this request to a known lithography workflow. "
                "Please name a recipe parameter, OPC convergence problem, process window, "
                "prior session, or job deck."
            )
            updates["chosen_tool"] = None
            updates["outcome"] = "clarification"
        return updates

    def route_tool_node(self, state: AgentState) -> AgentState:
        if state.get("plan") != "tool":
            step = {"node": "route_tool", "detail": "no tool required", "data": {}}
            return {"steps": [*state.get("steps", []), step]}
        intent = state["intent"]
        tool_name = INTENT_TO_TOOL[intent]
        args = self._build_args(tool_name, state)
        step = {
            "node": "route_tool",
            "detail": f"chosen_tool={tool_name}",
            "data": {"tool_args": args},
        }
        return {
            "chosen_tool": tool_name,
            "tool_args": args,
            "steps": [*state.get("steps", []), step],
        }

    def execute_node(self, state: AgentState) -> AgentState:
        tool_name = state.get("chosen_tool")
        if not tool_name:
            step = {"node": "execute", "detail": "answered from memory", "data": {}}
            return {"steps": [*state.get("steps", []), step]}
        spec: ToolSpec = TOOLS[tool_name]
        try:
            result = spec.run(state.get("tool_args", {}))
            outcome = "tool_error" if result.get("status") == "not_found" else "tool_success"
            step = {
                "node": "execute",
                "detail": f"{tool_name} -> {result.get('status')}",
                "data": {"result": result},
            }
            return {
                "tool_result": result,
                "answer": str(result.get("answer", json.dumps(result, default=str))),
                "outcome": outcome,
                "steps": [*state.get("steps", []), step],
            }
        except Exception as exc:  # defensive: a bad tool must still yield a trace
            step = {
                "node": "execute",
                "detail": f"{tool_name} failed: {exc}",
                "data": {"error": str(exc)},
            }
            return {
                "tool_result": None,
                "answer": f"The {tool_name} tool failed: {exc}",
                "outcome": "tool_error",
                "error": str(exc),
                "steps": [*state.get("steps", []), step],
            }

    def reflect_and_write_node(self, state: AgentState) -> AgentState:
        if not state.get("memory_enabled", True):
            step = {"node": "reflect_and_write", "detail": "memory disabled; skipped", "data": {}}
            return {"steps": [*state.get("steps", []), step]}
        writes: list[dict[str, Any]] = []
        fact = extract_durable_fact(state["request"])
        if fact is not None:
            writes.append(self._write_fact(fact, state["session_id"]))
        result = state.get("tool_result") or {}
        durable = result.get("durable_fact")
        if durable:
            writes.append(
                self._write_fact(
                    (durable["subject"], normalize_text(durable["claim"])),
                    state["session_id"],
                    confidence=float(durable.get("confidence", 0.75)),
                )
            )
        step = {
            "node": "reflect_and_write",
            "detail": f"{len(writes)} memory writes",
            "data": {"writes": writes},
        }
        return {
            "memory_writes": writes,
            "steps": [*state.get("steps", []), step],
        }

    def summarize_turn_node(self, state: AgentState) -> AgentState:
        steps = list(state.get("steps", []))
        if state.get("memory_enabled", True):
            self.store.scratchpad.append(state["session_id"], "user", state["request"])
            self.store.scratchpad.append(
                state["session_id"], "assistant", state.get("answer", "")
            )
            turn_count = state.get("turn_count", 1)
            if turn_count % self.settings.summarize_every_turns == 0:
                turns = self.store.scratchpad.window(state["session_id"], 20)
                self.store.episodic.summarize_session(
                    state["session_id"], turns, 0, max(0, len(turns) - 1)
                )
                steps.append(
                    {
                        "node": "summarize_turn",
                        "detail": f"episodic summary written (turn {turn_count})",
                        "data": {},
                    }
                )
            else:
                steps.append(
                    {"node": "summarize_turn", "detail": "scratchpad updated", "data": {}}
                )
        else:
            steps.append(
                {"node": "summarize_turn", "detail": "memory disabled; trace only", "data": {}}
            )

        trace_id = new_id()
        started = state.get("started_at", time.perf_counter())
        latency_ms = max(0, int((time.perf_counter() - started) * 1000))
        with self.store.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO resolution_traces "
                    "(id, session_id, request, intent, chosen_tool, memory_hits, steps, "
                    "outcome, latency_ms, created_at) VALUES "
                    "(CAST(:id AS uuid), :session_id, :request, :intent, :chosen_tool, "
                    "CAST(:memory_hits AS jsonb), CAST(:steps AS jsonb), :outcome, "
                    ":latency_ms, :created_at)"
                ),
                {
                    "id": trace_id,
                    "session_id": state["session_id"],
                    "request": state["request"],
                    "intent": state["intent"],
                    "chosen_tool": state.get("chosen_tool"),
                    "memory_hits": json.dumps(state.get("memory_hits", []), default=str),
                    "steps": json.dumps(steps, default=str),
                    "outcome": state.get("outcome", "unknown"),
                    "latency_ms": latency_ms,
                    "created_at": utcnow(),
                },
            )
        return {"steps": steps, "trace_id": trace_id}

    # ----------------------------------------------------------------- graph
    def _build(self):
        builder = StateGraph(AgentState)
        builder.add_node("classify_intent", self.classify_intent_node)
        builder.add_node("recall_memory", self.recall_memory_node)
        builder.add_node("plan", self.plan_node)
        builder.add_node("route_tool", self.route_tool_node)
        builder.add_node("execute", self.execute_node)
        builder.add_node("reflect_and_write", self.reflect_and_write_node)
        builder.add_node("summarize_turn", self.summarize_turn_node)

        builder.add_edge(START, "classify_intent")
        builder.add_conditional_edges(
            "classify_intent",
            lambda s: "recall_memory" if s.get("memory_enabled", True) else "plan",
            {"recall_memory": "recall_memory", "plan": "plan"},
        )
        builder.add_edge("recall_memory", "plan")
        builder.add_conditional_edges(
            "plan",
            lambda s: "recall_memory" if s.get("plan") == "recall_again" else "route_tool",
            {"recall_memory": "recall_memory", "route_tool": "route_tool"},
        )
        builder.add_edge("route_tool", "execute")
        builder.add_edge("execute", "reflect_and_write")
        builder.add_edge("reflect_and_write", "summarize_turn")
        builder.add_edge("summarize_turn", END)
        return builder.compile()

    # --------------------------------------------------------------- helpers
    def invoke(self, state: AgentState) -> AgentState:
        """Run the compiled graph."""
        return self.graph.invoke(state)

    def _build_args(self, tool_name: str, state: AgentState) -> dict[str, Any]:
        slots = state.get("slots", {})
        parameter = str(slots.get("parameter", "parameter"))
        observed = None
        unit = ""
        if parameter == "dose" and "dose" in slots:
            observed, unit = slots["dose"], "mJ/cm2"
        elif parameter == "numerical aperture" and "na" in slots:
            observed, unit = slots["na"], ""
        elif parameter == "dof" and "dof" in slots:
            observed, unit = slots["dof"], "nm"
        source = next(
            (
                hit["provenance"].get("source_doc_id") or hit["provenance"].get("session_id")
                for hit in state.get("memory_hits", [])
                if hit.get("tier") == "semantic"
            ),
            None,
        )
        if tool_name == "lookup_recipe_parameter":
            return {
                "parameter": parameter,
                "stack": str(slots.get("stack", "")),
                "observed_value": observed,
                "unit": unit,
                "source": source,
            }
        if tool_name == "diagnose_opc_convergence":
            return {
                "iterations": int(slots.get("iterations", 0)),
                "epe_nm": float(slots.get("epe", 2.0)),
                "epe_threshold_nm": 1.5,
            }
        if tool_name == "compute_process_window":
            return {
                "dose_center_mj_cm2": float(slots.get("dose", 32.0)),
                "dose_latitude_pct": float(slots.get("dose_latitude", 0.0)),
                "depth_of_focus_nm": float(slots.get("dof", 0.0)),
                "min_depth_of_focus_nm": 80.0,
            }
        if tool_name == "summarize_prior_session":
            return {
                "session_id": state["session_id"],
                "facts": [hit.get("text", "") for hit in state.get("memory_hits", [])],
                "focus": str(slots.get("parameter", "")),
            }
        return {
            "layer": str(slots.get("layer", "M1")),
            "passes": int(slots.get("passes", 1)),
            "dose_mj_cm2": float(slots.get("dose", 32.0)),
            "stack": str(slots.get("stack", "")),
        }

    def _write_fact(
        self, fact: tuple[str, str], session_id: str, confidence: float = 0.75
    ) -> dict[str, Any]:
        subject, claim = fact
        record = MemoryRecord(
            tier=MemoryTier.SEMANTIC,
            subject=subject,
            claim=claim,
            body=claim,
            session_id=session_id,
            confidence=confidence,
            status=EntryStatus.ACTIVE,
            provenance={
                "sources": [],
                "session_id": session_id,
                "origin": "agent-reflect",
            },
        )
        decision = self.pipeline.ingest(record)
        return {
            "action": decision.action,
            "id": decision.target_id,
            "subject": subject,
            "claim": claim,
        }


def fact_is_missing_prior_facts(required: tuple[str, ...], hits: list[MemoryHit]) -> bool:
    """A summarize tool needs at least one retrieved memory snippet."""
    return "prior_facts" in required and not hits
