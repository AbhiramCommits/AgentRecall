"""Deterministic rule-based chat model plus a factory for real LLM backends.

``StubChatModel`` is a LangChain ``BaseChatModel`` that produces reproducible output for
three task families used by AgentRecall: turn summarization, contradiction adjudication,
and intent classification. Prompts carry a ``[TASK:...]`` marker so the stub can dispatch
without any network access. ``llm_factory`` returns an OpenAI-compatible chat model when
``AGENTRECALL_LLM_BASE_URL`` is configured, otherwise the stub.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agentrecall.config import Settings, get_settings

INTENTS: tuple[str, ...] = (
    "lookup_recipe_parameter",
    "diagnose_opc_convergence",
    "compute_process_window",
    "summarize_prior_session",
    "draft_job_deck",
    "unknown",
)

_TASK_RE = re.compile(r"\[TASK:([A-Z_]+)\]")

_INTENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "summarize_prior_session",
        (
            "summarize",
            "summarise",
            "recap",
            "prior session",
            "last session",
            "previous session",
            "what did we",
            "remind me what we",
        ),
    ),
    (
        "diagnose_opc_convergence",
        (
            "converg",
            "diverge",
            "oscillat",
            "debug",
            "failing",
            "failed",
            "why does",
            "why is",
            "runtime error",
            "not converging",
            "stuck",
        ),
    ),
    (
        "compute_process_window",
        (
            "process window",
            "dose latitude",
            "depth of focus",
            "exposure latitude",
            "compute",
            "calculate",
            "window",
        ),
    ),
    (
        "draft_job_deck",
        (
            "draft",
            "job deck for",
            "job-deck for",
            "deck skeleton",
            "write a job deck",
            "create a job deck",
            "write a recipe",
        ),
    ),
    (
        "lookup_recipe_parameter",
        (
            "what is",
            "what's",
            "value",
            "parameter",
            "aperture",
            "numerical aperture",
            " na ",
            "dose",
            "threshold",
            "resist",
            "recipe",
            "which",
        ),
    ),
]

_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")
_SUMMARY_FACT_RE = re.compile(
    r"\d|\b(must|use|set|remember|prefer|avoid|never|always|requires?|should)\b", re.I
)


def _messages_to_text(messages: list[BaseMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        role = getattr(message, "type", "human")
        content = message.content if isinstance(message.content, str) else str(message.content)
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _extract_summary(text: str) -> str:
    """Select factual sentences (numbers, imperatives) and compress them deterministically."""
    body = _TASK_RE.sub("", text)
    facts: list[str] = []
    for sentence in _SENTENCE_RE.findall(body):
        clean = " ".join(sentence.split()).strip(" :-")
        if not clean or clean.lower().startswith(("system", "human", "user", "assistant")):
            continue
        if _SUMMARY_FACT_RE.search(clean):
            facts.append(clean)
        if len(facts) >= 6:
            break
    if not facts:
        facts = [" ".join(s.strip() for s in _SENTENCE_RE.findall(body) if s.strip())[:240]]
    summary = "Session summary: " + " ".join(facts)
    return summary[:600].strip()


def _adjudicate(text: str) -> str:
    """Resolve A-versus-B conflict JSON by confidence, then by provenance recency."""
    start = text.find("{")
    payload: dict[str, Any] = {}
    if start >= 0:
        try:
            payload = json.loads(text[start:])
        except json.JSONDecodeError:
            payload = {}
    a = payload.get("a", {}) or {}
    b = payload.get("b", {}) or {}
    conf_a = float(a.get("confidence", 0.0))
    conf_b = float(b.get("confidence", 0.0))
    created_a = str(a.get("created_at", ""))
    created_b = str(b.get("created_at", ""))
    if abs(conf_a - conf_b) >= 0.05:
        winner = "A" if conf_a > conf_b else "B"
        reason = f"higher confidence ({max(conf_a, conf_b):.2f} vs {min(conf_a, conf_b):.2f})"
    else:
        winner = "A" if created_a >= created_b else "B"
        reason = "newer provenance at comparable confidence"
    return json.dumps({"winner": winner, "reason": reason}, sort_keys=True)


def rule_based_intent(request: str) -> str:
    """Classify a request with deterministic keyword precedence rules."""
    text = f" {request.lower().strip()} "
    for intent, keywords in _INTENT_RULES:
        if any(keyword in text for keyword in keywords):
            return intent
    return "unknown"


class StubChatModel(BaseChatModel):
    """Deterministic, offline chat model used for every default AgentRecall workflow."""

    model_name: str = "agentrecall-stub"

    @property
    def _llm_type(self) -> str:
        return "agentrecall-stub"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = _messages_to_text(messages)
        match = _TASK_RE.search(text)
        task = match.group(1) if match else "GENERAL"
        if task == "SUMMARIZE":
            content = _extract_summary(text)
        elif task == "ADJUDICATE":
            content = _adjudicate(text)
        elif task == "CLASSIFY":
            request = text.split("Request:", 1)[-1] if "Request:" in text else text
            content = rule_based_intent(request)
        else:
            content = "StubChatModel: " + " ".join(text.split())[:200]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def llm_factory(settings: Settings | None = None) -> BaseChatModel:
    """Return a real OpenAI-compatible model when configured, else the deterministic stub."""
    settings = settings or get_settings()
    if settings.llm_base_url:
        from langchain_openai import ChatOpenAI  # imported lazily so offline tests need no key

        return ChatOpenAI(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            temperature=0,
        )
    return StubChatModel()


def classify_intent(request: str, llm: BaseChatModel | None = None) -> str:
    """Classify a natural-language request into one of ``INTENTS`` via the chat model."""
    model = llm or llm_factory()
    prompt = (
        "[TASK:CLASSIFY]\n"
        "Classify the engineering request into exactly one intent from: "
        + ", ".join(INTENTS)
        + f".\nRequest: {request}"
    )
    response = model.invoke(prompt)
    intent = str(response.content).strip().splitlines()[0].strip()
    return intent if intent in INTENTS else "unknown"


def summarize_turns(turns: list[dict[str, Any]], llm: BaseChatModel | None = None) -> str:
    """Summarize a list of role/text turns deterministically through the chat model."""
    model = llm or llm_factory()
    lines = "\n".join(f"{t.get('role', 'user')}: {t.get('text', '')}" for t in turns)
    response = model.invoke(f"[TASK:SUMMARIZE]\n{lines}")
    return str(response.content).strip()


def adjudicate(a: dict[str, Any], b: dict[str, Any], llm: BaseChatModel | None = None) -> dict:
    """Ask the model to pick the winning version of a contradictory pair."""
    model = llm or llm_factory()
    prompt = "[TASK:ADJUDICATE]\n" + json.dumps({"a": a, "b": b}, default=str, sort_keys=True)
    response = model.invoke(prompt)
    try:
        parsed = json.loads(str(response.content))
    except json.JSONDecodeError:
        parsed = {"winner": "A", "reason": "unparseable adjudication; defaulted to A"}
    return parsed
