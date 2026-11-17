"""Five deterministic lithography-support tools with Pydantic argument schemas.

Every tool is offline, has a docstring, declares the memory facts it needs, and returns a
structured dictionary. No tool performs network I/O; real integrations would replace the
bodies while keeping the same contracts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class LookupRecipeParameterArgs(BaseModel):
    """Arguments for looking up a qualified recipe parameter."""

    parameter: str = Field(description="Parameter name, e.g. 'dose' or 'numerical aperture'.")
    stack: str = Field(default="", description="Resist stack or layer identifier.")
    observed_value: float | None = Field(default=None, description="Value recalled from memory.")
    unit: str = Field(default="", description="Unit for the recalled value.")
    source: str | None = Field(default=None, description="Memory or source-document id.")


class DiagnoseOpcConvergenceArgs(BaseModel):
    """Arguments for diagnosing an OPC model convergence complaint."""

    iterations: int = Field(default=0, ge=0, description="Iterations completed before failure.")
    epe_nm: float = Field(default=2.0, ge=0, description="Current edge-placement error in nm.")
    epe_threshold_nm: float = Field(default=1.5, gt=0, description="Acceptance threshold in nm.")


class ComputeProcessWindowArgs(BaseModel):
    """Arguments for a deterministic process-window calculation."""

    dose_center_mj_cm2: float = Field(default=32.0, gt=0)
    dose_latitude_pct: float = Field(default=0.0, ge=0)
    depth_of_focus_nm: float = Field(default=0.0, ge=0)
    min_depth_of_focus_nm: float = Field(default=80.0, gt=0)


class SummarizePriorSessionArgs(BaseModel):
    """Arguments for summarizing facts retrieved from earlier sessions."""

    session_id: str
    facts: list[str] = Field(default_factory=list, description="Retrieved memory snippets.")
    focus: str = Field(default="", description="Optional topic to prioritize.")


class DraftJobDeckArgs(BaseModel):
    """Arguments for drafting a job-deck skeleton."""

    layer: str = Field(default="M1")
    passes: int = Field(default=1, ge=1, le=12)
    dose_mj_cm2: float = Field(default=32.0, gt=0)
    stack: str = Field(default="")


@dataclass(frozen=True)
class ToolSpec:
    """A registered tool: schema, required memory facts, and deterministic implementation."""

    name: str
    description: str
    args_model: type[BaseModel]
    required_facts: tuple[str, ...]
    func: Callable[..., dict[str, Any]]

    def run(self, args: dict[str, Any]) -> dict[str, Any]:
        """Validate arguments against the Pydantic schema, then execute."""
        validated = self.args_model(**args)
        return self.func(**validated.model_dump())


def lookup_recipe_parameter(
    parameter: str,
    stack: str = "",
    observed_value: float | None = None,
    unit: str = "",
    source: str | None = None,
) -> dict[str, Any]:
    """Answer a recipe-parameter question from a recalled value, or report it missing."""
    label = f"{parameter} for {stack}" if stack else parameter
    if observed_value is None:
        return {
            "status": "not_found",
            "parameter": parameter,
            "stack": stack,
            "answer": f"No stored value for {label}; recall another session or re-measure.",
            "source": source,
        }
    return {
        "status": "found",
        "parameter": parameter,
        "stack": stack,
        "value": observed_value,
        "unit": unit,
        "source": source,
        "answer": f"{label} is {observed_value:g} {unit}".strip(),
    }


def diagnose_opc_convergence(
    iterations: int = 0,
    epe_nm: float = 2.0,
    epe_threshold_nm: float = 1.5,
) -> dict[str, Any]:
    """Classify an OPC convergence failure into a deterministic diagnosis and next action."""
    if iterations <= 0:
        diagnosis = "no_progress"
        recommendation = "verify the model started and rerun with a smaller step size"
    elif epe_nm > epe_threshold_nm * 2:
        diagnosis = "unstable"
        recommendation = "reduce the update step size and re-run model calibration"
    elif epe_nm > epe_threshold_nm:
        diagnosis = "slow_convergence"
        recommendation = "add iterations or tighten the EPE target before locking the model"
    else:
        diagnosis = "converged"
        recommendation = "lock the model and record the calibration provenance"
    return {
        "status": "diagnosed",
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "iterations": iterations,
        "epe_nm": epe_nm,
        "epe_threshold_nm": epe_threshold_nm,
        "answer": (
            f"OPC diagnosis: {diagnosis} after {iterations} iterations at {epe_nm:g} nm EPE; "
            f"{recommendation}."
        ),
    }


def compute_process_window(
    dose_center_mj_cm2: float = 32.0,
    dose_latitude_pct: float = 0.0,
    depth_of_focus_nm: float = 0.0,
    min_depth_of_focus_nm: float = 80.0,
) -> dict[str, Any]:
    """Compute a rectangular process-window area and a deterministic robustness verdict."""
    area = (dose_latitude_pct / 100.0) * depth_of_focus_nm
    robust = dose_latitude_pct >= 5.0 and depth_of_focus_nm >= min_depth_of_focus_nm
    verdict = "robust" if robust else "marginal"
    claim = (
        f"process window at {dose_center_mj_cm2:g} mj/cm2 is {area:.2f} pct*nm "
        f"({verdict}) with dose latitude {dose_latitude_pct:g}% and "
        f"depth of focus {depth_of_focus_nm:g} nm"
    )
    return {
        "status": "computed",
        "window_area_pct_nm": round(area, 4),
        "verdict": verdict,
        "dose_center_mj_cm2": dose_center_mj_cm2,
        "dose_latitude_pct": dose_latitude_pct,
        "depth_of_focus_nm": depth_of_focus_nm,
        "durable_fact": {
            "subject": "process window",
            "claim": claim,
            "confidence": 0.8,
        },
        "answer": (
            f"Process window area is {area:.2f} %·nm at {dose_center_mj_cm2:g} mJ/cm2; "
            f"verdict: {verdict}."
        ),
    }


def summarize_prior_session(
    session_id: str, facts: list[str] | None = None, focus: str = ""
) -> dict[str, Any]:
    """Summarize retrieved facts into a compact prior-session briefing."""
    facts = facts or []
    selected = facts[:5]
    if focus:
        focused = [fact for fact in facts if focus.lower() in fact.lower()]
        selected = (focused + [fact for fact in facts if fact not in focused])[:5]
    summary = "; ".join(selected) if selected else "No prior-session facts were available."
    return {
        "status": "summarized",
        "session_id": session_id,
        "fact_count": len(selected),
        "summary": summary,
        "answer": f"Prior session {session_id}: {summary}",
    }


def draft_job_deck(
    layer: str = "M1",
    passes: int = 1,
    dose_mj_cm2: float = 32.0,
    stack: str = "",
) -> dict[str, Any]:
    """Draft a deterministic job-deck skeleton for a lithography layer."""
    stack_line = f"STACK {stack}\n" if stack else ""
    deck = (
        f"JOB deck for layer {layer}\n"
        f"{stack_line}"
        f"EXPOSURES {passes}\n"
        f"DOSE {dose_mj_cm2:g} mJ/cm2\n"
        "DOSE_MONITOR every 5 lots\n"
        "ALIGNMENT reticle\n"
    )
    return {
        "status": "drafted",
        "layer": layer,
        "passes": passes,
        "dose_mj_cm2": dose_mj_cm2,
        "deck": deck,
        "answer": f"Drafted job deck for layer {layer} with {passes} exposure pass(es).",
    }


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            name="lookup_recipe_parameter",
            description="Look up a qualified recipe parameter recalled from memory.",
            args_model=LookupRecipeParameterArgs,
            required_facts=("parameter",),
            func=lookup_recipe_parameter,
        ),
        ToolSpec(
            name="diagnose_opc_convergence",
            description="Diagnose why an OPC calibration is not converging.",
            args_model=DiagnoseOpcConvergenceArgs,
            required_facts=("iterations",),
            func=diagnose_opc_convergence,
        ),
        ToolSpec(
            name="compute_process_window",
            description="Compute a rectangular dose/focus process window.",
            args_model=ComputeProcessWindowArgs,
            required_facts=("dose_latitude", "dof"),
            func=compute_process_window,
        ),
        ToolSpec(
            name="summarize_prior_session",
            description="Summarize facts retained from an earlier session.",
            args_model=SummarizePriorSessionArgs,
            required_facts=("prior_facts",),
            func=summarize_prior_session,
        ),
        ToolSpec(
            name="draft_job_deck",
            description="Draft a job-deck skeleton for a lithography layer.",
            args_model=DraftJobDeckArgs,
            required_facts=("layer",),
            func=draft_job_deck,
        ),
    )
}

INTENT_TO_TOOL: dict[str, str] = {
    "lookup_recipe_parameter": "lookup_recipe_parameter",
    "diagnose_opc_convergence": "diagnose_opc_convergence",
    "compute_process_window": "compute_process_window",
    "summarize_prior_session": "summarize_prior_session",
    "draft_job_deck": "draft_job_deck",
}
