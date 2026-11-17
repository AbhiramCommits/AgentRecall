"""Seeded 40-task multi-session benchmark over computational-lithography support work.

Each task is a three-session script: session 1 states a fact, session 2 does unrelated
work, and session 3 asks a question that can only be answered by retaining session-1
knowledge. Every task carries an exact expected answer fragment for automatic grading.
"""

from __future__ import annotations

from pydantic import BaseModel

DISTRACTOR = "Check the reticle stage maintenance log for belt tension."


class EvalTask(BaseModel):
    """One three-session memory task with its exact answer key."""

    id: str
    intent: str
    group: str
    subject: str
    session1_fact: str
    distractor: str
    question: str
    expected_answer: str


def build_tasks(seed: int = 7) -> list[EvalTask]:
    """Build the deterministic 40-task benchmark."""
    tasks: list[EvalTask] = []

    for index in range(8):
        stack = f"P{index + 1:02d}"
        value = 41 + (seed + index) % 8
        tasks.append(
            EvalTask(
                id=f"dose-{index:02d}",
                intent="lookup_recipe_parameter",
                group="dose",
                subject=f"pilot stack {stack} dose",
                session1_fact=(
                    f"Remember: for pilot stack {stack}, the qualified dose is {value} "
                    f"mJ/cm2 at 193 nm."
                ),
                distractor=DISTRACTOR,
                question=f"What is the qualified dose for pilot stack {stack}?",
                expected_answer=f"{value} mj/cm2",
            )
        )

    for index in range(8):
        task_index = 8 + index
        stack = f"P{task_index + 1:02d}"
        na = 1.31 + ((seed + index) % 8) * 0.01
        tasks.append(
            EvalTask(
                id=f"na-{index:02d}",
                intent="lookup_recipe_parameter",
                group="numerical_aperture",
                subject=f"pilot stack {stack} numerical aperture",
                session1_fact=(
                    f"Remember: for pilot stack {stack}, the numerical aperture is "
                    f"{na:.2f} at 193 nm."
                ),
                distractor=DISTRACTOR,
                question=f"What is the numerical aperture for pilot stack {stack}?",
                expected_answer=f"{na:.2f}",
            )
        )

    for index in range(8):
        task_index = 16 + index
        stack = f"P{task_index + 1:02d}"
        latitude = 5 + (seed + index) % 8
        dof = 85 + ((seed + index) * 5) % 40
        area = latitude * dof / 100.0
        tasks.append(
            EvalTask(
                id=f"window-{index:02d}",
                intent="compute_process_window",
                group="process_window",
                subject=f"pilot stack {stack} process window",
                session1_fact=(
                    f"Remember: pilot stack {stack} uses dose latitude {latitude} percent "
                    f"and depth of focus {dof} nm at dose 32 mJ/cm2."
                ),
                distractor=DISTRACTOR,
                question=(
                    f"Compute the process window for pilot stack {stack} using the "
                    "qualified latitude and focus."
                ),
                expected_answer=f"{area:.2f}",
            )
        )

    for index in range(8):
        task_index = 24 + index
        stack = f"P{task_index + 1:02d}"
        layer = f"M{task_index + 1}"
        passes = 2 + (seed + index) % 4
        tasks.append(
            EvalTask(
                id=f"deck-{index:02d}",
                intent="draft_job_deck",
                group="job_deck",
                subject=f"job deck layer {layer.lower()}",
                session1_fact=(
                    f"Remember: pilot stack {stack} job deck for layer {layer} uses "
                    f"{passes} exposure passes at dose 33 mJ/cm2."
                ),
                distractor=DISTRACTOR,
                question=f"Draft a job deck for layer {layer} for pilot stack {stack}.",
                expected_answer=f"{passes} exposure pass",
            )
        )

    diagnose_cases = [(4.0, "unstable"), (2.2, "slow_convergence"),
                      (1.2, "converged"), (2.8, "slow_convergence")]
    for index, (epe, diagnosis) in enumerate(diagnose_cases):
        task_index = 32 + index
        stack = f"P{task_index + 1:02d}"
        iterations = 3 + index
        tasks.append(
            EvalTask(
                id=f"opc-{index:02d}",
                intent="diagnose_opc_convergence",
                group="opc",
                subject=f"pilot stack {stack} opc convergence",
                session1_fact=(
                    f"Remember: pilot stack {stack} OPC calibration stopped after "
                    f"{iterations} iterations with edge placement error {epe:g} nm."
                ),
                distractor=DISTRACTOR,
                question=f"Diagnose the OPC convergence status for pilot stack {stack}.",
                expected_answer=diagnosis,
            )
        )

    for index in range(4):
        task_index = 36 + index
        stack = f"P{task_index + 1:02d}"
        value = 46 + (seed + index) % 5
        tasks.append(
            EvalTask(
                id=f"summary-{index:02d}",
                intent="summarize_prior_session",
                group="session_summary",
                subject=f"pilot stack {stack} dose",
                session1_fact=(
                    f"Remember: for pilot stack {stack}, the qualified dose is {value} "
                    f"mJ/cm2 with numerical aperture 1.33 at 193 nm."
                ),
                distractor=DISTRACTOR,
                question=f"Summarize the prior session for pilot stack {stack}.",
                expected_answer=f"{value} mj/cm2",
            )
        )

    return tasks
