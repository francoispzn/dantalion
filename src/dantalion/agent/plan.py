"""The planner: a quick first pass before any tools run.

Asking the model to sketch a short plan up front measurably steadies smaller
local models — it commits them to an approach instead of flailing tool-to-tool.
The plan is advisory, injected into the conversation as context; the executor is
free to deviate as evidence comes in. It is produced through the structured-output
layer, so even a model without a JSON mode returns a clean list of steps.
"""

from __future__ import annotations

from dantalion.agent.result import Plan
from dantalion.providers.base import Provider
from dantalion.structured import StructuredResult, structure
from dantalion.types import Message

_PLANNER_SYSTEM = (
    "You are planning an investigation. Given the task, produce a short ordered "
    "list of 3 to 6 concrete steps that gather evidence before reaching a "
    "conclusion. Keep each step to a single sentence."
)


def make_plan(provider: Provider, task: str, *, temperature: float = 0.0) -> StructuredResult[Plan]:
    """Draft an investigation plan for ``task``."""
    messages = [
        Message.system(_PLANNER_SYSTEM),
        Message.user(f"Task: {task}"),
    ]
    return structure(provider, messages, Plan, temperature=temperature)


def format_plan(plan: Plan) -> str:
    """Render a plan as a numbered list for injection into the conversation."""
    return "\n".join(f"{i}. {step}" for i, step in enumerate(plan.steps, start=1))
