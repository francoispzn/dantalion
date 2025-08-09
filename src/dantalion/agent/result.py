"""What a run produces.

These are plain records, deliberately rich: a run is not just its final answer
but the whole reasoning trail — the plan it drew up, every assistant turn, every
tool call and its result, and any critiques of the proposed answer. The CLI
prints a summary, the evaluator scores the trail, and the tracer can reconstruct
it. Keeping all of it makes the agent auditable rather than a black box.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from dantalion.tools.base import ToolResult
from dantalion.types import Message, ToolCall, Usage


class Plan(BaseModel):
    """An ordered list of steps the agent intends to take."""

    steps: list[str]


class Critique(BaseModel):
    """A reviewer's judgement on whether a proposed answer is well supported."""

    sufficient: bool
    reasoning: str
    guidance: str = ""


@dataclass(frozen=True)
class ToolInvocation:
    """One tool call and the result it produced."""

    call: ToolCall
    result: ToolResult


@dataclass(frozen=True)
class Step:
    """One iteration of the loop: an assistant turn and any tools it ran."""

    index: int
    message: Message
    invocations: list[ToolInvocation] = field(default_factory=list)


@dataclass
class RunResult:
    """The full outcome of a run."""

    output: str | None
    messages: list[Message]
    steps: list[Step] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finished: bool = True
    stop_reason: str = "completed"
    plan: Plan | None = None
    critiques: list[Critique] = field(default_factory=list)

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Every tool call made during the run, in order."""
        return [inv.call for step in self.steps for inv in step.invocations]
