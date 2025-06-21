"""What a run produces.

These are plain records, deliberately rich: a run is not just its final answer
but the whole reasoning trail — every assistant turn, every tool call, and what
each tool returned. The CLI prints a summary, the evaluator scores the trail, and
the tracer can reconstruct it. Keeping all of it makes the agent auditable rather
than a black box.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dantalion.tools.base import ToolResult
from dantalion.types import Message, ToolCall, Usage


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

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Every tool call made during the run, in order."""
        return [inv.call for step in self.steps for inv in step.invocations]
