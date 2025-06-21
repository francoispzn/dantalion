"""The agent loop.

This is the first, deliberately simple shape of the loop: send the conversation
to the model with the available tools, run whatever tools it asks for, feed the
results back, and repeat until the model answers in plain text or we hit the step
ceiling. It is the classic act/observe cycle and nothing more.

Budgets, cancellation, planning and self-critique arrive in later milestones and
wrap *around* this core; keeping the core small is what makes those additions
clean rather than tangled.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dantalion.agent.result import RunResult, Step, ToolInvocation
from dantalion.errors import ToolNotFound
from dantalion.providers.base import Provider
from dantalion.tools.base import ToolResult
from dantalion.tools.registry import ToolRegistry
from dantalion.types import CompletionRequest, Message, ToolCall, Usage

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful investigator. Work step by step. Use the provided tools to "
    "gather evidence before drawing conclusions, and prefer checking a fact over "
    "guessing it. When you have enough evidence, answer directly and concisely."
)


class Agent:
    """A minimal tool-using agent over a single provider."""

    def __init__(
        self,
        provider: Provider,
        tools: ToolRegistry | None = None,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 8,
        temperature: float = 0.2,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools or ToolRegistry()
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.temperature = temperature
        self.context = context or {}

    def run(self, task: str) -> RunResult:
        """Investigate ``task`` and return the full trail."""
        messages: list[Message] = [
            Message.system(self.system_prompt),
            Message.user(task),
        ]
        specs = self.tools.specs()
        steps: list[Step] = []
        usage = Usage()

        for index in range(self.max_steps):
            response = self.provider.complete(
                CompletionRequest(messages=messages, tools=specs, temperature=self.temperature)
            )
            usage += response.usage
            messages.append(response.message)

            if not response.message.tool_calls:
                return RunResult(
                    output=response.message.content,
                    messages=messages,
                    steps=steps,
                    usage=usage,
                    finished=True,
                    stop_reason="completed",
                )

            invocations = [self._invoke(call) for call in response.message.tool_calls]
            for invocation in invocations:
                messages.append(
                    Message.tool(
                        invocation.result.to_content(),
                        tool_call_id=invocation.call.id,
                        name=invocation.call.name,
                    )
                )
            steps.append(Step(index=index, message=response.message, invocations=invocations))

        return RunResult(
            output=None,
            messages=messages,
            steps=steps,
            usage=usage,
            finished=False,
            stop_reason="max_steps",
        )

    def _invoke(self, call: ToolCall) -> ToolInvocation:
        try:
            tool = self.tools.get(call.name)
        except ToolNotFound as exc:
            return ToolInvocation(call=call, result=ToolResult(ok=False, error=str(exc)))
        return ToolInvocation(call=call, result=tool.run(call.arguments, context=self.context))
