"""The agent loop.

The loop is a plan → act/observe → review cycle:

* an optional **planner** sketches an approach before any tool runs;
* the **executor** calls the model with the available tools, runs whatever it
  asks for, and feeds the results back, repeating until the model answers;
* an optional **critic** then checks the answer against the evidence and, if it
  falls short, sends the agent back for another round.

Every iteration is bounded by a :class:`Budget` and can be cancelled between
steps, so the loop always terminates with a coherent result. Planning and review
are off by default — the bare loop stays simple, and richer behaviour is opt-in.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dantalion.agent.budget import Budget, CancellationToken
from dantalion.agent.critic import review_answer
from dantalion.agent.plan import format_plan, make_plan
from dantalion.agent.result import Critique, Plan, RunResult, Step, ToolInvocation
from dantalion.errors import ToolNotFound
from dantalion.providers.base import Provider
from dantalion.tools.base import ToolResult
from dantalion.tools.registry import ToolRegistry
from dantalion.types import CompletionRequest, Message, ToolCall

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful investigator. Work step by step. Use the provided tools to "
    "gather evidence before drawing conclusions, and prefer checking a fact over "
    "guessing it. When you have enough evidence, answer directly and concisely."
)


class Agent:
    """A tool-using agent with optional planning, self-review, and budgets."""

    def __init__(
        self,
        provider: Provider,
        tools: ToolRegistry | None = None,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 8,
        max_tokens: int | None = None,
        max_seconds: float | None = None,
        temperature: float = 0.2,
        context: Mapping[str, Any] | None = None,
        plan: bool = False,
        review: bool = False,
        max_reviews: int = 1,
        cancellation: CancellationToken | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools or ToolRegistry()
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.max_seconds = max_seconds
        self.temperature = temperature
        self.context = context or {}
        self.plan = plan
        self.review = review
        self.max_reviews = max_reviews
        self.cancellation = cancellation

    def run(self, task: str) -> RunResult:
        """Investigate ``task`` and return the full trail."""
        budget = Budget(
            max_steps=self.max_steps, max_tokens=self.max_tokens, max_seconds=self.max_seconds
        )
        budget.start()
        cancel = self.cancellation or CancellationToken()

        messages: list[Message] = [Message.system(self.system_prompt), Message.user(task)]
        plan = self._draft_plan(task, messages, budget)
        specs = self.tools.specs()
        steps: list[Step] = []
        critiques: list[Critique] = []
        reviews = 0

        while True:
            if cancel.cancelled:
                return self._result(
                    None, messages, steps, budget, False, "cancelled", plan, critiques
                )
            breached = budget.exceeded()
            if breached is not None:
                return self._result(None, messages, steps, budget, False, breached, plan, critiques)

            response = self.provider.complete(
                CompletionRequest(messages=messages, tools=specs, temperature=self.temperature)
            )
            budget.add_usage(response.usage)
            budget.tick()
            messages.append(response.message)

            if response.message.tool_calls:
                invocations = [self._invoke(call) for call in response.message.tool_calls]
                for invocation in invocations:
                    messages.append(
                        Message.tool(
                            invocation.result.to_content(),
                            tool_call_id=invocation.call.id,
                            name=invocation.call.name,
                        )
                    )
                steps.append(
                    Step(index=len(steps), message=response.message, invocations=invocations)
                )
                continue

            answer = response.message.content
            if self.review and reviews < self.max_reviews:
                critique = review_answer(
                    self.provider, task, messages, answer or "", temperature=self.temperature
                )
                budget.add_usage(critique.usage)
                critiques.append(critique.value)
                reviews += 1
                if not critique.value.sufficient:
                    messages.append(
                        Message.user(
                            "A reviewer judged that answer insufficient: "
                            f"{critique.value.guidance} Keep investigating, then answer again."
                        )
                    )
                    continue

            return self._result(answer, messages, steps, budget, True, "completed", plan, critiques)

    # -- helpers ---------------------------------------------------------

    def _draft_plan(self, task: str, messages: list[Message], budget: Budget) -> Plan | None:
        if not self.plan:
            return None
        planned = make_plan(self.provider, task, temperature=self.temperature)
        budget.add_usage(planned.usage)
        messages.append(Message.system("Investigation plan:\n" + format_plan(planned.value)))
        return planned.value

    def _invoke(self, call: ToolCall) -> ToolInvocation:
        try:
            tool = self.tools.get(call.name)
        except ToolNotFound as exc:
            return ToolInvocation(call=call, result=ToolResult(ok=False, error=str(exc)))
        return ToolInvocation(call=call, result=tool.run(call.arguments, context=self.context))

    def _result(
        self,
        output: str | None,
        messages: list[Message],
        steps: list[Step],
        budget: Budget,
        finished: bool,
        stop_reason: str,
        plan: Plan | None,
        critiques: list[Critique],
    ) -> RunResult:
        return RunResult(
            output=output,
            messages=messages,
            steps=steps,
            usage=budget.usage,
            finished=finished,
            stop_reason=stop_reason,
            plan=plan,
            critiques=critiques,
        )
