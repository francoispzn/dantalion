from __future__ import annotations

from dantalion.agent import Agent, CancellationToken
from dantalion.providers.mock import MockProvider, text_response, tool_call_response
from dantalion.tools import ToolRegistry, tool
from dantalion.types import (
    CompletionRequest,
    CompletionResponse,
    FinishReason,
    Message,
    Role,
    ToolCall,
    Usage,
)


@tool
def noop() -> str:
    """Do nothing useful."""
    return "ok"


def _registry() -> ToolRegistry:
    return ToolRegistry([noop])


def test_planner_injects_a_plan() -> None:
    provider = MockProvider(
        [
            '{"steps": ["look at errors", "correlate", "conclude"]}',
            text_response("the cause is disk pressure"),
        ]
    )
    result = Agent(provider, _registry(), plan=True).run("why did latency spike?")

    assert result.plan is not None
    assert result.plan.steps == ["look at errors", "correlate", "conclude"]
    executor_turn = provider.requests[1]
    assert any("Investigation plan" in (m.content or "") for m in executor_turn.messages)


def test_review_sends_agent_back_when_insufficient() -> None:
    provider = MockProvider(
        [
            text_response("it was probably the network"),
            '{"sufficient": false, "reasoning": "no evidence", "guidance": "check the logs"}',
            text_response("it was disk pressure, confirmed in the logs"),
            '{"sufficient": true, "reasoning": "evidence supports it"}',
        ]
    )
    result = Agent(provider, _registry(), review=True, max_reviews=2).run("root cause?")

    assert result.output == "it was disk pressure, confirmed in the logs"
    assert [c.sufficient for c in result.critiques] == [False, True]


def test_review_accepts_a_supported_answer_immediately() -> None:
    provider = MockProvider(
        [
            text_response("disk pressure"),
            '{"sufficient": true, "reasoning": "well supported"}',
        ]
    )
    result = Agent(provider, _registry(), review=True).run("root cause?")

    assert result.output == "disk pressure"
    assert len(result.critiques) == 1


def test_review_stops_after_max_reviews() -> None:
    provider = MockProvider(
        [
            text_response("first guess"),
            '{"sufficient": false, "reasoning": "weak", "guidance": "dig deeper"}',
            text_response("second guess"),
        ]
    )
    result = Agent(provider, _registry(), review=True, max_reviews=1).run("root cause?")

    assert result.output == "second guess"
    assert len(result.critiques) == 1


def test_token_budget_stops_an_eager_agent() -> None:
    def script(request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(
            message=Message.assistant(tool_calls=[ToolCall(id="c", name="noop", arguments={})]),
            model="mock",
            usage=Usage(completion_tokens=50),
            finish_reason=FinishReason.TOOL_CALLS,
        )

    result = Agent(MockProvider(script=script), _registry(), max_steps=100, max_tokens=120).run(
        "go"
    )

    assert not result.finished
    assert result.stop_reason == "token_budget"


def test_pre_cancelled_run_stops_immediately() -> None:
    token = CancellationToken()
    token.cancel()
    result = Agent(MockProvider(["unused"]), _registry(), cancellation=token).run("go")

    assert not result.finished
    assert result.stop_reason == "cancelled"
    assert result.steps == []


def test_cancellation_between_steps() -> None:
    token = CancellationToken()
    calls = {"n": 0}

    def script(request: CompletionRequest) -> CompletionResponse:
        calls["n"] += 1
        if calls["n"] >= 2:
            token.cancel()
        return tool_call_response("noop", {}, call_id=f"c{calls['n']}")

    result = Agent(MockProvider(script=script), _registry(), cancellation=token, max_steps=99).run(
        "go"
    )

    assert result.stop_reason == "cancelled"
    assert len(result.steps) == 2


def test_tool_evidence_reaches_the_critic() -> None:
    provider = MockProvider(
        [
            tool_call_response("noop", {}, call_id="c1"),
            text_response("done"),
            '{"sufficient": true, "reasoning": "ok"}',
        ]
    )
    Agent(provider, _registry(), review=True).run("go")

    critic_turn = provider.requests[-1]
    user_message = next(m for m in critic_turn.messages if m.role is Role.USER)
    assert "noop" in (user_message.content or "")
