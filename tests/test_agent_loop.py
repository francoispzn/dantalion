from __future__ import annotations

from dantalion.agent import Agent
from dantalion.providers.mock import MockProvider, text_response, tool_call_response
from dantalion.tools import ToolRegistry, tool
from dantalion.types import (
    CompletionRequest,
    CompletionResponse,
    FinishReason,
    Message,
    Role,
    ToolCall,
)


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool(inject=("store",))
def lookup(store: dict[str, int], key: str) -> int:
    return store[key]


def _registry() -> ToolRegistry:
    return ToolRegistry([add, lookup])


def test_single_tool_round_then_final_answer() -> None:
    provider = MockProvider(
        [
            tool_call_response("add", {"a": 2, "b": 3}, call_id="c1"),
            text_response("the answer is 5"),
        ]
    )
    result = Agent(provider, _registry(), max_steps=5).run("add 2 and 3")

    assert result.finished
    assert result.output == "the answer is 5"
    assert len(result.steps) == 1
    invocation = result.steps[0].invocations[0]
    assert invocation.result.ok
    assert invocation.result.data == 5


def test_tool_result_is_fed_back_to_the_model() -> None:
    provider = MockProvider(
        [tool_call_response("add", {"a": 2, "b": 3}, call_id="c1"), text_response("done")]
    )
    Agent(provider, _registry()).run("x")

    second_turn = provider.requests[1]
    tool_messages = [m for m in second_turn.messages if m.role is Role.TOOL]
    assert tool_messages[0].content == "5"
    assert tool_messages[0].tool_call_id == "c1"


def test_unknown_tool_is_reported_and_loop_recovers() -> None:
    provider = MockProvider(
        [tool_call_response("does_not_exist", {}, call_id="c1"), text_response("recovered")]
    )
    result = Agent(provider, _registry()).run("x")

    assert result.finished
    assert result.output == "recovered"
    assert not result.steps[0].invocations[0].result.ok


def test_multiple_tool_calls_in_one_step() -> None:
    parallel = CompletionResponse(
        message=Message.assistant(
            tool_calls=[
                ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2}),
                ToolCall(id="c2", name="add", arguments={"a": 3, "b": 4}),
            ]
        ),
        model="mock",
        finish_reason=FinishReason.TOOL_CALLS,
    )
    provider = MockProvider([parallel, text_response("done")])
    result = Agent(provider, _registry()).run("x")

    assert [inv.result.data for inv in result.steps[0].invocations] == [3, 7]
    assert len(result.tool_calls) == 2


def test_context_is_injected_into_tools() -> None:
    provider = MockProvider(
        [tool_call_response("lookup", {"key": "x"}, call_id="c1"), text_response("ok")]
    )
    result = Agent(provider, _registry(), context={"store": {"x": 42}}).run("t")
    assert result.steps[0].invocations[0].result.data == 42


def test_max_steps_stops_an_unfinished_run() -> None:
    def script(request: CompletionRequest) -> CompletionResponse:
        return tool_call_response("add", {"a": 1, "b": 1}, call_id="c1")

    result = Agent(MockProvider(script=script), _registry(), max_steps=3).run("loop")

    assert not result.finished
    assert result.stop_reason == "max_steps"
    assert len(result.steps) == 3
    assert result.output is None


def test_usage_accumulates_across_steps() -> None:
    provider = MockProvider(
        [tool_call_response("add", {"a": 1, "b": 1}, call_id="c1"), text_response("done")]
    )
    result = Agent(provider, _registry()).run("x")
    assert result.usage.total_tokens >= 1
