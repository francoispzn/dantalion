from __future__ import annotations

from dantalion.agent import Agent
from dantalion.memory import (
    ContextCompactor,
    fits,
    model_summariser,
    overflow_tokens,
    structural_digest,
)
from dantalion.providers.mock import MockProvider, text_response, tool_call_response
from dantalion.tools import ToolRegistry, tool
from dantalion.types import Message, Role


@tool
def big() -> str:
    """Return a large blob."""
    return "x" * 400


def _transcript() -> list[Message]:
    return [
        Message.system("system framing"),
        Message.user("the task"),
        Message.assistant("thinking"),
        Message.tool("aggregate result: p99=1200ms", tool_call_id="c1", name="aggregate"),
        Message.assistant("more thinking"),
        Message.tool("found a spike", tool_call_id="c2", name="detect_spikes"),
        Message.user("keep going"),
        Message.assistant("almost there"),
    ]


def test_window_fits_and_overflow() -> None:
    messages = [Message.user("hello")]
    assert fits(messages, context_window=8192) is True
    assert overflow_tokens(messages, context_window=8192) == 0
    assert fits(messages, context_window=5, reserve_output=0) is False
    assert overflow_tokens(messages, context_window=1, reserve_output=0) > 0


def test_structural_digest_lists_tools_and_results() -> None:
    digest = structural_digest(_transcript())
    assert "aggregate" in digest
    assert "detect_spikes" in digest
    assert "p99=1200ms" in digest


def test_compactor_is_a_noop_under_budget() -> None:
    messages = _transcript()
    compactor = ContextCompactor(max_tokens=100_000)
    assert compactor.compact(messages) == messages


def test_compactor_preserves_head_and_tail() -> None:
    messages = _transcript()
    compactor = ContextCompactor(max_tokens=10, keep_recent=2, reserve_output=0)
    compacted = compactor.compact(messages)

    assert compacted[0] == messages[0]  # system framing kept
    assert compacted[1] == messages[1]  # task kept
    assert compacted[2].role is Role.SYSTEM
    assert compacted[2].content is not None and compacted[2].content.startswith("[Condensed")
    assert compacted[-2:] == messages[-2:]  # most recent turns kept verbatim


def test_model_summariser_calls_provider() -> None:
    provider = MockProvider(["earlier: found a disk spike"])
    summarise = model_summariser(provider)
    summary = summarise(_transcript())
    assert summary == "earlier: found a disk spike"
    assert "aggregate" in (provider.requests[0].messages[-1].content or "")


def test_compaction_triggers_inside_the_loop() -> None:
    provider = MockProvider(
        [
            tool_call_response("big", {}, call_id="c1"),
            tool_call_response("big", {}, call_id="c2"),
            text_response("done"),
        ]
    )
    compactor = ContextCompactor(max_tokens=50, keep_recent=2)
    result = Agent(provider, ToolRegistry([big]), compactor=compactor, max_steps=10).run("go")

    assert result.output == "done"
    assert any(
        (m.content or "").startswith("[Condensed") for m in result.messages if m.role is Role.SYSTEM
    )
