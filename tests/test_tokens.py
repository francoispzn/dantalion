from __future__ import annotations

from dantalion.providers.tokens import estimate_message_tokens, estimate_tokens
from dantalion.types import Message, ToolCall


def test_empty_text_is_zero_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


def test_short_text_is_at_least_one_token() -> None:
    assert estimate_tokens("a") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


def test_estimate_grows_with_length() -> None:
    short = estimate_tokens("hello")
    long = estimate_tokens("hello " * 100)
    assert long > short


def test_message_tokens_include_overhead_and_tool_calls() -> None:
    messages = [
        Message.system("you are a careful assistant"),
        Message.user("what happened at 09:00?"),
        Message.assistant(
            tool_calls=[ToolCall(id="c0", name="slice_timewindow", arguments={"start": "09:00"})]
        ),
    ]
    total = estimate_message_tokens(messages)
    # three messages worth of overhead at minimum, plus content.
    assert total > 12
