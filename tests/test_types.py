from __future__ import annotations

import pytest
from pydantic import ValidationError

from dantalion.types import Message, Role, ToolCall, Usage


def test_message_constructors_set_roles() -> None:
    assert Message.system("x").role is Role.SYSTEM
    assert Message.user("x").role is Role.USER
    assert Message.assistant("x").role is Role.ASSISTANT
    assert Message.tool("x", tool_call_id="c0", name="t").role is Role.TOOL


def test_tool_result_message_links_back_to_call() -> None:
    message = Message.tool("result", tool_call_id="c1", name="aggregate")
    assert message.tool_call_id == "c1"
    assert message.name == "aggregate"


def test_assistant_can_carry_tool_calls() -> None:
    call = ToolCall(id="c0", name="search", arguments={"q": "error"})
    message = Message.assistant(tool_calls=[call])
    assert message.content is None
    assert message.tool_calls == [call]


def test_usage_adds_componentwise() -> None:
    a = Usage(prompt_tokens=10, completion_tokens=5)
    b = Usage(prompt_tokens=1, completion_tokens=2)
    total = a + b
    assert total.prompt_tokens == 11
    assert total.completion_tokens == 7
    assert total.total_tokens == 18


def test_tool_call_is_frozen() -> None:
    call = ToolCall(id="c0", name="t", arguments={"a": 1})
    with pytest.raises(ValidationError):
        call.id = "c1"  # type: ignore[misc]
