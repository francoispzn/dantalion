from __future__ import annotations

import pytest

from dantalion.errors import ProviderError
from dantalion.providers.mock import MockProvider, text_response, tool_call_response
from dantalion.types import CompletionRequest, CompletionResponse, FinishReason, Message


def _request(text: str = "hi") -> CompletionRequest:
    return CompletionRequest(messages=[Message.user(text)])


def test_plays_back_strings_in_order() -> None:
    provider = MockProvider(["first", "second"])
    assert provider.complete(_request()).message.content == "first"
    assert provider.complete(_request()).message.content == "second"


def test_records_every_request() -> None:
    provider = MockProvider(["ok"])
    request = _request("remember me")
    provider.complete(request)
    assert provider.requests == [request]


def test_exhausted_script_raises() -> None:
    provider = MockProvider([])
    with pytest.raises(ProviderError):
        provider.complete(_request())


def test_script_callable_sees_request() -> None:
    def script(request: CompletionRequest) -> CompletionResponse:
        return text_response("echo:" + (request.messages[-1].content or ""))

    provider = MockProvider(script=script)
    assert provider.complete(_request("ping")).message.content == "echo:ping"


def test_tool_call_response_helper() -> None:
    response = tool_call_response("aggregate", {"col": "latency"})
    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.message.tool_calls[0].name == "aggregate"
    assert response.message.tool_calls[0].arguments == {"col": "latency"}


def test_stream_reassembles_to_full_text() -> None:
    provider = MockProvider(["hello world"])
    text = "".join(chunk.delta for chunk in provider.stream(_request()))
    assert text == "hello world"
