from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from dantalion.errors import ProviderError, ProviderTimeout
from dantalion.providers import build_provider
from dantalion.providers.openai_compat import (
    OpenAICompatibleProvider,
    _load_arguments,
    _parse_sse_line,
)
from dantalion.types import (
    CompletionRequest,
    FinishReason,
    Message,
    ResponseFormat,
    ToolSpec,
)

Handler = Callable[[httpx.Request], httpx.Response]


def _provider(handler: Handler, model: str = "llama3.1", **kwargs: Any) -> OpenAICompatibleProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test/v1")
    return OpenAICompatibleProvider(model, client=client, **kwargs)


def _chat(content: str = "hi", **extra: Any) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    message.update(extra)
    return {
        "model": "llama3.1",
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3},
    }


def test_complete_parses_text_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json=_chat("hello"))

    out = _provider(handler).complete(CompletionRequest(messages=[Message.user("hi")]))
    assert out.message.content == "hello"
    assert out.usage.prompt_tokens == 9
    assert out.usage.completion_tokens == 3
    assert out.finish_reason is FinishReason.STOP


def test_tool_call_arguments_are_json_decoded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = {
            "model": "llama3.1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "aggregate",
                                    "arguments": '{"col": "latency", "op": "p99"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        return httpx.Response(200, json=body)

    out = _provider(handler).complete(CompletionRequest(messages=[Message.user("x")]))
    assert out.finish_reason is FinishReason.TOOL_CALLS
    call = out.message.tool_calls[0]
    assert call.id == "call_abc"
    assert call.arguments == {"col": "latency", "op": "p99"}


def test_payload_encodes_schema_tools_and_options() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_chat("ok"))

    request = CompletionRequest(
        messages=[
            Message.assistant(content="prior"),
            Message.tool("result", tool_call_id="call_1", name="aggregate"),
        ],
        tools=[ToolSpec(name="aggregate", description="d")],
        response_format=ResponseFormat(name="report", json_schema={"type": "object"}),
        max_tokens=128,
        seed=11,
        stop=["END"],
    )
    _provider(handler).complete(request)

    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["name"] == "report"
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert captured["tool_choice"] == "auto"
    assert captured["max_tokens"] == 128
    assert captured["seed"] == 11
    assert captured["stop"] == ["END"]
    tool_message = captured["messages"][1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"


def test_usage_is_estimated_when_server_omits_it() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = {
            "model": "m",
            "choices": [{"message": {"role": "assistant", "content": "abcd efgh"}}],
        }
        return httpx.Response(200, json=body)

    out = _provider(handler, model="m").complete(
        CompletionRequest(messages=[Message.user("some prompt text")])
    )
    assert out.usage.prompt_tokens > 0
    assert out.usage.completion_tokens > 0


def test_no_choices_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "m", "choices": []})

    with pytest.raises(ProviderError):
        _provider(handler, model="m").complete(CompletionRequest(messages=[Message.user("x")]))


def test_capabilities_respect_overrides() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    caps = _provider(
        handler,
        model="unknown",
        supports_tools=True,
        supports_json_schema=False,
        context_window=4096,
    ).capabilities()
    assert caps.tool_calling is True
    assert caps.json_schema is False
    assert caps.context_window == 4096
    assert caps.grammar is False


def test_capabilities_fall_back_to_registry() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    caps = _provider(handler).capabilities()
    assert caps.tool_calling is True  # llama3.1 in registry
    assert caps.json_schema is True  # default for openai-compatible


def test_timeout_is_translated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow")

    with pytest.raises(ProviderTimeout):
        _provider(handler, model="m").complete(CompletionRequest(messages=[Message.user("x")]))


def test_stream_parses_sse_deltas() -> None:
    body = (
        'data: {"choices":[{"delta":{"content":"he"}}]}\n'
        'data: {"choices":[{"delta":{"content":"llo"},"finish_reason":"stop"}]}\n'
        "data: [DONE]\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body.encode())

    request = CompletionRequest(messages=[Message.user("x")])
    chunks = list(_provider(handler, model="m").stream(request))
    assert "".join(c.delta for c in chunks) == "hello"
    assert chunks[-1].finish_reason is FinishReason.STOP


def test_load_arguments_tolerates_bad_input() -> None:
    assert _load_arguments('{"a": 1}') == {"a": 1}
    assert _load_arguments({"a": 1}) == {"a": 1}
    assert _load_arguments("not json") == {}
    assert _load_arguments("[1, 2]") == {}
    assert _load_arguments(None) == {}


def test_parse_sse_line_skips_non_data() -> None:
    assert _parse_sse_line("") is None
    assert _parse_sse_line(": comment") is None
    assert _parse_sse_line("data: [DONE]") is None


def test_build_provider_aliases_map_to_openai() -> None:
    assert isinstance(build_provider("vllm", "m"), OpenAICompatibleProvider)
    assert isinstance(build_provider("lmstudio", "m"), OpenAICompatibleProvider)
    assert isinstance(build_provider("openai-compatible", "m"), OpenAICompatibleProvider)
