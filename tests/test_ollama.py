from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from dantalion.errors import ProviderTimeout
from dantalion.providers.ollama import (
    OllamaProvider,
    _context_length,
    _parse_stream_line,
)
from dantalion.types import (
    CompletionRequest,
    FinishReason,
    Message,
    ResponseFormat,
    ToolSpec,
)

Handler = Callable[[httpx.Request], httpx.Response]


def _provider(handler: Handler, model: str = "llama3.1") -> OllamaProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    return OllamaProvider(model, client=client)


def test_complete_parses_text_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(
            200,
            json={
                "model": "llama3.1",
                "message": {"role": "assistant", "content": "hi there"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 12,
                "eval_count": 5,
            },
        )

    out = _provider(handler).complete(CompletionRequest(messages=[Message.user("hi")]))
    assert out.message.content == "hi there"
    assert out.usage.prompt_tokens == 12
    assert out.usage.completion_tokens == 5
    assert out.finish_reason is FinishReason.STOP


def test_complete_parses_tool_calls_and_synthesises_ids() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "llama3.1",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "aggregate", "arguments": {"col": "latency"}}}
                    ],
                },
                "done": True,
                "done_reason": "stop",
            },
        )

    out = _provider(handler).complete(CompletionRequest(messages=[Message.user("x")]))
    assert out.finish_reason is FinishReason.TOOL_CALLS
    assert out.message.tool_calls[0].name == "aggregate"
    assert out.message.tool_calls[0].id == "call_0"
    assert out.message.tool_calls[0].arguments == {"col": "latency"}


def test_request_payload_carries_schema_tools_and_options() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"model": "m", "message": {"role": "assistant", "content": "ok"}, "done": True},
        )

    request = CompletionRequest(
        messages=[Message.user("x")],
        tools=[ToolSpec(name="t", description="d")],
        response_format=ResponseFormat(json_schema={"type": "object"}),
        max_tokens=64,
        seed=7,
        stop=["END"],
    )
    _provider(handler, model="m").complete(request)
    assert captured["format"] == {"type": "object"}
    assert captured["tools"][0]["function"]["name"] == "t"
    assert captured["options"]["num_predict"] == 64
    assert captured["options"]["seed"] == 7
    assert captured["options"]["stop"] == ["END"]


def test_capabilities_uses_server_probe() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(
                200,
                json={
                    "capabilities": ["completion", "tools"],
                    "model_info": {"llama.context_length": 131072},
                },
            )
        return httpx.Response(404)

    caps = _provider(handler, model="mystery").capabilities()
    assert caps.tool_calling is True
    assert caps.json_schema is True
    assert caps.context_window == 131072


def test_capabilities_fall_back_to_registry_when_probe_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    caps = _provider(handler).capabilities()
    assert caps.tool_calling is True  # registry knows llama3.1 can call tools
    assert caps.json_schema is True


def test_capabilities_are_cached() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"capabilities": ["completion"]})

    provider = _provider(handler)
    provider.capabilities()
    provider.capabilities()
    assert calls["n"] == 1


def test_timeout_is_translated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow")

    with pytest.raises(ProviderTimeout):
        _provider(handler, model="m").complete(CompletionRequest(messages=[Message.user("x")]))


def test_stream_reassembles_deltas() -> None:
    body = (
        '{"message":{"content":"hel"},"done":false}\n'
        '{"message":{"content":"lo"},"done":true,"done_reason":"stop","eval_count":2}\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body.encode())

    request = CompletionRequest(messages=[Message.user("x")])
    chunks = list(_provider(handler, model="m").stream(request))
    assert "".join(c.delta for c in chunks) == "hello"
    assert chunks[-1].finish_reason is FinishReason.STOP


def test_parse_stream_line_ignores_garbage() -> None:
    assert _parse_stream_line("not json") is None


def test_context_length_extraction() -> None:
    assert _context_length({"llama.context_length": 4096}) == 4096
    assert _context_length({"unrelated": 1}) is None
    assert _context_length(None) is None
