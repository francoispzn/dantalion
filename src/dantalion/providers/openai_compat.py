"""Adapter for any server that speaks the OpenAI chat-completions dialect.

That dialect has become the lingua franca of local inference: vLLM, LM Studio,
llama.cpp's ``server``, text-generation-webui and others all expose a ``/v1``
endpoint shaped like OpenAI's. One adapter therefore covers a large slice of the
local ecosystem.

Capability detection is harder here than with Ollama, because the wire protocol
says nothing about what the model can do. We lean on the model-name registry and
let the caller override the few things that genuinely vary between servers
(whether tools and JSON-schema decoding are supported).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from dantalion.errors import ProviderError, ProviderTimeout
from dantalion.providers.base import Capabilities, Provider
from dantalion.providers.registry import lookup_capabilities
from dantalion.providers.tokens import estimate_message_tokens, estimate_tokens
from dantalion.types import (
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    FinishReason,
    Message,
    Role,
    ToolCall,
    ToolSpec,
    Usage,
)

DEFAULT_BASE_URL = "http://localhost:8000/v1"

_FINISH_REASONS = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}


class OpenAICompatibleProvider:
    """Talk to an OpenAI-compatible ``/v1/chat/completions`` server."""

    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
        supports_tools: bool | None = None,
        supports_json_schema: bool = True,
        context_window: int | None = None,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._supports_tools = supports_tools
        self._supports_json_schema = supports_json_schema
        self._context_window = context_window
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(
            base_url=self._base_url, timeout=timeout, headers=headers
        )
        self._capabilities: Capabilities | None = None

    def capabilities(self) -> Capabilities:
        if self._capabilities is not None:
            return self._capabilities
        caps = lookup_capabilities(self.model)
        if self._supports_tools is not None:
            caps.tool_calling = self._supports_tools
        caps.json_schema = self._supports_json_schema
        caps.grammar = False  # in-process llama.cpp handles GBNF; servers vary too much.
        if self._context_window is not None:
            caps.context_window = self._context_window
        self._capabilities = caps
        return caps

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        payload = self._build_payload(request, stream=False)
        data = self._post_json("/chat/completions", payload)
        return self._parse_response(data, request)

    def stream(self, request: CompletionRequest) -> Iterator[CompletionChunk]:
        payload = self._build_payload(request, stream=True)
        try:
            with self._client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    chunk = _parse_sse_line(line)
                    if chunk is not None:
                        yield chunk
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"openai-compatible stream timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"openai-compatible stream failed: {exc}") from exc

    # -- internals -------------------------------------------------------

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(path, json=payload)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"openai-compatible request timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"openai-compatible server returned {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"openai-compatible request failed: {exc}") from exc

    def _build_payload(self, request: CompletionRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_openai_message(m) for m in request.messages],
            "temperature": request.temperature,
            "stream": stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.stop:
            payload["stop"] = request.stop
        if request.tools:
            payload["tools"] = [_to_openai_tool(t) for t in request.tools]
            payload["tool_choice"] = "auto"
        if request.response_format and request.response_format.json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.response_format.name,
                    "schema": request.response_format.json_schema,
                    "strict": True,
                },
            }
        return payload

    def _parse_response(
        self, data: dict[str, Any], request: CompletionRequest
    ) -> CompletionResponse:
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("openai-compatible response contained no choices")
        choice = choices[0]
        message = _parse_message(choice.get("message", {}))
        finish = _finish_reason(choice.get("finish_reason"), message)
        usage = _parse_usage(data.get("usage"), request, message)
        return CompletionResponse(
            message=message,
            model=data.get("model", self.model),
            usage=usage,
            finish_reason=finish,
            raw=data,
        )


# -- pure translation helpers --------------------------------------------


def _to_openai_message(message: Message) -> dict[str, Any]:
    if message.role is Role.TOOL:
        return {
            "role": "tool",
            "content": message.content or "",
            "tool_call_id": message.tool_call_id or "",
        }
    out: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, separators=(",", ":")),
                },
            }
            for call in message.tool_calls
        ]
    return out


def _to_openai_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _parse_message(raw: dict[str, Any]) -> Message:
    tool_calls: list[ToolCall] = []
    for index, call in enumerate(raw.get("tool_calls") or []):
        function = call.get("function", {})
        tool_calls.append(
            ToolCall(
                id=call.get("id") or f"call_{index}",
                name=function.get("name", ""),
                arguments=_load_arguments(function.get("arguments")),
            )
        )
    return Message(
        role=Role.ASSISTANT,
        content=raw.get("content") or None,
        tool_calls=tool_calls,
    )


def _load_arguments(arguments: Any) -> dict[str, Any]:
    """Tool-call arguments arrive as a JSON string; tolerate dicts and junk."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _parse_usage(raw: dict[str, Any] | None, request: CompletionRequest, message: Message) -> Usage:
    if raw:
        return Usage(
            prompt_tokens=int(raw.get("prompt_tokens") or 0),
            completion_tokens=int(raw.get("completion_tokens") or 0),
        )
    # Some servers omit usage entirely; estimate so budgets still hold.
    return Usage(
        prompt_tokens=estimate_message_tokens(request.messages),
        completion_tokens=estimate_tokens(message.content),
    )


def _finish_reason(reason: str | None, message: Message) -> FinishReason:
    if message.tool_calls:
        return FinishReason.TOOL_CALLS
    if reason is None:
        return FinishReason.STOP
    return _FINISH_REASONS.get(reason, FinishReason.STOP)


def _parse_sse_line(line: str) -> CompletionChunk | None:
    if not line or not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if not data or data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except ValueError:
        return None
    choices = payload.get("choices") or []
    if not choices:
        return None
    choice = choices[0]
    delta = (choice.get("delta") or {}).get("content") or ""
    reason = choice.get("finish_reason")
    finish = _FINISH_REASONS.get(reason) if reason else None
    if not delta and finish is None:
        return None
    return CompletionChunk(delta=delta, finish_reason=finish)


_: type[Provider] = OpenAICompatibleProvider
