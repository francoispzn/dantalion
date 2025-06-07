"""Adapter for a local `Ollama <https://ollama.com>`_ server.

Ollama is the most common way people run models locally, so it is the reference
adapter. It speaks its own JSON API (not the OpenAI shape), supports server-side
JSON-schema constrained decoding, and reports per-model capabilities through
``/api/show`` — which we probe so the agent knows whether a given model can
actually call tools.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from dantalion.errors import ProviderError, ProviderTimeout
from dantalion.providers.base import Capabilities, Provider
from dantalion.providers.registry import lookup_capabilities
from dantalion.providers.tokens import estimate_tokens
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

DEFAULT_BASE_URL = "http://localhost:11434"

_FINISH_REASONS = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
}


class OllamaProvider:
    """Talk to a model served by a local Ollama daemon."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(base_url=self._base_url, timeout=timeout)
        self._capabilities: Capabilities | None = None

    # -- capabilities ----------------------------------------------------

    def capabilities(self) -> Capabilities:
        """Registry guess, refined by what the server reports for this model."""
        if self._capabilities is not None:
            return self._capabilities

        caps = lookup_capabilities(self.model)
        # Structured decoding is a server feature in Ollama, independent of the
        # model, so it is available even when the model cannot call tools.
        caps.json_schema = True

        info = self._show()
        if info is not None:
            reported = info.get("capabilities")
            if isinstance(reported, list):
                caps.tool_calling = "tools" in reported
                caps.embeddings = "embedding" in reported
            context = _context_length(info.get("model_info"))
            if context is not None:
                caps.context_window = context

        self._capabilities = caps
        return caps

    def _show(self) -> dict[str, Any] | None:
        try:
            response = self._client.post("/api/show", json={"model": self.model})
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            return data
        except (httpx.HTTPError, ValueError):
            # A missing or old server just means we fall back to the registry;
            # any real connectivity problem will surface on the first complete().
            return None

    # -- completion ------------------------------------------------------

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        payload = self._build_payload(request, stream=False)
        data = self._post_json("/api/chat", payload)
        return self._parse_response(data)

    def stream(self, request: CompletionRequest) -> Iterator[CompletionChunk]:
        payload = self._build_payload(request, stream=True)
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = _parse_stream_line(line)
                    if chunk is not None:
                        yield chunk
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"ollama stream timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"ollama stream failed: {exc}") from exc

    # -- internals -------------------------------------------------------

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post(path, json=payload)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(f"ollama request timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"ollama returned {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"ollama request failed: {exc}") from exc

    def _build_payload(self, request: CompletionRequest, *, stream: bool) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.seed is not None:
            options["seed"] = request.seed
        if request.stop:
            options["stop"] = request.stop

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_ollama_message(m) for m in request.messages],
            "stream": stream,
            "options": options,
        }
        if request.tools:
            payload["tools"] = [_to_ollama_tool(t) for t in request.tools]
        if request.response_format and request.response_format.json_schema is not None:
            payload["format"] = request.response_format.json_schema
        return payload

    def _parse_response(self, data: dict[str, Any]) -> CompletionResponse:
        message = _parse_message(data.get("message", {}))
        usage = _parse_usage(data, message=message)
        finish = _finish_reason(data.get("done_reason"), message)
        return CompletionResponse(
            message=message,
            model=data.get("model", self.model),
            usage=usage,
            finish_reason=finish,
            raw=data,
        )


# -- translation helpers (module level: pure and easy to unit test) ------


def _to_ollama_message(message: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": message.role.value, "content": message.content or ""}
    if message.tool_calls:
        out["tool_calls"] = [
            {"function": {"name": c.name, "arguments": c.arguments}} for c in message.tool_calls
        ]
    if message.role is Role.TOOL and message.name:
        out["tool_name"] = message.name
    return out


def _to_ollama_tool(tool: ToolSpec) -> dict[str, Any]:
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
                arguments=function.get("arguments") or {},
            )
        )
    return Message(
        role=Role.ASSISTANT,
        content=raw.get("content") or None,
        tool_calls=tool_calls,
    )


def _parse_usage(data: dict[str, Any], *, message: Message) -> Usage:
    prompt = data.get("prompt_eval_count")
    completion = data.get("eval_count")
    return Usage(
        prompt_tokens=int(prompt) if prompt is not None else 0,
        completion_tokens=(
            int(completion) if completion is not None else estimate_tokens(message.content)
        ),
    )


def _finish_reason(done_reason: str | None, message: Message) -> FinishReason:
    if message.tool_calls:
        return FinishReason.TOOL_CALLS
    if done_reason is None:
        return FinishReason.STOP
    return _FINISH_REASONS.get(done_reason, FinishReason.STOP)


def _parse_stream_line(line: str) -> CompletionChunk | None:
    try:
        data = json.loads(line)
    except ValueError:
        return None
    delta = (data.get("message") or {}).get("content", "")
    done = data.get("done", False)
    usage = None
    finish = None
    if done:
        finish = _FINISH_REASONS.get(data.get("done_reason", "stop"), FinishReason.STOP)
        usage = Usage(
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
        )
    if not delta and finish is None:
        return None
    return CompletionChunk(delta=delta, finish_reason=finish, usage=usage)


def _context_length(model_info: Any) -> int | None:
    if not isinstance(model_info, dict):
        return None
    for key, value in model_info.items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    return None


# A small assurance that the adapter satisfies the structural contract.
_: type[Provider] = OllamaProvider
