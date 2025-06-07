"""A scripted in-memory provider.

This is how the whole stack is tested without a model in the loop. You hand it a
list of turns to return — plain strings, tool calls, or full responses — and it
plays them back in order while recording every request it received, so tests can
assert on what the agent actually asked for.

It is also the seed of the record/replay machinery: a cassette is, in essence, a
serialised script for one of these.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from dantalion.errors import ProviderError
from dantalion.providers.base import Capabilities
from dantalion.providers.base import Provider as ProviderProto
from dantalion.types import (
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    FinishReason,
    Message,
    ToolCall,
    Usage,
)

Script = Callable[[CompletionRequest], CompletionResponse]
Scripted = str | CompletionResponse


def text_response(content: str, *, model: str = "mock") -> CompletionResponse:
    """Wrap a plain string as a finished assistant turn."""
    return CompletionResponse(
        message=Message.assistant(content),
        model=model,
        usage=Usage(completion_tokens=max(1, len(content) // 4)),
        finish_reason=FinishReason.STOP,
    )


def tool_call_response(
    name: str, arguments: dict[str, Any], *, call_id: str = "call_0", model: str = "mock"
) -> CompletionResponse:
    """Build an assistant turn that requests a single tool call."""
    return CompletionResponse(
        message=Message.assistant(
            tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)]
        ),
        model=model,
        finish_reason=FinishReason.TOOL_CALLS,
    )


class MockProvider:
    """A provider that returns a predetermined sequence of turns."""

    name = "mock"

    def __init__(
        self,
        responses: list[Scripted] | None = None,
        *,
        model: str = "mock",
        capabilities: Capabilities | None = None,
        script: Script | None = None,
    ) -> None:
        self.model = model
        self._responses: list[CompletionResponse] = [
            text_response(r, model=model) if isinstance(r, str) else r for r in (responses or [])
        ]
        self._script = script
        self._capabilities = capabilities or Capabilities(
            tool_calling=True,
            json_schema=True,
            grammar=True,
            context_window=8192,
            max_output_tokens=2048,
        )
        self.requests: list[CompletionRequest] = []

    def capabilities(self) -> Capabilities:
        return self._capabilities

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if self._script is not None:
            return self._script(request)
        if not self._responses:
            raise ProviderError("mock provider script is exhausted")
        return self._responses.pop(0)

    def stream(self, request: CompletionRequest) -> Iterator[CompletionChunk]:
        response = self.complete(request)
        yield CompletionChunk(
            delta=response.message.content or "",
            finish_reason=response.finish_reason,
            usage=response.usage,
        )


_: type[ProviderProto] = MockProvider
