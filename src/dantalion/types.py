"""Core value types shared across the whole agent.

These describe the *conversation* with a model in provider-neutral terms. Every
provider adapter is responsible for translating to and from these types, which
keeps the rest of the codebase from ever having to know whether it is talking to
Ollama, an OpenAI-compatible server, or an in-process llama.cpp model.

They are deliberately plain pydantic models: cheap to build, trivially
serialisable (which the tracer relies on), and validated at the edges.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    """Who an utterance belongs to."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A request, made by the model, to invoke one tool with some arguments."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """One turn in a conversation.

    A single model can speak text, ask for tools, or both at once, so
    ``content`` and ``tool_calls`` are not mutually exclusive. Tool *results*
    come back as messages with ``role=TOOL`` and a ``tool_call_id`` linking them
    to the call that produced them.
    """

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(
        cls, content: str | None = None, *, tool_calls: list[ToolCall] | None = None
    ) -> Message:
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, content: str, *, tool_call_id: str, name: str) -> Message:
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id, name=name)


class Usage(BaseModel):
    """Token accounting for one or more completions.

    Local servers are inconsistent about reporting usage; when they do not, the
    provider estimates it (see ``providers.tokens``) so budgets still mean
    something. ``Usage`` values add, which makes per-run totals a fold.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class ToolSpec(BaseModel):
    """The provider-facing description of a tool: a name, a sentence, a schema.

    ``parameters`` is a JSON Schema object. The tool framework generates these
    from typed Python signatures; providers translate them into whatever their
    function-calling API expects.
    """

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ResponseFormat(BaseModel):
    """A request for structured output.

    Providers honour whichever of these they can: ``json_schema`` for servers
    with schema-constrained decoding, ``grammar`` (GBNF) for llama.cpp. When a
    provider can do neither, the structured layer falls back to prompting and
    repair — see ``structured``.
    """

    name: str = "response"
    json_schema: dict[str, Any] | None = None
    grammar: str | None = None


class FinishReason(StrEnum):
    """Why a completion stopped."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class CompletionRequest(BaseModel):
    """Everything a provider needs to produce one assistant turn."""

    messages: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    response_format: ResponseFormat | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    stop: list[str] = Field(default_factory=list)
    seed: int | None = None


class CompletionResponse(BaseModel):
    """One assistant turn, plus the bookkeeping the agent cares about."""

    message: Message
    model: str
    usage: Usage = Field(default_factory=Usage)
    finish_reason: FinishReason = FinishReason.STOP
    raw: dict[str, Any] | None = None


class CompletionChunk(BaseModel):
    """A single streamed fragment of an assistant turn.

    Streaming is text-only on purpose: tool calls and structured output need the
    whole payload before they mean anything, so those paths use ``complete``.
    Streaming exists for the human-readable narration the CLI prints.
    """

    delta: str = ""
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
