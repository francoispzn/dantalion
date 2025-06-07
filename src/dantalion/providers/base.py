"""The provider contract.

A :class:`Provider` is the only thing the agent needs in order to talk to a
model. Keeping the surface this small is what makes the project genuinely model
agnostic: swapping Ollama for a vLLM server, or for an in-process GGUF, is a one
line change and nothing downstream notices.

:class:`Capabilities` is the other half of the contract. Local models vary wildly
in what they support, so rather than discovering the hard way that a model cannot
call tools, the agent asks up front and adapts.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from dantalion.types import (
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
)


class Capabilities(BaseModel):
    """What a particular model, behind a particular server, can actually do.

    These drive real decisions: the structured-output layer picks a strategy
    from ``json_schema``/``grammar``, the agent only offers ``tools`` when
    ``tool_calling`` is true, and the context budgeter sizes compaction against
    ``context_window``.
    """

    tool_calling: bool = False
    json_schema: bool = False
    grammar: bool = False
    context_window: int = 8192
    max_output_tokens: int = 2048
    embeddings: bool = False


@runtime_checkable
class Provider(Protocol):
    """A synchronous handle to one model.

    Synchronous on purpose: an investigation is an inherently sequential
    plan/act/observe loop, and a blocking interface keeps traces deterministic
    and tests free of an event loop. Concurrency, where it matters, lives above
    this layer.
    """

    name: str
    model: str

    def capabilities(self) -> Capabilities:
        """Return what this model/server can do, ideally probed and cached."""
        ...

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Produce a single assistant turn for ``request``."""
        ...

    def stream(self, request: CompletionRequest) -> Iterator[CompletionChunk]:
        """Yield the assistant turn as text fragments, for human-facing output."""
        ...
