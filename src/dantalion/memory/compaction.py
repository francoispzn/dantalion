"""Compacting a conversation that has outgrown its window.

When an investigation runs long, the transcript eventually threatens the context
window. Rather than truncate blindly and lose the thread, the compactor keeps
what matters — the system framing, the original task, and the most recent turns —
and condenses the middle into a single digest. The digest can be produced by the
model itself (``model_summariser``) or, for offline and deterministic runs, by a
structural summary that lists the tools called and their (truncated) results.

The invariant the loop relies on: compaction never drops the task or the latest
evidence, so the agent always knows what it was asked and what it just learned.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from dantalion.providers.base import Provider
from dantalion.providers.tokens import estimate_message_tokens
from dantalion.types import CompletionRequest, Message, Role

Summariser = Callable[[Sequence[Message]], str]


@dataclass
class ContextCompactor:
    """Shrinks a conversation back under a token budget when it overflows."""

    max_tokens: int
    keep_recent: int = 6
    reserve_output: int = 512
    summariser: Summariser | None = None

    def should_compact(self, messages: Sequence[Message]) -> bool:
        return estimate_message_tokens(messages) + self.reserve_output > self.max_tokens

    def compact(self, messages: Sequence[Message]) -> list[Message]:
        if not self.should_compact(messages):
            return list(messages)

        head_end = _head_end(messages)
        tail_start = len(messages) - self.keep_recent
        if tail_start <= head_end:  # nothing in the middle to condense
            return list(messages)

        head = list(messages[:head_end])
        middle = messages[head_end:tail_start]
        tail = list(messages[tail_start:])

        digest = self.summariser(middle) if self.summariser else structural_digest(middle)
        summary = Message.system(f"[Condensed {len(middle)} earlier messages]\n{digest}")
        return [*head, summary, *tail]


def _head_end(messages: Sequence[Message]) -> int:
    """Index just past the leading system messages and the first user task."""
    index = 0
    while index < len(messages) and messages[index].role is Role.SYSTEM:
        index += 1
    if index < len(messages) and messages[index].role is Role.USER:
        index += 1
    return index


def structural_digest(messages: Sequence[Message]) -> str:
    """A deterministic, model-free summary of a slice of conversation."""
    lines: list[str] = []
    for message in messages:
        if message.role is Role.ASSISTANT and message.tool_calls:
            for call in message.tool_calls:
                lines.append(f"called {call.name}({_short_args(call.arguments)})")
        elif message.role is Role.TOOL:
            lines.append(f"{message.name} -> {_truncate(message.content, 160)}")
        elif message.content:
            lines.append(f"{message.role.value}: {_truncate(message.content, 160)}")
    return "\n".join(lines)


def model_summariser(provider: Provider, *, max_tokens: int = 256) -> Summariser:
    """A summariser that asks the model to condense a transcript slice."""

    def summarise(messages: Sequence[Message]) -> str:
        conversation = [
            Message.system(
                "Condense the following investigation transcript into a short factual "
                "digest. Preserve findings, numbers, and any dead ends. No preamble."
            ),
            Message.user(structural_digest(messages)),
        ]
        response = provider.complete(
            CompletionRequest(messages=conversation, temperature=0.0, max_tokens=max_tokens)
        )
        return response.message.content or ""

    return summarise


def _short_args(arguments: dict[str, object]) -> str:
    return ", ".join(f"{key}={value}" for key, value in arguments.items())


def _truncate(text: str | None, limit: int) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[: limit - 1] + "…"
