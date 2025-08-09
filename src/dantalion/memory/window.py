"""Reasoning about how much conversation fits in a model's context window.

Local models span a huge range of context sizes, and the agent has no business
assuming the generous end of it. These helpers estimate the token footprint of a
conversation and answer the only two questions the loop actually asks: does this
still fit, and if not, by how much is it over?

A reserve is always held back for the model's own reply — running the prompt
right up to the window edge leaves no room to answer.
"""

from __future__ import annotations

from collections.abc import Sequence

from dantalion.providers.tokens import estimate_message_tokens
from dantalion.types import Message

DEFAULT_OUTPUT_RESERVE = 512


def fits(
    messages: Sequence[Message],
    *,
    context_window: int,
    reserve_output: int = DEFAULT_OUTPUT_RESERVE,
) -> bool:
    """Whether ``messages`` plus a reply reserve fit inside the window."""
    return estimate_message_tokens(messages) + reserve_output <= context_window


def overflow_tokens(
    messages: Sequence[Message],
    *,
    context_window: int,
    reserve_output: int = DEFAULT_OUTPUT_RESERVE,
) -> int:
    """How many tokens over the window the conversation is (0 if it fits)."""
    projected = estimate_message_tokens(messages) + reserve_output
    return max(0, projected - context_window)
