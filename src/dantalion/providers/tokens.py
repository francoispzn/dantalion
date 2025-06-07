"""Token estimation.

Local servers are wildly inconsistent about reporting token usage: some give
exact counts, some give none, some only count the prompt. Budgets, context
compaction, and cost reporting all need *a* number regardless, so when a server
stays quiet we fall back to a cheap heuristic.

The heuristic is intentionally model-agnostic and slightly conservative (it
tends to over-count a touch), because for budgeting an over-estimate fails safe.
It is not a replacement for a real tokenizer and never pretends to be.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from dantalion.types import Message

# Roughly four characters per token across common BPE vocabularies. Good enough
# for guard rails; wrong enough that we never quote it as exact.
_CHARS_PER_TOKEN = 4
_PER_MESSAGE_OVERHEAD = 4


def estimate_tokens(text: str | None) -> int:
    """Estimate the token count of a string."""
    if not text:
        return 0
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def estimate_message_tokens(messages: Iterable[Message]) -> int:
    """Estimate the token cost of a list of messages, including tool calls."""
    total = 0
    for message in messages:
        total += _PER_MESSAGE_OVERHEAD
        total += estimate_tokens(message.content)
        for call in message.tool_calls:
            total += estimate_tokens(call.name)
            total += estimate_tokens(json.dumps(call.arguments, separators=(",", ":")))
    return total
