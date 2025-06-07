"""A small static knowledge base of model capabilities.

Probing a live server is the source of truth, but it is not always available
(the server may be down, old, or simply not report the field we need). This
registry gives every adapter a sensible starting point keyed off the model name,
which the adapter then refines with whatever the server actually tells it.

The patterns are deliberately coarse. The goal is "don't offer tools to a model
that obviously can't use them", not a perfect census of the open-weights world.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dantalion.providers.base import Capabilities


@dataclass(frozen=True)
class ModelProfile:
    """A capability guess attached to a model-name pattern."""

    pattern: re.Pattern[str]
    context_window: int
    tool_calling: bool
    embeddings: bool = False


# Ordered most-specific first; the first match wins.
_PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(re.compile(r"llama-?3\.[13]"), 131072, tool_calling=True),
    ModelProfile(re.compile(r"llama-?3"), 8192, tool_calling=True),
    ModelProfile(re.compile(r"qwen-?2\.5"), 32768, tool_calling=True),
    ModelProfile(re.compile(r"qwen"), 32768, tool_calling=True),
    ModelProfile(re.compile(r"mistral-nemo"), 131072, tool_calling=True),
    ModelProfile(re.compile(r"mistral|mixtral"), 32768, tool_calling=True),
    ModelProfile(re.compile(r"firefunction|functionary"), 32768, tool_calling=True),
    ModelProfile(re.compile(r"command-?r"), 131072, tool_calling=True),
    ModelProfile(re.compile(r"phi-?3"), 131072, tool_calling=False),
    ModelProfile(re.compile(r"gemma-?2"), 8192, tool_calling=False),
    ModelProfile(re.compile(r"nomic-embed|mxbai-embed|bge-|all-minilm"), 8192, False, True),
)

_DEFAULT = Capabilities(
    tool_calling=False,
    json_schema=False,
    grammar=False,
    context_window=8192,
    max_output_tokens=2048,
    embeddings=False,
)


def lookup_capabilities(model: str) -> Capabilities:
    """Best-effort capability guess for a model name, before any live probe."""
    name = model.lower()
    for profile in _PROFILES:
        if profile.pattern.search(name):
            return Capabilities(
                tool_calling=profile.tool_calling,
                json_schema=False,
                grammar=False,
                context_window=profile.context_window,
                max_output_tokens=2048,
                embeddings=profile.embeddings,
            )
    return _DEFAULT.model_copy()
