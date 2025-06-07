"""Model providers and the factory that builds them.

Importing from here, rather than from the concrete modules, keeps call sites
decoupled from any single backend. ``build_provider`` is the one place that knows
how a ``kind`` string maps to an adapter.
"""

from __future__ import annotations

from typing import Any

from dantalion.errors import ConfigError
from dantalion.providers.base import Capabilities, Provider
from dantalion.providers.mock import MockProvider
from dantalion.providers.ollama import OllamaProvider

__all__ = [
    "Capabilities",
    "MockProvider",
    "OllamaProvider",
    "Provider",
    "build_provider",
]


def build_provider(kind: str, model: str, **kwargs: Any) -> Provider:
    """Construct a provider by name.

    ``kind`` selects the adapter (``ollama``, ``mock``); ``kwargs`` are passed
    through, so e.g. ``base_url`` reaches the Ollama adapter unchanged.
    """
    if kind == "ollama":
        return OllamaProvider(model, **kwargs)
    if kind == "mock":
        return MockProvider(model=model, **kwargs)
    raise ConfigError(f"unknown provider kind: {kind!r}")
