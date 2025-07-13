"""Model providers and the factory that builds them.

Importing from here, rather than from the concrete modules, keeps call sites
decoupled from any single backend. ``build_provider`` is the one place that knows
how a ``kind`` string maps to an adapter.
"""

from __future__ import annotations

from typing import Any

from dantalion.errors import ConfigError
from dantalion.providers.base import Capabilities, Provider
from dantalion.providers.llama_cpp import LlamaCppProvider
from dantalion.providers.mock import MockProvider
from dantalion.providers.ollama import OllamaProvider
from dantalion.providers.openai_compat import OpenAICompatibleProvider

__all__ = [
    "Capabilities",
    "LlamaCppProvider",
    "MockProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "Provider",
    "build_provider",
]

_ALIASES = {
    "openai-compatible": "openai",
    "vllm": "openai",
    "lmstudio": "openai",
    "llamacpp": "llama-cpp",
    "llama.cpp": "llama-cpp",
}


def build_provider(kind: str, model: str, **kwargs: Any) -> Provider:
    """Construct a provider by name.

    ``kind`` selects the adapter (``ollama``, ``openai``, ``llama-cpp``,
    ``mock``); ``kwargs`` are passed through, so e.g. ``base_url`` reaches the
    adapter unchanged. Common synonyms (``vllm``, ``lmstudio``,
    ``openai-compatible``, ``llamacpp``) map to their adapter.
    """
    kind = _ALIASES.get(kind, kind)
    if kind == "ollama":
        return OllamaProvider(model, **kwargs)
    if kind == "openai":
        return OpenAICompatibleProvider(model, **kwargs)
    if kind == "llama-cpp":
        return LlamaCppProvider(model, **kwargs)
    if kind == "mock":
        return MockProvider(model=model, **kwargs)
    raise ConfigError(f"unknown provider kind: {kind!r}")
