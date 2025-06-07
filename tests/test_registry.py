from __future__ import annotations

import pytest

from dantalion.providers.registry import lookup_capabilities


@pytest.mark.parametrize(
    ("model", "tool_calling", "min_context"),
    [
        ("llama3.1:8b", True, 131072),
        ("llama3:latest", True, 8192),
        ("qwen2.5-coder:7b", True, 32768),
        ("mistral-nemo", True, 131072),
        ("gemma2:9b", False, 8192),
        ("phi3:mini", False, 131072),
    ],
)
def test_known_models_get_reasonable_profiles(
    model: str, tool_calling: bool, min_context: int
) -> None:
    caps = lookup_capabilities(model)
    assert caps.tool_calling is tool_calling
    assert caps.context_window >= min_context


def test_unknown_model_falls_back_to_conservative_default() -> None:
    caps = lookup_capabilities("some-bespoke-model-v9")
    assert caps.tool_calling is False
    assert caps.context_window == 8192


def test_embedding_models_are_flagged() -> None:
    caps = lookup_capabilities("nomic-embed-text")
    assert caps.embeddings is True
