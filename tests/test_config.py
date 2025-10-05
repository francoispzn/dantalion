from __future__ import annotations

from pathlib import Path

import pytest

from dantalion.config import Settings, make_provider, parse_model_spec
from dantalion.providers.mock import MockProvider
from dantalion.providers.ollama import OllamaProvider
from dantalion.providers.openai_compat import OpenAICompatibleProvider
from dantalion.trace import Cassette, ReplayProvider


def test_parse_model_spec() -> None:
    assert parse_model_spec("ollama:llama3.1") == ("ollama", "llama3.1")
    assert parse_model_spec("llama-cpp:/models/x.gguf") == ("llama-cpp", "/models/x.gguf")
    assert parse_model_spec("llama3.1") == ("ollama", "llama3.1")


def test_make_provider_builds_each_kind() -> None:
    assert isinstance(make_provider("ollama:llama3.1"), OllamaProvider)
    assert isinstance(
        make_provider("openai:qwen", base_url="http://x/v1", api_key="k"),
        OpenAICompatibleProvider,
    )
    assert isinstance(make_provider("mock:test"), MockProvider)


def test_make_provider_only_passes_relevant_options() -> None:
    # api_key would be rejected by the Ollama adapter if it leaked through.
    assert isinstance(make_provider("ollama:llama3.1", api_key="secret"), OllamaProvider)


def test_make_provider_replay_from_cassette(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    Cassette().save(path)
    assert isinstance(make_provider(f"replay:{path}"), ReplayProvider)


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DANTALION_MODEL", "openai:qwen2.5")
    monkeypatch.setenv("DANTALION_MAX_STEPS", "5")
    settings = Settings()
    assert settings.model == "openai:qwen2.5"
    assert settings.max_steps == 5
