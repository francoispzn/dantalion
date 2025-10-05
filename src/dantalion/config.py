"""Configuration and provider resolution.

A model is named by a single ``kind:model`` spec — ``ollama:llama3.1``,
``openai:qwen2.5``, ``llama-cpp:/models/q4.gguf``, ``replay:run.json`` — which is
all most callers ever need. :class:`Settings` supplies the surrounding defaults
(base URL, budgets, whether to plan and review) and reads them from the
environment with a ``DANTALION_`` prefix, so the same binary behaves differently
across machines without code changes.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from dantalion.providers import Provider, build_provider

DEFAULT_SPEC = "ollama:llama3.1"


class Settings(BaseSettings):
    """Environment-driven defaults, overridable per invocation."""

    model_config = SettingsConfigDict(env_prefix="DANTALION_", env_file=".env", extra="ignore")

    model: str = DEFAULT_SPEC
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.1
    max_steps: int = 12
    max_tokens: int | None = None
    plan: bool = True
    review: bool = True


def parse_model_spec(spec: str, *, default_kind: str = "ollama") -> tuple[str, str]:
    """Split a ``kind:model`` spec; a bare value uses the default kind."""
    if ":" in spec:
        kind, model = spec.split(":", 1)
        return kind.strip(), model.strip()
    return default_kind, spec.strip()


def make_provider(
    spec: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    strict_replay: bool = True,
) -> Provider:
    """Build a provider from a spec, wiring only the options its kind accepts."""
    kind, model = parse_model_spec(spec)

    if kind == "replay":
        from dantalion.trace import Cassette, ReplayProvider

        return ReplayProvider(Cassette.load(model), strict=strict_replay)

    kwargs: dict[str, object] = {}
    if base_url and kind in ("ollama", "openai"):
        kwargs["base_url"] = base_url
    if api_key and kind == "openai":
        kwargs["api_key"] = api_key
    return build_provider(kind, model, **kwargs)
