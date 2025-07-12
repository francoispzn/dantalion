from __future__ import annotations

import pytest
from pydantic import BaseModel

from dantalion.errors import SchemaValidationError
from dantalion.providers.base import Capabilities
from dantalion.providers.mock import MockProvider
from dantalion.structured import structure
from dantalion.types import Message


class Report(BaseModel):
    title: str
    score: int


def _caps(*, json_schema: bool) -> Capabilities:
    return Capabilities(tool_calling=False, json_schema=json_schema, grammar=False)


def _prompt() -> list[Message]:
    return [Message.user("write a report")]


def test_schema_mode_sets_response_format_and_validates() -> None:
    provider = MockProvider(
        ['{"title": "disk pressure", "score": 8}'], capabilities=_caps(json_schema=True)
    )
    result = structure(provider, _prompt(), Report)

    assert result.value == Report(title="disk pressure", score=8)
    assert result.strategy == "schema"
    assert result.attempts == 1
    assert provider.requests[0].response_format is not None
    assert provider.requests[0].response_format.json_schema is not None


def test_prompt_mode_when_schema_unsupported() -> None:
    provider = MockProvider(['{"title": "x", "score": 1}'], capabilities=_caps(json_schema=False))
    result = structure(provider, _prompt(), Report)

    assert result.strategy == "prompt"
    assert provider.requests[0].response_format is None


def test_repair_path_is_reported() -> None:
    provider = MockProvider(['{"title": "x", "score": 1,}'], capabilities=_caps(json_schema=True))
    result = structure(provider, _prompt(), Report)

    assert result.value.score == 1
    assert result.repaired is True
    assert result.strategy == "repair"


def test_reflection_recovers_after_invalid_first_turn() -> None:
    provider = MockProvider(
        ["I cannot do that", '{"title": "ok", "score": 3}'],
        capabilities=_caps(json_schema=False),
    )
    result = structure(provider, _prompt(), Report, max_attempts=3)

    assert result.value.score == 3
    assert result.attempts == 2
    assert result.strategy == "reflection"
    reflection_turn = provider.requests[1].messages[-1]
    assert "not valid" in (reflection_turn.content or "")


def test_usage_accumulates_across_attempts() -> None:
    provider = MockProvider(
        ["garbage", '{"title": "ok", "score": 3}'],
        capabilities=_caps(json_schema=False),
    )
    result = structure(provider, _prompt(), Report, max_attempts=3)
    assert result.usage.total_tokens >= 1


def test_exhausting_attempts_raises_with_raw() -> None:
    provider = MockProvider(
        ["nope", "still bad", "{not json"],
        capabilities=_caps(json_schema=True),
    )
    with pytest.raises(SchemaValidationError) as excinfo:
        structure(provider, _prompt(), Report, max_attempts=3)
    assert excinfo.value.raw == "{not json"
