"""Coaxing a typed object out of a model, whatever that model can do.

This is the heart of the project's model-agnosticism. A frontier model with a
JSON mode and a tiny quantised model that has never heard of JSON Schema must
both, in the end, hand back something that validates against a pydantic type.
The strategy adapts to the model's declared capabilities and degrades in clear
steps:

1. **schema** — ask the server to constrain decoding to the JSON Schema.
2. **prompt** — no constraint available, so describe the schema and ask for JSON.
3. **repair** — parse leniently, fixing fences/commas/truncation.
4. **reflection** — show the model its own validation error and ask for a fix.

The return value records which rung it landed on, which is invaluable when
comparing models in the evaluator.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from dantalion.errors import SchemaValidationError
from dantalion.providers.base import Provider
from dantalion.structured.extract import parse_lenient
from dantalion.types import CompletionRequest, Message, ResponseFormat, Usage

T = TypeVar("T", bound=BaseModel)

_NAME_SANITISER = re.compile(r"[^a-zA-Z0-9_]")


@dataclass
class StructuredResult(Generic[T]):
    """A validated object plus how much work it took to get it."""

    value: T
    raw: str
    usage: Usage
    attempts: int
    strategy: str
    repaired: bool


def structure(
    provider: Provider,
    messages: Sequence[Message],
    model_type: type[T],
    *,
    temperature: float = 0.0,
    seed: int | None = None,
    max_attempts: int = 3,
) -> StructuredResult[T]:
    """Return an instance of ``model_type`` produced by ``provider``."""
    caps = provider.capabilities()
    schema = model_type.model_json_schema()
    base_strategy = "schema" if caps.json_schema else "prompt"
    response_format = (
        ResponseFormat(name=_schema_name(model_type), json_schema=schema)
        if caps.json_schema
        else None
    )

    conversation = [*messages, Message.user(_schema_instruction(schema))]
    usage = Usage()
    last_raw = ""
    last_error = "no response"

    for attempt in range(1, max_attempts + 1):
        response = provider.complete(
            CompletionRequest(
                messages=conversation,
                response_format=response_format,
                temperature=temperature,
                seed=seed,
            )
        )
        usage += response.usage
        last_raw = response.message.content or ""

        outcome = _try_validate(last_raw, model_type)
        if isinstance(outcome, _Parsed):
            return StructuredResult(
                value=outcome.value,
                raw=last_raw,
                usage=usage,
                attempts=attempt,
                strategy=_final_strategy(base_strategy, attempt, outcome.repaired),
                repaired=outcome.repaired,
            )

        last_error = outcome.error
        conversation.append(Message.assistant(last_raw))
        conversation.append(Message.user(_reflection_instruction(last_error)))

    raise SchemaValidationError(
        f"no valid structured output after {max_attempts} attempts ({last_error})",
        raw=last_raw,
    )


@dataclass
class _Parsed(Generic[T]):
    value: T
    repaired: bool


@dataclass
class _Failed:
    error: str


def _try_validate(raw: str, model_type: type[T]) -> _Parsed[T] | _Failed:
    parsed = parse_lenient(raw)
    if parsed is None:
        return _Failed("no JSON object found in the response")
    obj, repaired = parsed
    try:
        value = model_type.model_validate(obj)
    except ValidationError as exc:
        return _Failed(_short_error(exc))
    return _Parsed(value=value, repaired=repaired)


def _final_strategy(base: str, attempt: int, repaired: bool) -> str:
    if attempt > 1:
        return "reflection"
    return "repair" if repaired else base


def _schema_name(model_type: type[BaseModel]) -> str:
    return _NAME_SANITISER.sub("_", model_type.__name__) or "response"


def _schema_instruction(schema: dict[str, Any]) -> str:
    return (
        "Respond with a single JSON object that conforms to this JSON Schema:\n"
        f"{json.dumps(schema, indent=2)}\n"
        "Return only the JSON object — no explanation, no Markdown code fences."
    )


def _reflection_instruction(error: str) -> str:
    return (
        f"That response was not valid: {error}. "
        "Return a corrected JSON object that satisfies the schema. Output only JSON."
    )


def _short_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "validation failed"
    first = errors[0]
    location = ".".join(str(p) for p in first["loc"]) or "(root)"
    suffix = f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
    return f"{location}: {first['msg']}{suffix}"
