"""The runtime representation of a tool.

A :class:`Tool` couples three things: a JSON Schema the model is shown, a
pydantic model that validates whatever the model sends back, and the Python
callable that actually does the work. Keeping validation next to execution means
a hallucinated argument is rejected with a useful message instead of blowing up
somewhere deep in a handler.

Tools never raise into the agent loop. A bad argument or a failing handler comes
back as a :class:`ToolResult` with ``ok=False``, which the model can read and
react to — recovering from its own mistakes is part of the job.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from dantalion.types import ToolSpec


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one tool invocation.

    ``data`` holds the structured result, which the agent keeps as evidence and
    the tracer records verbatim. ``to_content`` renders the string the model
    actually sees on its next turn.
    """

    ok: bool
    data: Any = None
    error: str | None = None

    def to_content(self) -> str:
        if not self.ok:
            return json.dumps({"error": self.error or "tool failed"})
        return json.dumps(self.data, default=str, ensure_ascii=False)


@dataclass(frozen=True)
class Tool:
    """A validated, schema-bearing wrapper around a callable."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    args_model: type[BaseModel]
    arg_names: tuple[str, ...] = ()
    inject: tuple[str, ...] = ()
    read_only: bool = True
    timeout: float | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def spec(self) -> ToolSpec:
        """The provider-facing description shown to the model."""
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    def run(
        self, arguments: Mapping[str, Any], *, context: Mapping[str, Any] | None = None
    ) -> ToolResult:
        """Validate ``arguments``, inject dependencies, and execute."""
        context = context or {}
        try:
            validated = self.args_model.model_validate(dict(arguments))
        except ValidationError as exc:
            return ToolResult(ok=False, error=_format_validation_error(exc))

        kwargs = {name: getattr(validated, name) for name in self.arg_names}
        injected: dict[str, Any] = {}
        for name in self.inject:
            if name not in context:
                return ToolResult(ok=False, error=f"missing injected dependency: {name!r}")
            injected[name] = context[name]

        try:
            result = self.func(**injected, **kwargs)
        except Exception as exc:  # tools must not crash the loop
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")

        if isinstance(result, ToolResult):
            return result
        return ToolResult(ok=True, data=result)


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        location = ".".join(str(p) for p in err["loc"]) or "(root)"
        parts.append(f"{location}: {err['msg']}")
    return "invalid arguments: " + "; ".join(parts)
