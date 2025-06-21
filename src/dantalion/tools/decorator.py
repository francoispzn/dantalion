"""Turn an ordinary typed function into a :class:`Tool`.

The decorator reads the function's signature and type hints, builds a pydantic
model from the parameters, and derives a JSON Schema from that. The single source
of truth is therefore the function itself — there is no second place to update a
parameter, and no schema that can drift out of sync with the code.

Parameters named in ``inject`` are runtime dependencies (a data store, say), not
things the model chooses; they are hidden from the schema and supplied from the
execution context at call time.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any, get_type_hints, overload

from pydantic import create_model

from dantalion.tools.base import Tool

_IGNORED_KINDS = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def function_to_tool(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    inject: Sequence[str] = (),
    read_only: bool = True,
    timeout: float | None = None,
    tags: Sequence[str] = (),
) -> Tool:
    """Build a :class:`Tool` from a typed callable."""
    signature = inspect.signature(func)
    hints = get_type_hints(func, include_extras=True)
    inject_set = set(inject)

    fields: dict[str, Any] = {}
    arg_names: list[str] = []
    for param_name, param in signature.parameters.items():
        if param_name in inject_set or param.kind in _IGNORED_KINDS:
            continue
        annotation = hints.get(param_name, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param_name] = (annotation, default)
        arg_names.append(param_name)

    args_model = create_model(f"{func.__name__}_Args", **fields)
    parameters = _clean_schema(args_model.model_json_schema())

    return Tool(
        name=name or func.__name__,
        description=description or _first_line(func.__doc__),
        parameters=parameters,
        func=func,
        args_model=args_model,
        arg_names=tuple(arg_names),
        inject=tuple(inject),
        read_only=read_only,
        timeout=timeout,
        tags=tuple(tags),
    )


@overload
def tool(func: Callable[..., Any]) -> Tool: ...


@overload
def tool(
    *,
    name: str | None = ...,
    description: str | None = ...,
    inject: Sequence[str] = ...,
    read_only: bool = ...,
    timeout: float | None = ...,
    tags: Sequence[str] = ...,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    inject: Sequence[str] = (),
    read_only: bool = True,
    timeout: float | None = None,
    tags: Sequence[str] = (),
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Decorator form of :func:`function_to_tool`.

    Works bare (``@tool``) or configured (``@tool(inject=("store",))``).
    """

    def wrap(target: Callable[..., Any]) -> Tool:
        return function_to_tool(
            target,
            name=name,
            description=description,
            inject=inject,
            read_only=read_only,
            timeout=timeout,
            tags=tags,
        )

    if func is not None:
        return wrap(func)
    return wrap


def _first_line(docstring: str | None) -> str:
    if not docstring:
        return ""
    for line in docstring.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop pydantic's ``title`` noise so the model sees a lean schema."""
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        if isinstance(prop, dict):
            prop.pop("title", None)
    for definition in schema.get("$defs", {}).values():
        if isinstance(definition, dict):
            definition.pop("title", None)
    return schema
