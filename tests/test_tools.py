from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import Literal

import pytest

from dantalion.errors import ToolError, ToolNotFound
from dantalion.tools import Tool, ToolRegistry, ToolResult, tool


@tool
def sample(stream: str, limit: int = 10) -> dict[str, object]:
    """Sample events from a stream."""
    return {"stream": stream, "limit": limit}


@tool(name="picker", description="Pick an aggregation.")
def pick(op: Literal["sum", "mean"]) -> str:
    return op


@tool(inject=("store",))
def fetch(store: dict[str, int], key: str) -> int:
    return store[key]


@tool
def boom(x: int) -> int:
    raise ValueError("nope")


def test_schema_is_derived_from_signature() -> None:
    props = sample.parameters["properties"]
    assert props["stream"]["type"] == "string"
    assert props["limit"]["type"] == "integer"
    assert sample.parameters["required"] == ["stream"]
    assert "title" not in sample.parameters


def test_description_comes_from_docstring_or_override() -> None:
    assert sample.description == "Sample events from a stream."
    assert pick.name == "picker"
    assert pick.description == "Pick an aggregation."


def test_literal_becomes_enum() -> None:
    assert pick.parameters["properties"]["op"]["enum"] == ["sum", "mean"]


def test_run_with_valid_arguments() -> None:
    result = sample.run({"stream": "errors", "limit": 5})
    assert result.ok
    assert result.data == {"stream": "errors", "limit": 5}


def test_defaults_are_applied() -> None:
    result = sample.run({"stream": "errors"})
    assert result.ok
    assert result.data == {"stream": "errors", "limit": 10}


def test_missing_required_argument_is_reported() -> None:
    result = sample.run({})
    assert not result.ok
    assert result.error is not None
    assert "stream" in result.error


def test_wrong_type_is_reported() -> None:
    result = sample.run({"stream": "errors", "limit": "lots"})
    assert not result.ok
    assert result.error is not None
    assert "limit" in result.error


def test_enum_violation_is_reported() -> None:
    result = pick.run({"op": "median"})
    assert not result.ok


def test_injected_dependency_is_supplied_from_context() -> None:
    result = fetch.run({"key": "a"}, context={"store": {"a": 7}})
    assert result.ok
    assert result.data == 7


def test_missing_injected_dependency_is_reported() -> None:
    result = fetch.run({"key": "a"})
    assert not result.ok
    assert result.error is not None
    assert "store" in result.error


def test_handler_exception_becomes_failed_result() -> None:
    result = boom.run({"x": 1})
    assert not result.ok
    assert result.error is not None
    assert "ValueError" in result.error


def test_handler_may_return_tool_result_directly() -> None:
    @tool
    def custom(x: int) -> ToolResult:
        return ToolResult(ok=False, error="explicit")

    assert custom.run({"x": 1}) == ToolResult(ok=False, error="explicit")


def test_tool_result_content_serialisation() -> None:
    assert json.loads(ToolResult(ok=True, data={"n": 1}).to_content()) == {"n": 1}
    assert json.loads(ToolResult(ok=False, error="bad").to_content()) == {"error": "bad"}


def test_registry_registers_and_looks_up() -> None:
    registry = ToolRegistry([sample, pick])
    assert "sample" in registry
    assert registry.get("picker") is pick
    assert len(registry) == 2
    assert set(registry.names()) == {"sample", "picker"}


def test_registry_rejects_duplicates() -> None:
    registry = ToolRegistry([sample])
    with pytest.raises(ToolError):
        registry.register(sample)


def test_registry_unknown_tool_raises() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolNotFound):
        registry.get("missing")


def test_registry_specs_match_tools() -> None:
    registry = ToolRegistry([sample])
    specs = registry.specs()
    assert specs[0].name == "sample"
    assert specs[0].parameters == sample.parameters


def test_tool_is_immutable() -> None:
    assert isinstance(sample, Tool)
    with pytest.raises(FrozenInstanceError):
        sample.name = "other"  # type: ignore[misc]
