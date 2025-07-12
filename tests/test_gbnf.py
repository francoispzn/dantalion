from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from dantalion.structured.gbnf import model_to_gbnf, schema_to_gbnf

_STRING_LITERAL = re.compile(r'"(?:[^"\\]|\\.)*"')
_CHAR_CLASS = re.compile(r"\[(?:[^\]\\]|\\.)*\]")
_IDENTIFIER = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


class Inner(BaseModel):
    label: str
    weight: float


class Outer(BaseModel):
    name: str
    count: int
    active: bool
    tags: list[str]
    inner: Inner
    kind: Literal["a", "b"]
    note: str | None = None


class AllOptional(BaseModel):
    a: int = 0
    b: str = ""


def _defined_and_referenced(grammar: str) -> tuple[set[str], set[str]]:
    defined: set[str] = set()
    referenced: set[str] = set()
    for line in grammar.strip().splitlines():
        if "::=" not in line:
            continue
        head, body = line.split("::=", 1)
        defined.add(head.strip())
        without_literals = _CHAR_CLASS.sub(" ", _STRING_LITERAL.sub(" ", body))
        referenced.update(_IDENTIFIER.findall(without_literals))
    return defined, referenced


def test_grammar_is_self_consistent() -> None:
    grammar = model_to_gbnf(Outer)
    defined, referenced = _defined_and_referenced(grammar)
    assert "root" in defined
    assert referenced <= defined, f"dangling rules: {referenced - defined}"


def test_grammar_pins_property_keys() -> None:
    grammar = model_to_gbnf(Outer)
    for key in ("name", "count", "active", "tags", "inner", "kind"):
        assert f'\\"{key}\\"' in grammar


def test_literal_becomes_quoted_alternation() -> None:
    grammar = model_to_gbnf(Outer)
    assert '"\\"a\\""' in grammar
    assert '"\\"b\\""' in grammar


def test_scalar_primitives_are_emitted_when_used() -> None:
    grammar = model_to_gbnf(Outer)
    defined, _ = _defined_and_referenced(grammar)
    assert {"string", "integer", "number", "boolean", "ws"} <= defined


def test_optional_field_is_optional_group() -> None:
    grammar = model_to_gbnf(Outer)
    # the optional "note" appears inside a (... )? group
    assert re.search(r"\(ws \",\" ws \"\\\"note\\\"\".*\)\?", grammar)


def test_all_optional_object_forces_every_key() -> None:
    grammar = model_to_gbnf(AllOptional)
    defined, referenced = _defined_and_referenced(grammar)
    assert referenced <= defined
    assert '\\"a\\"' in grammar
    assert '\\"b\\"' in grammar


def test_freeform_object_uses_generic_rules() -> None:
    grammar = schema_to_gbnf({"type": "object"})
    defined, referenced = _defined_and_referenced(grammar)
    assert "object" in defined
    assert referenced <= defined


def test_plain_array_schema() -> None:
    grammar = schema_to_gbnf({"type": "array", "items": {"type": "integer"}})
    defined, referenced = _defined_and_referenced(grammar)
    assert referenced <= defined
    assert "integer" in defined
