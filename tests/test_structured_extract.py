from __future__ import annotations

import json
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

from dantalion.structured.extract import (
    extract_json_span,
    parse_lenient,
    repair_json,
    strip_fences,
)

_text = st.text(
    alphabet=st.characters(codec="utf-8", exclude_characters="`"),
    max_size=20,
)
_scalars = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | _text
)
_json = st.recursive(
    _scalars,
    lambda children: st.lists(children, max_size=4) | st.dictionaries(_text, children, max_size=4),
    max_leaves=8,
)
_containers = st.dictionaries(_text, _json, max_size=4) | st.lists(_json, max_size=4)


def test_strip_fences_handles_json_block() -> None:
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_fences_handles_unterminated_block() -> None:
    assert strip_fences('```json\n{"a": 1}').strip() == '{"a": 1}'


def test_extract_finds_object_inside_prose() -> None:
    text = 'Here is the result: {"root_cause": "disk"} — hope that helps.'
    assert extract_json_span(text) == '{"root_cause": "disk"}'


def test_extract_respects_braces_in_strings() -> None:
    text = '{"note": "use {curly} braces", "n": 1}'
    assert extract_json_span(text) == text


def test_extract_handles_nested_structures() -> None:
    text = 'prefix [{"a": [1, 2]}, {"b": {}}] suffix'
    assert extract_json_span(text) == '[{"a": [1, 2]}, {"b": {}}]'


def test_extract_returns_none_without_json() -> None:
    assert extract_json_span("no structure here") is None


def test_repair_removes_trailing_commas() -> None:
    assert json.loads(repair_json('{"a": 1, "b": 2,}')) == {"a": 1, "b": 2}


def test_repair_strips_comments() -> None:
    raw = '{\n  "a": 1, // inline\n  /* block */ "b": 2\n}'
    assert json.loads(repair_json(raw)) == {"a": 1, "b": 2}


def test_repair_closes_truncated_object() -> None:
    assert json.loads(repair_json('{"a": 1, "b": [1, 2')) == {"a": 1, "b": [1, 2]}


def test_repair_closes_truncated_string() -> None:
    assert json.loads(repair_json('{"a": "unterminated')) == {"a": "unterminated"}


def test_parse_lenient_marks_repair() -> None:
    assert parse_lenient('{"a": 1}') == ({"a": 1}, False)
    assert parse_lenient('{"a": 1,}') == ({"a": 1}, True)
    assert parse_lenient("nothing") is None


@given(_containers)
def test_valid_json_round_trips_without_repair(value: Any) -> None:
    rendered = json.dumps(value)
    assert parse_lenient(rendered) == (value, False)


@given(_containers)
def test_json_survives_prose_and_fences(value: Any) -> None:
    rendered = json.dumps(value)
    wrapped = f"Result follows:\n```json\n{rendered}\n```\nDone."
    parsed = parse_lenient(wrapped)
    assert parsed is not None
    assert parsed[0] == value
