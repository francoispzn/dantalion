"""Structured-output: turning model text into validated pydantic objects."""

from __future__ import annotations

from dantalion.structured.extract import (
    extract_json_span,
    parse_lenient,
    repair_json,
    strip_fences,
)
from dantalion.structured.gbnf import model_to_gbnf, schema_to_gbnf
from dantalion.structured.strategy import StructuredResult, structure

__all__ = [
    "StructuredResult",
    "extract_json_span",
    "model_to_gbnf",
    "parse_lenient",
    "repair_json",
    "schema_to_gbnf",
    "strip_fences",
    "structure",
]
