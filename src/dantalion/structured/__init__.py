"""Structured-output: turning model text into validated pydantic objects."""

from __future__ import annotations

from dantalion.structured.extract import (
    extract_json_span,
    parse_lenient,
    repair_json,
    strip_fences,
)
from dantalion.structured.strategy import StructuredResult, structure

__all__ = [
    "StructuredResult",
    "extract_json_span",
    "parse_lenient",
    "repair_json",
    "strip_fences",
    "structure",
]
