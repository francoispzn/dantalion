"""Pulling clean JSON out of messy model output.

Even when asked nicely, local models wrap JSON in prose, fence it in Markdown,
add trailing commas, leave comments, or get cut off mid-object. These helpers are
the unglamorous, heavily-tested layer that turns that reality into something
``json.loads`` will accept — or admits defeat cleanly so the caller can ask the
model to try again.

Everything here is pure and deterministic, which is why it carries property-based
tests: whatever we do, valid JSON must survive a round trip unchanged.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n\r]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")

_OPENERS = {"{": "}", "[": "]"}
_CLOSERS = {"}", "]"}


def strip_fences(text: str) -> str:
    """Remove a Markdown code fence, even an unterminated one."""
    match = _FENCE.search(text)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json|JSON)?", "", stripped).strip()
    return stripped


def extract_json_span(text: str) -> str | None:
    """Return the first balanced JSON object/array substring, if any.

    String contents and escapes are respected so braces inside strings do not
    confuse the bracket counter. An unterminated value is returned as-is on the
    theory that :func:`repair_json` may still rescue it.
    """
    body = strip_fences(text)
    start = _first_opener(body)
    if start is None:
        return None

    opener = body[start]
    closer = _OPENERS[opener]
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(body)):
        char = body[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return body[start : index + 1]
    return body[start:]


def repair_json(text: str) -> str:
    """Apply conservative, structure-preserving fixes to near-JSON."""
    candidate = strip_fences(text).strip()
    candidate = _BLOCK_COMMENT.sub("", candidate)
    candidate = _LINE_COMMENT.sub("", candidate)
    candidate = _TRAILING_COMMA.sub(r"\1", candidate)
    candidate = _close_open_structures(candidate)
    candidate = _TRAILING_COMMA.sub(r"\1", candidate)
    return candidate.strip()


def parse_lenient(text: str) -> tuple[Any, bool] | None:
    """Best-effort parse. Returns ``(value, repaired)`` or ``None``.

    Tries the extracted span verbatim first; only if that fails does it apply
    repairs, so well-formed output is never "fixed" into something subtly wrong.
    """
    span = extract_json_span(text)
    if span is None:
        return None
    try:
        return json.loads(span), False
    except ValueError:
        pass
    repaired = repair_json(span)
    try:
        return json.loads(repaired), True
    except ValueError:
        return None


def _first_opener(text: str) -> int | None:
    for index, char in enumerate(text):
        if char in _OPENERS:
            return index
    return None


def _close_open_structures(text: str) -> str:
    """Append closers (and a quote) for a value the model left dangling."""
    stack: list[str] = []
    in_string = False
    escaped = False

    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in _OPENERS:
            stack.append(_OPENERS[char])
        elif char in _CLOSERS and stack and stack[-1] == char:
            stack.pop()

    suffix = ""
    if in_string:
        suffix += '"'
    suffix += "".join(reversed(stack))
    return text + suffix
