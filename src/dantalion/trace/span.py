"""Lightweight, dependency-free tracing.

An autonomous agent is hard to trust if you cannot see what it did. The tracer
records nested spans — plan, each step, each tool call — with timings and
arbitrary attributes, and serialises them to JSONL that a human can read or a
tool can ingest. It is intentionally tiny: no OpenTelemetry dependency, just the
shape (named, nested, timed spans) that makes a run legible after the fact.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Span:
    """One named, timed unit of work, optionally nested under a parent."""

    name: str
    start: float
    end: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    parent: int | None = None

    @property
    def duration(self) -> float | None:
        return None if self.end is None else self.end - self.start


class Tracer:
    """Collects spans for one run and renders them as JSONL."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self.spans: list[Span] = []
        self._stack: list[int] = []

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        index = len(self.spans)
        parent = self._stack[-1] if self._stack else None
        span = Span(name=name, start=self._clock(), attributes=dict(attributes), parent=parent)
        self.spans.append(span)
        self._stack.append(index)
        try:
            yield span
        finally:
            span.end = self._clock()
            self._stack.pop()

    def to_records(self) -> list[dict[str, Any]]:
        return [
            {
                "name": span.name,
                "start": span.start,
                "end": span.end,
                "duration": span.duration,
                "parent": span.parent,
                "attributes": span.attributes,
            }
            for span in self.spans
        ]

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(record) for record in self.to_records())

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_jsonl() + "\n", encoding="utf-8")
