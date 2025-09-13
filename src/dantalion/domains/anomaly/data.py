"""Loading local logs and metrics into something the tools can query.

The agent never touches raw files; it works through this in-memory dataset, which
normalises CSV and JSONL into a single stream of timestamped events. Each event
belongs to a named *stream* (a metric series or a log source) and carries
arbitrary fields. Keeping the representation this plain means the analysis tools
stay small and the whole thing is trivially testable from fixtures, no files
required.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_TS_KEYS = ("ts", "timestamp", "time", "@timestamp", "datetime")
_STREAM_KEYS = ("stream", "metric", "source", "series", "name")


@dataclass(frozen=True)
class Event:
    """One timestamped record on a stream."""

    ts: float
    stream: str
    fields: dict[str, Any]
    raw_ts: str


class Dataset:
    """An ordered, in-memory collection of events."""

    def __init__(self, events: list[Event]) -> None:
        self.events = sorted(events, key=lambda event: event.ts)

    def __len__(self) -> int:
        return len(self.events)

    # -- loading ---------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        suffix = Path(path).suffix.lower()
        if suffix in (".jsonl", ".ndjson", ".json"):
            return cls.from_jsonl(path)
        if suffix == ".csv":
            return cls.from_csv(path)
        raise ValueError(f"unsupported dataset format: {suffix!r}")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Dataset:
        rows = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return cls.from_rows(rows)

    @classmethod
    def from_csv(cls, path: str | Path) -> Dataset:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        return cls.from_rows(rows)

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]]) -> Dataset:
        events = [event for event in (_row_to_event(row) for row in rows) if event is not None]
        return cls(events)

    # -- querying --------------------------------------------------------

    def streams(self) -> list[str]:
        seen: dict[str, None] = {}
        for event in self.events:
            seen.setdefault(event.stream, None)
        return list(seen)

    def for_stream(self, stream: str) -> list[Event]:
        return [event for event in self.events if event.stream == stream]

    def window(self, start: float | None, end: float | None) -> list[Event]:
        lo = start if start is not None else float("-inf")
        hi = end if end is not None else float("inf")
        return [event for event in self.events if lo <= event.ts <= hi]

    def bounds(self) -> tuple[float, float] | None:
        if not self.events:
            return None
        return self.events[0].ts, self.events[-1].ts

    def numeric_series(
        self, stream: str, field: str, start: float | None = None, end: float | None = None
    ) -> list[tuple[float, float]]:
        series = []
        for event in self.window(start, end):
            if event.stream != stream:
                continue
            value = event.fields.get(field)
            number = _as_number(value)
            if number is not None:
                series.append((event.ts, number))
        return series


def parse_time(value: Any) -> float:
    """Parse an epoch number or ISO-8601 string into epoch seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError as exc:
        raise ValueError(f"unparseable timestamp: {value!r}") from exc


def _row_to_event(row: dict[str, Any]) -> Event | None:
    raw_ts = _first_present(row, _TS_KEYS)
    if raw_ts is None:
        return None
    try:
        ts = parse_time(raw_ts)
    except ValueError:
        return None
    stream = _first_present(row, _STREAM_KEYS)
    consumed = _TS_KEYS + _STREAM_KEYS
    fields = {key: _coerce(value) for key, value in row.items() if key not in consumed}
    return Event(
        ts=ts,
        stream=str(stream) if stream is not None else "default",
        fields=fields,
        raw_ts=str(raw_ts),
    )


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _coerce(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
