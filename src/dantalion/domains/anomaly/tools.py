"""The read-only toolset the agent uses to investigate an anomaly.

Each tool is a typed function over the in-memory :class:`Dataset`, injected as
``dataset`` so it never appears in the schema the model sees. They are
deliberately read-only and side-effect-free: the worst a confused model can do is
ask a pointless question. Together they cover the moves a human on-call engineer
makes — list what is there, slice a time window, aggregate, correlate, diff two
windows, search logs, and look for spikes.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import UTC, datetime
from typing import Any, Literal

from dantalion.domains.anomaly.data import Dataset, parse_time
from dantalion.domains.anomaly.stats import aggregate as aggregate_values
from dantalion.domains.anomaly.stats import pearson
from dantalion.tools import Tool, tool

Aggregation = Literal["count", "sum", "mean", "min", "max", "std", "p50", "p90", "p95", "p99"]


@tool(inject=("dataset",))
def list_streams(dataset: Dataset) -> dict[str, Any]:
    """List every stream with its event count and time span."""
    summary: dict[str, Any] = {}
    for stream in dataset.streams():
        events = dataset.for_stream(stream)
        summary[stream] = {
            "count": len(events),
            "first": _iso(events[0].ts),
            "last": _iso(events[-1].ts),
            "example_fields": sorted(events[0].fields),
        }
    return {"streams": summary, "total_events": len(dataset)}


@tool(inject=("dataset",))
def slice_timewindow(
    dataset: Dataset, stream: str, start: str | None = None, end: str | None = None
) -> dict[str, Any]:
    """Count events for a stream within an optional [start, end] time window."""
    lo, hi = _bounds(start, end)
    events = [event for event in dataset.window(lo, hi) if event.stream == stream]
    return {
        "stream": stream,
        "count": len(events),
        "first": _iso(events[0].ts) if events else None,
        "last": _iso(events[-1].ts) if events else None,
    }


@tool(inject=("dataset",))
def aggregate(
    dataset: Dataset,
    stream: str,
    field: str,
    op: Aggregation,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Aggregate a numeric field of a stream (mean, p99, count, …)."""
    lo, hi = _bounds(start, end)
    series = [value for _, value in dataset.numeric_series(stream, field, lo, hi)]
    return {"stream": stream, "field": field, "op": op, "value": aggregate_values(series, op)}


@tool(inject=("dataset",))
def top_k(
    dataset: Dataset,
    stream: str,
    field: str,
    k: int = 5,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """The most frequent values of a (categorical) field on a stream."""
    lo, hi = _bounds(start, end)
    counter: Counter[str] = Counter()
    for event in dataset.window(lo, hi):
        if event.stream == stream and field in event.fields:
            counter[str(event.fields[field])] += 1
    return {"stream": stream, "field": field, "top": counter.most_common(k)}


@tool(inject=("dataset",))
def correlate(
    dataset: Dataset,
    stream_a: str,
    field_a: str,
    stream_b: str,
    field_b: str,
    bucket_seconds: int = 60,
) -> dict[str, Any]:
    """Pearson correlation of two numeric streams, bucketed over time."""
    buckets_a = _bucket_means(dataset.numeric_series(stream_a, field_a), bucket_seconds)
    buckets_b = _bucket_means(dataset.numeric_series(stream_b, field_b), bucket_seconds)
    shared = sorted(set(buckets_a) & set(buckets_b))
    xs = [buckets_a[bucket] for bucket in shared]
    ys = [buckets_b[bucket] for bucket in shared]
    return {
        "stream_a": stream_a,
        "stream_b": stream_b,
        "buckets_compared": len(shared),
        "pearson": pearson(xs, ys),
    }


@tool(inject=("dataset",))
def diff_windows(
    dataset: Dataset,
    stream: str,
    field: str,
    a_start: str,
    a_end: str,
    b_start: str,
    b_end: str,
    op: Aggregation = "mean",
) -> dict[str, Any]:
    """Compare an aggregate of a field between two time windows."""
    a = [value for _, value in dataset.numeric_series(stream, field, *_bounds(a_start, a_end))]
    b = [value for _, value in dataset.numeric_series(stream, field, *_bounds(b_start, b_end))]
    value_a = aggregate_values(a, op)
    value_b = aggregate_values(b, op)
    delta = value_b - value_a
    pct = (delta / value_a * 100.0) if value_a else None
    return {"op": op, "window_a": value_a, "window_b": value_b, "delta": delta, "pct_change": pct}


@tool(inject=("dataset",))
def sample_events(
    dataset: Dataset, stream: str, n: int = 5, start: str | None = None, end: str | None = None
) -> dict[str, Any]:
    """Return up to ``n`` raw events from a stream for a quick look."""
    lo, hi = _bounds(start, end)
    events = [event for event in dataset.window(lo, hi) if event.stream == stream][:n]
    return {
        "stream": stream,
        "events": [{"ts": _iso(event.ts), **event.fields} for event in events],
    }


@tool(inject=("dataset",))
def search_logs(
    dataset: Dataset, stream: str, pattern: str, field: str = "message", limit: int = 5
) -> dict[str, Any]:
    """Find events on a stream whose text field matches a regular expression."""
    matcher = re.compile(pattern, re.IGNORECASE)
    matches = []
    for event in dataset.for_stream(stream):
        text = str(event.fields.get(field, ""))
        if matcher.search(text):
            matches.append({"ts": _iso(event.ts), field: text})
            if len(matches) >= limit:
                break
    return {"stream": stream, "pattern": pattern, "matches": matches}


@tool(inject=("dataset",))
def describe(
    dataset: Dataset, stream: str, field: str, start: str | None = None, end: str | None = None
) -> dict[str, Any]:
    """Summary statistics for a numeric field: count, mean, std, quartiles."""
    lo, hi = _bounds(start, end)
    series = [value for _, value in dataset.numeric_series(stream, field, lo, hi)]
    if not series:
        return {"stream": stream, "field": field, "count": 0}
    return {
        "stream": stream,
        "field": field,
        "count": len(series),
        "mean": aggregate_values(series, "mean"),
        "std": aggregate_values(series, "std"),
        "min": aggregate_values(series, "min"),
        "p50": aggregate_values(series, "p50"),
        "p95": aggregate_values(series, "p95"),
        "max": aggregate_values(series, "max"),
    }


@tool(inject=("dataset",))
def detect_spikes(
    dataset: Dataset, stream: str, field: str, bucket_seconds: int = 60, z: float = 3.0
) -> dict[str, Any]:
    """Time buckets whose mean exceeds the overall mean by ``z`` std deviations."""
    buckets = _bucket_means(dataset.numeric_series(stream, field), bucket_seconds)
    if len(buckets) < 2:
        return {"stream": stream, "field": field, "spikes": [], "note": "not enough data"}
    means = list(buckets.values())
    overall = aggregate_values(means, "mean")
    spread = aggregate_values(means, "std")
    threshold = overall + z * spread
    spikes = [
        {"bucket_start": _iso(bucket * bucket_seconds), "value": value, "threshold": threshold}
        for bucket, value in sorted(buckets.items())
        if spread > 0 and value > threshold
    ]
    return {"stream": stream, "field": field, "spikes": spikes}


# -- registry & helpers --------------------------------------------------

ANOMALY_TOOLS: list[Tool] = [
    list_streams,
    slice_timewindow,
    aggregate,
    top_k,
    correlate,
    diff_windows,
    sample_events,
    search_logs,
    describe,
    detect_spikes,
]


def _bounds(start: str | None, end: str | None) -> tuple[float | None, float | None]:
    return (
        parse_time(start) if start is not None else None,
        parse_time(end) if end is not None else None,
    )


def _bucket_means(series: list[tuple[float, float]], bucket_seconds: int) -> dict[int, float]:
    if bucket_seconds <= 0:
        raise ValueError("bucket_seconds must be positive")
    grouped: dict[int, list[float]] = {}
    for ts, value in series:
        grouped.setdefault(int(ts // bucket_seconds), []).append(value)
    return {bucket: aggregate_values(values, "mean") for bucket, values in grouped.items()}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()
