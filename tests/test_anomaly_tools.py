from __future__ import annotations

from typing import Any

from dantalion.domains.anomaly.data import Dataset
from dantalion.domains.anomaly.tools import (
    aggregate,
    correlate,
    describe,
    detect_spikes,
    diff_windows,
    list_streams,
    search_logs,
    slice_timewindow,
    top_k,
)


def _ctx(dataset: Dataset) -> dict[str, Any]:
    return {"dataset": dataset}


def test_list_streams(incident_dataset: Dataset) -> None:
    result = list_streams.run({}, context=_ctx(incident_dataset))
    assert result.ok
    assert set(result.data["streams"]) == {"latency_ms", "disk_used_pct", "app_log"}
    assert result.data["total_events"] == 20


def test_aggregate_count_and_mean(incident_dataset: Dataset) -> None:
    count = aggregate.run(
        {"stream": "latency_ms", "field": "value", "op": "count"}, context=_ctx(incident_dataset)
    )
    assert count.data["value"] == 10
    mean = aggregate.run(
        {"stream": "latency_ms", "field": "value", "op": "mean"}, context=_ctx(incident_dataset)
    )
    assert mean.data["value"] > 0


def test_slice_timewindow(incident_dataset: Dataset) -> None:
    result = slice_timewindow.run(
        {
            "stream": "latency_ms",
            "start": "2025-01-01T00:05:00Z",
            "end": "2025-01-01T00:09:00Z",
        },
        context=_ctx(incident_dataset),
    )
    assert result.data["count"] == 5


def test_top_k_counts_log_levels(incident_dataset: Dataset) -> None:
    result = top_k.run({"stream": "app_log", "field": "level"}, context=_ctx(incident_dataset))
    counts = dict(result.data["top"])
    assert counts["ERROR"] == 2


def test_correlate_finds_positive_relationship(incident_dataset: Dataset) -> None:
    result = correlate.run(
        {
            "stream_a": "latency_ms",
            "field_a": "value",
            "stream_b": "disk_used_pct",
            "field_b": "value",
        },
        context=_ctx(incident_dataset),
    )
    assert result.data["pearson"] is not None
    assert result.data["pearson"] > 0.5


def test_diff_windows_shows_regression(incident_dataset: Dataset) -> None:
    result = diff_windows.run(
        {
            "stream": "latency_ms",
            "field": "value",
            "a_start": "2025-01-01T00:00:00Z",
            "a_end": "2025-01-01T00:04:00Z",
            "b_start": "2025-01-01T00:05:00Z",
            "b_end": "2025-01-01T00:07:00Z",
        },
        context=_ctx(incident_dataset),
    )
    assert result.data["pct_change"] is not None
    assert result.data["pct_change"] > 100


def test_search_logs(incident_dataset: Dataset) -> None:
    result = search_logs.run(
        {"stream": "app_log", "pattern": "disk"}, context=_ctx(incident_dataset)
    )
    assert len(result.data["matches"]) >= 1


def test_describe(incident_dataset: Dataset) -> None:
    result = describe.run(
        {"stream": "latency_ms", "field": "value"}, context=_ctx(incident_dataset)
    )
    assert result.data["count"] == 10
    assert result.data["max"] >= 1300


def test_detect_spikes(incident_dataset: Dataset) -> None:
    result = detect_spikes.run(
        {"stream": "latency_ms", "field": "value", "z": 1.0}, context=_ctx(incident_dataset)
    )
    assert len(result.data["spikes"]) >= 1


def test_invalid_aggregation_is_rejected(incident_dataset: Dataset) -> None:
    result = aggregate.run(
        {"stream": "latency_ms", "field": "value", "op": "median"}, context=_ctx(incident_dataset)
    )
    assert not result.ok
