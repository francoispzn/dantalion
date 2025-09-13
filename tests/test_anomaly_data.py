from __future__ import annotations

from typing import Any

from dantalion.domains.anomaly.data import Dataset, parse_time


def _rows() -> list[dict[str, Any]]:
    return [
        {"ts": "2025-01-01T00:00:00Z", "stream": "latency_ms", "value": 100},
        {"ts": "2025-01-01T00:01:00Z", "stream": "latency_ms", "value": "200"},
        {"ts": 1735689720, "stream": "disk", "value": 50},
        {"stream": "no_timestamp"},
    ]


def test_from_rows_parses_coerces_and_sorts() -> None:
    dataset = Dataset.from_rows(_rows())
    assert len(dataset) == 3  # the row without a timestamp is dropped
    assert dataset.streams() == ["latency_ms", "disk"]
    series = dataset.numeric_series("latency_ms", "value")
    assert [value for _, value in series] == [100.0, 200.0]  # "200" coerced


def test_parse_time_accepts_epoch_and_iso() -> None:
    assert parse_time(1735689600) == 1735689600.0
    assert parse_time("1735689600") == 1735689600.0
    assert parse_time("2025-01-01T00:00:00Z") > 0


def test_window_filters_inclusively() -> None:
    dataset = Dataset.from_rows(_rows())
    bounds = dataset.bounds()
    assert bounds is not None
    inside = dataset.window(bounds[0], bounds[0])
    assert len(inside) == 1


def test_loads_example_incident(incident_dataset: Dataset) -> None:
    assert len(incident_dataset) == 20
    assert set(incident_dataset.streams()) == {"latency_ms", "disk_used_pct", "app_log"}
