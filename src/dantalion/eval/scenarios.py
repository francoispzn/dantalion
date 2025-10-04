"""Synthetic incidents with a planted root cause.

Each generator fabricates a small, noisy but realistic dataset where the cause is
known by construction, so an investigation against it can be graded
automatically. They are seeded, so a scenario is reproducible: the same seed
always yields the same data, which is what lets the evaluator double as a
regression test.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from dantalion.domains.anomaly.data import Dataset

_BASE = datetime(2025, 1, 1, tzinfo=UTC)


@dataclass
class Scenario:
    """A generated incident plus the answer it was built around."""

    name: str
    dataset: Dataset
    alert: str
    gold_keywords: list[str]
    relevant_streams: list[str]


def _ts(minute: int) -> str:
    return (_BASE + timedelta(minutes=minute)).isoformat()


def _metric(minute: int, stream: str, value: float) -> dict[str, Any]:
    return {"ts": _ts(minute), "stream": stream, "value": round(value, 2)}


def _log(minute: int, level: str, message: str) -> dict[str, Any]:
    return {"ts": _ts(minute), "stream": "app_log", "level": level, "message": message}


def disk_exhaustion(seed: int = 0, *, minutes: int = 24, onset: int = 12) -> Scenario:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for minute in range(minutes):
        if minute < onset:
            rows.append(_metric(minute, "latency_ms", 100 + rng.uniform(-5, 5)))
            rows.append(_metric(minute, "disk_used_pct", 40 + minute * 0.6 + rng.uniform(-1, 1)))
        else:
            rows.append(_metric(minute, "latency_ms", 1100 + rng.uniform(-50, 200)))
            rows.append(_metric(minute, "disk_used_pct", 93 + rng.uniform(-2, 3)))
            rows.append(_log(minute, "ERROR", "disk write failed: no space left on device"))
    return Scenario(
        name="disk_exhaustion",
        dataset=Dataset.from_rows(rows),
        alert=f"latency spiked sharply around minute {onset}",
        gold_keywords=["disk", "space"],
        relevant_streams=["latency_ms", "disk_used_pct"],
    )


def memory_leak(seed: int = 0, *, minutes: int = 24, onset: int = 16) -> Scenario:
    rng = random.Random(seed + 1)
    rows: list[dict[str, Any]] = []
    for minute in range(minutes):
        rows.append(_metric(minute, "memory_mb", 500 + minute * 40 + rng.uniform(-10, 10)))
        if minute < onset:
            rows.append(_metric(minute, "latency_ms", 120 + rng.uniform(-8, 8)))
        else:
            rows.append(_metric(minute, "latency_ms", 800 + rng.uniform(-40, 120)))
            rows.append(_log(minute, "ERROR", "OutOfMemoryError: heap exhausted"))
    return Scenario(
        name="memory_leak",
        dataset=Dataset.from_rows(rows),
        alert="the service slowed down and started crashing late in the window",
        gold_keywords=["memory", "oom", "heap", "leak"],
        relevant_streams=["memory_mb", "latency_ms"],
    )


def deploy_regression(seed: int = 0, *, minutes: int = 24, onset: int = 10) -> Scenario:
    rng = random.Random(seed + 2)
    rows: list[dict[str, Any]] = []
    for minute in range(minutes):
        if minute == onset:
            rows.append(_log(minute, "INFO", "deployed release v2.3.0"))
        rate = (0.5 + rng.uniform(0, 0.3)) if minute < onset else (9.0 + rng.uniform(-1, 2))
        rows.append(_metric(minute, "error_rate", rate))
    return Scenario(
        name="deploy_regression",
        dataset=Dataset.from_rows(rows),
        alert=f"error rate jumped after minute {onset}",
        gold_keywords=["deploy", "release", "regression", "v2.3"],
        relevant_streams=["error_rate"],
    )


def default_scenarios(seed: int = 0) -> list[Scenario]:
    """The standard suite the evaluator runs."""
    return [disk_exhaustion(seed), memory_leak(seed), deploy_regression(seed)]
