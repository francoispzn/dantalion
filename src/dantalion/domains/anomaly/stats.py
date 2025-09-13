"""Small statistical helpers, hand-rolled to keep the dependency surface tiny.

Nothing here is novel; it is deliberately implemented in plain Python (plus the
stdlib ``statistics`` module) rather than pulling in pandas/numpy, so the package
installs light and the maths is right there to read.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

_PERCENTILE_OPS = {"p50": 50.0, "p90": 90.0, "p95": 95.0, "p99": 99.0}


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile (the common 'type 7' definition)."""
    if not values:
        raise ValueError("percentile of empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def aggregate(values: Sequence[float], op: str) -> float:
    """Apply a named aggregation to a list of numbers."""
    if op == "count":
        return float(len(values))
    if not values:
        raise ValueError(f"cannot compute {op!r} of empty sequence")
    if op == "sum":
        return float(sum(values))
    if op == "mean":
        return float(statistics.fmean(values))
    if op == "min":
        return float(min(values))
    if op == "max":
        return float(max(values))
    if op == "std":
        return float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    if op in _PERCENTILE_OPS:
        return percentile(values, _PERCENTILE_OPS[op])
    raise ValueError(f"unknown aggregation: {op!r}")


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation, or ``None`` when it is undefined."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return numerator / (denom_x * denom_y)
