"""Anomaly investigation: the reference domain pack.

Point it at local logs/metrics and an alert; get back a typed incident report.
"""

from __future__ import annotations

from dantalion.domains.anomaly.data import Dataset, Event
from dantalion.domains.anomaly.investigator import (
    InvestigationResult,
    build_registry,
    investigate,
)
from dantalion.domains.anomaly.report import Hypothesis, Investigation
from dantalion.domains.anomaly.tools import ANOMALY_TOOLS

__all__ = [
    "ANOMALY_TOOLS",
    "Dataset",
    "Event",
    "Hypothesis",
    "Investigation",
    "InvestigationResult",
    "build_registry",
    "investigate",
]
