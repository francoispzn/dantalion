from __future__ import annotations

from pathlib import Path

import pytest

from dantalion.domains.anomaly.data import Dataset

EXAMPLE_INCIDENT = Path(__file__).resolve().parents[1] / "examples" / "incident.jsonl"


@pytest.fixture
def incident_dataset() -> Dataset:
    return Dataset.load(EXAMPLE_INCIDENT)
