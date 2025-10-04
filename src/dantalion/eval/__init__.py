"""Evaluation: synthetic incidents, scoring, and a regression-friendly runner."""

from __future__ import annotations

from dantalion.eval.metrics import EvalReport, ScenarioScore, score_scenario
from dantalion.eval.runner import Solver, provider_solver, run_eval
from dantalion.eval.scenarios import (
    Scenario,
    default_scenarios,
    deploy_regression,
    disk_exhaustion,
    memory_leak,
)

__all__ = [
    "EvalReport",
    "Scenario",
    "ScenarioScore",
    "Solver",
    "default_scenarios",
    "deploy_regression",
    "disk_exhaustion",
    "memory_leak",
    "provider_solver",
    "run_eval",
    "score_scenario",
]
