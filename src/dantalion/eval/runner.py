"""Running a set of scenarios and collecting their scores.

The runner is parameterised by a *solver* — anything that turns a scenario into
an :class:`InvestigationResult`. The default solver is the real agent driven by a
provider, but tests pass a canned solver so the harness itself can be checked
without a model. This is the same separation the rest of the codebase uses:
behaviour is injected, never hard-wired.
"""

from __future__ import annotations

from collections.abc import Callable

from dantalion.domains.anomaly.investigator import InvestigationResult, investigate
from dantalion.eval.metrics import EvalReport, score_scenario
from dantalion.eval.scenarios import Scenario
from dantalion.providers.base import Provider

Solver = Callable[[Scenario], InvestigationResult]


def provider_solver(provider: Provider, **investigate_kwargs: object) -> Solver:
    """A solver that runs the real investigation agent against each scenario."""

    def solve(scenario: Scenario) -> InvestigationResult:
        return investigate(provider, scenario.dataset, scenario.alert, **investigate_kwargs)  # type: ignore[arg-type]

    return solve


def run_eval(scenarios: list[Scenario], solver: Solver) -> EvalReport:
    """Solve every scenario and score the results."""
    scores = [score_scenario(scenario, solver(scenario)) for scenario in scenarios]
    return EvalReport(scores=scores)
