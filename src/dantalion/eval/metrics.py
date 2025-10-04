"""Scoring an investigation against a known answer.

Because the synthetic scenarios are generated with a planted root cause, every
run can be graded automatically. The metrics are deliberately blunt and
interpretable: did the report land on the right cause, how much of the relevant
data did it actually look at, and what did it cost in steps and tokens. Blunt is
the point — these are regression signals, not a leaderboard.
"""

from __future__ import annotations

from dataclasses import dataclass

from dantalion.domains.anomaly.investigator import InvestigationResult
from dantalion.eval.scenarios import Scenario


@dataclass(frozen=True)
class ScenarioScore:
    """How one investigation did against its planted answer."""

    name: str
    root_cause_hit: bool
    relevant_tool_use: float
    steps: int
    tokens: int
    report_strategy: str


@dataclass
class EvalReport:
    """Aggregated scores across a set of scenarios."""

    scores: list[ScenarioScore]

    @property
    def accuracy(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for score in self.scores if score.root_cause_hit) / len(self.scores)

    @property
    def mean_relevant_tool_use(self) -> float:
        if not self.scores:
            return 0.0
        return sum(score.relevant_tool_use for score in self.scores) / len(self.scores)

    @property
    def total_tokens(self) -> int:
        return sum(score.tokens for score in self.scores)

    def summary(self) -> dict[str, float | int]:
        return {
            "scenarios": len(self.scores),
            "accuracy": round(self.accuracy, 3),
            "mean_relevant_tool_use": round(self.mean_relevant_tool_use, 3),
            "total_tokens": self.total_tokens,
        }


def score_scenario(scenario: Scenario, result: InvestigationResult) -> ScenarioScore:
    """Grade one investigation result against its scenario's gold answer."""
    haystack = " ".join(
        [
            result.report.root_cause,
            result.report.summary,
            *(hypothesis.statement for hypothesis in result.report.hypotheses),
        ]
    ).lower()
    hit = any(keyword.lower() in haystack for keyword in scenario.gold_keywords)

    referenced = _referenced_streams(result)
    relevant = scenario.relevant_streams
    coverage = (
        sum(1 for stream in relevant if stream in referenced) / len(relevant) if relevant else 0.0
    )

    return ScenarioScore(
        name=scenario.name,
        root_cause_hit=hit,
        relevant_tool_use=coverage,
        steps=len(result.run.steps),
        tokens=result.usage.total_tokens,
        report_strategy=result.report_strategy,
    )


def _referenced_streams(result: InvestigationResult) -> set[str]:
    referenced: set[str] = set()
    for call in result.run.tool_calls:
        for value in call.arguments.values():
            if isinstance(value, str):
                referenced.add(value)
    return referenced
