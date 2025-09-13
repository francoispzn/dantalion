"""Assembling the pieces into an end-to-end anomaly investigation.

This is where the generic agent core meets the anomaly domain: it wires the
read-only toolset, an investigator system prompt, optional planning and
self-review, and a final pass that turns everything the agent learned into a
typed :class:`Investigation`. Callers get one function — :func:`investigate` —
and a result that carries both the report and the full reasoning trail behind it.
"""

from __future__ import annotations

from dataclasses import dataclass

from dantalion.agent import Agent, RunResult
from dantalion.domains.anomaly.data import Dataset
from dantalion.domains.anomaly.report import Investigation
from dantalion.domains.anomaly.tools import ANOMALY_TOOLS
from dantalion.memory.compaction import ContextCompactor, structural_digest
from dantalion.providers.base import Provider
from dantalion.structured import structure
from dantalion.tools import ToolRegistry
from dantalion.types import Message, Usage

ANOMALY_SYSTEM = (
    "You are an on-call engineer investigating an anomaly in local logs and "
    "metrics. Use the tools to inspect the data: list the streams first, then "
    "slice time windows, aggregate, correlate, and search logs to build evidence. "
    "Do not speculate beyond what the tools show you. When the evidence points to "
    "a likely root cause, state it plainly with the facts that support it."
)

_REPORT_SYSTEM = (
    "You are writing the final incident report. Every claim must rest on the "
    "evidence gathered. Rank hypotheses by confidence between 0 and 1, cite the "
    "evidence for each, and name the single most likely root cause."
)


@dataclass
class InvestigationResult:
    """The report plus the trail that produced it."""

    report: Investigation
    run: RunResult
    usage: Usage
    report_strategy: str


def build_registry() -> ToolRegistry:
    """A fresh registry of the anomaly-investigation tools."""
    return ToolRegistry(list(ANOMALY_TOOLS))


def investigate(
    provider: Provider,
    dataset: Dataset,
    alert: str,
    *,
    max_steps: int = 12,
    plan: bool = True,
    review: bool = True,
    max_reviews: int = 1,
    temperature: float = 0.1,
    compactor: ContextCompactor | None = None,
) -> InvestigationResult:
    """Investigate ``alert`` against ``dataset`` and return a typed report."""
    agent = Agent(
        provider,
        build_registry(),
        context={"dataset": dataset},
        system_prompt=ANOMALY_SYSTEM,
        max_steps=max_steps,
        plan=plan,
        review=review,
        max_reviews=max_reviews,
        temperature=temperature,
        compactor=compactor,
    )
    run = agent.run(_task(alert))
    report = structure(provider, _report_prompt(alert, run), Investigation, temperature=0.0)
    return InvestigationResult(
        report=report.value,
        run=run,
        usage=run.usage + report.usage,
        report_strategy=report.strategy,
    )


def _task(alert: str) -> str:
    return (
        f"Investigate this alert against the available data and determine the root cause.\n\n"
        f"Alert: {alert}"
    )


def _report_prompt(alert: str, run: RunResult) -> list[Message]:
    evidence = structural_digest(run.messages) or "(no tool evidence was gathered)"
    conclusion = run.output or "(the investigation did not reach a conclusion)"
    return [
        Message.system(_REPORT_SYSTEM),
        Message.user(
            f"Alert: {alert}\n\n"
            f"Evidence gathered during the investigation:\n{evidence}\n\n"
            f"Investigator's conclusion:\n{conclusion}\n\n"
            "Write the incident report."
        ),
    ]
