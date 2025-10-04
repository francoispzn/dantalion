from __future__ import annotations

from dantalion.agent import RunResult
from dantalion.domains.anomaly.investigator import InvestigationResult
from dantalion.domains.anomaly.report import Hypothesis, Investigation
from dantalion.eval import (
    default_scenarios,
    disk_exhaustion,
    provider_solver,
    run_eval,
    score_scenario,
)
from dantalion.providers.mock import MockProvider, text_response, tool_call_response
from dantalion.types import Usage


def _result(root_cause: str) -> InvestigationResult:
    report = Investigation(
        summary="",
        hypotheses=[Hypothesis(statement="x", confidence=0.5)],
        root_cause=root_cause,
    )
    run = RunResult(output=None, messages=[])
    return InvestigationResult(
        report=report,
        run=run,
        usage=Usage(prompt_tokens=10, completion_tokens=5),
        report_strategy="schema",
    )


def test_scenarios_are_deterministic() -> None:
    first = disk_exhaustion(0).dataset.numeric_series("latency_ms", "value")
    second = disk_exhaustion(0).dataset.numeric_series("latency_ms", "value")
    assert first == second


def test_default_suite_is_well_formed() -> None:
    scenarios = default_scenarios()
    assert {s.name for s in scenarios} == {"disk_exhaustion", "memory_leak", "deploy_regression"}
    for scenario in scenarios:
        assert len(scenario.dataset) > 0
        assert scenario.gold_keywords
        for stream in scenario.relevant_streams:
            assert stream in scenario.dataset.streams()


def test_score_detects_a_hit_and_a_miss() -> None:
    scenario = disk_exhaustion()
    assert score_scenario(scenario, _result("the disk ran out of space")).root_cause_hit is True
    assert score_scenario(scenario, _result("a network blip")).root_cause_hit is False


def test_run_eval_aggregates_scores() -> None:
    scenarios = default_scenarios()
    report = run_eval(scenarios, lambda scenario: _result(scenario.gold_keywords[0]))
    assert report.accuracy == 1.0
    assert report.summary()["scenarios"] == 3

    misses = run_eval(scenarios, lambda _scenario: _result("totally unrelated"))
    assert misses.accuracy == 0.0


def test_provider_solver_runs_the_real_agent() -> None:
    provider = MockProvider(
        [
            '{"steps": ["inspect", "conclude"]}',
            tool_call_response("list_streams", {}, call_id="c1"),
            text_response("the disk ran out of space"),
            '{"sufficient": true, "reasoning": "supported"}',
            '{"summary": "disk filled", "hypotheses": [{"statement": "disk exhaustion",'
            ' "confidence": 0.9, "evidence": []}], "root_cause": "disk exhaustion",'
            ' "recommended_actions": [], "open_questions": []}',
        ]
    )
    report = run_eval([disk_exhaustion()], provider_solver(provider, plan=True, review=True))
    assert report.accuracy == 1.0
    assert report.scores[0].report_strategy == "schema"
