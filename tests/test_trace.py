from __future__ import annotations

import json
from pathlib import Path

import pytest

from dantalion.agent import Agent
from dantalion.errors import ReplayMismatch
from dantalion.providers.mock import MockProvider, text_response, tool_call_response
from dantalion.tools import ToolRegistry, tool
from dantalion.trace import (
    Cassette,
    Price,
    RecordingProvider,
    ReplayProvider,
    Tracer,
    UsageMeter,
)
from dantalion.types import CompletionRequest, CompletionResponse, Message, Usage


@tool
def noop() -> str:
    """Do nothing."""
    return "ok"


def _registry() -> ToolRegistry:
    return ToolRegistry([noop])


def _script() -> list[str | CompletionResponse]:
    return [tool_call_response("noop", {}, call_id="c1"), text_response("the answer")]


# -- tracer --------------------------------------------------------------


def test_tracer_records_nested_spans_with_durations() -> None:
    ticks = iter([0.0, 1.0, 2.0, 3.0])
    tracer = Tracer(clock=lambda: next(ticks))
    with tracer.span("run", task="x"), tracer.span("step"):
        pass

    assert [s.name for s in tracer.spans] == ["run", "step"]
    assert tracer.spans[0].parent is None
    assert tracer.spans[1].parent == 0
    assert tracer.spans[0].duration == 3.0
    assert tracer.spans[1].duration == 1.0
    assert tracer.spans[0].attributes == {"task": "x"}


def test_tracer_jsonl_round_trips(tmp_path: Path) -> None:
    tracer = Tracer(clock=iter([0.0, 1.0]).__next__)
    with tracer.span("only"):
        pass
    path = tmp_path / "trace.jsonl"
    tracer.write(path)
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert records[0]["name"] == "only"
    assert records[0]["duration"] == 1.0


# -- meter ---------------------------------------------------------------


def test_usage_meter_totals_and_per_model() -> None:
    meter = UsageMeter()
    meter.record("a", Usage(prompt_tokens=10, completion_tokens=5))
    meter.record("a", Usage(prompt_tokens=1, completion_tokens=1))
    meter.record("b", Usage(prompt_tokens=2, completion_tokens=0))

    assert meter.usage.total_tokens == 19
    assert meter.by_model["a"].total_tokens == 17
    assert meter.cost() == 0.0  # no prices => local-free


def test_usage_meter_applies_prices() -> None:
    meter = UsageMeter(prices={"a": Price(prompt_per_1k=1.0, completion_per_1k=2.0)})
    meter.record("a", Usage(prompt_tokens=1000, completion_tokens=1000))
    assert meter.cost() == pytest.approx(3.0)


# -- cassette ------------------------------------------------------------


def test_recording_then_replay_reproduces_a_run() -> None:
    recorder = RecordingProvider(MockProvider(_script()))
    first = Agent(recorder, _registry()).run("go")

    replay = ReplayProvider(recorder.cassette)
    second = Agent(replay, _registry()).run("go")

    assert first.output == "the answer"
    assert second.output == "the answer"
    assert len(recorder.cassette.interactions) == 2


def test_cassette_survives_disk_round_trip(tmp_path: Path) -> None:
    recorder = RecordingProvider(MockProvider(_script()))
    Agent(recorder, _registry()).run("go")
    path = tmp_path / "run.json"
    recorder.cassette.save(path)

    loaded = Cassette.load(path)
    replay = ReplayProvider(loaded)
    result = Agent(replay, _registry()).run("go")
    assert result.output == "the answer"


def test_replay_detects_prompt_drift() -> None:
    recorder = RecordingProvider(MockProvider(_script()))
    Agent(recorder, _registry()).run("go")

    replay = ReplayProvider(recorder.cassette)
    with pytest.raises(ReplayMismatch):
        Agent(replay, _registry()).run("a different task")


def test_replay_reports_exhaustion() -> None:
    request = CompletionRequest(messages=[Message.user("hi")])
    recorder = RecordingProvider(MockProvider([text_response("done")]))
    recorder.complete(request)  # record exactly one interaction for this request

    replay = ReplayProvider(recorder.cassette)
    assert replay.complete(request).message.content == "done"
    with pytest.raises(ReplayMismatch):
        replay.complete(request)  # nothing left on the cassette
