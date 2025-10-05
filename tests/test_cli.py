from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dantalion import __version__
from dantalion.cli.app import app
from dantalion.domains.anomaly import Dataset, investigate
from dantalion.providers.mock import MockProvider, text_response, tool_call_response
from dantalion.trace import RecordingProvider

runner = CliRunner()
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "incident.jsonl"
ALERT = "latency spiked around 00:05"


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_models_reports_capabilities() -> None:
    result = runner.invoke(app, ["models", "mock:demo"])
    assert result.exit_code == 0
    caps = json.loads(result.stdout)
    assert caps["grammar"] is True
    assert caps["json_schema"] is True


def _record_cassette(path: Path) -> None:
    provider = RecordingProvider(
        MockProvider(
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
    )
    investigate(provider, Dataset.load(EXAMPLE), ALERT)
    provider.cassette.save(path)


def test_investigate_replays_a_cassette_as_json(tmp_path: Path) -> None:
    cassette = tmp_path / "run.json"
    _record_cassette(cassette)

    result = runner.invoke(
        app,
        ["investigate", str(EXAMPLE), "--model", f"replay:{cassette}", "--alert", ALERT, "--json"],
    )
    assert result.exit_code == 0, result.stdout
    report = json.loads(result.stdout)
    assert "disk" in report["root_cause"].lower()


def test_investigate_renders_human_output(tmp_path: Path) -> None:
    cassette = tmp_path / "run.json"
    _record_cassette(cassette)

    result = runner.invoke(
        app,
        ["investigate", str(EXAMPLE), "--model", f"replay:{cassette}", "--alert", ALERT],
    )
    assert result.exit_code == 0, result.stdout
    assert "root cause" in result.stdout.lower()
