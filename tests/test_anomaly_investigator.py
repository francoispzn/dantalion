from __future__ import annotations

from dantalion.domains.anomaly import Dataset, investigate
from dantalion.domains.anomaly.tools import ANOMALY_TOOLS
from dantalion.providers.mock import MockProvider, text_response, tool_call_response


def test_investigate_drives_tools_and_returns_a_typed_report(incident_dataset: Dataset) -> None:
    provider = MockProvider(
        [
            # planner
            '{"steps": ["list streams", "aggregate latency", "correlate with disk"]}',
            # executor: inspect, then correlate, then conclude
            tool_call_response("list_streams", {}, call_id="c1"),
            tool_call_response(
                "correlate",
                {
                    "stream_a": "latency_ms",
                    "field_a": "value",
                    "stream_b": "disk_used_pct",
                    "field_b": "value",
                },
                call_id="c2",
            ),
            text_response("Latency rose as the disk filled; the disk ran out of space."),
            # critic
            '{"sufficient": true, "reasoning": "evidence supports it"}',
            # final structured report
            '{"summary": "disk filled and latency spiked",'
            ' "timeline": ["00:05 disk at 92%"],'
            ' "hypotheses": [{"statement": "disk exhaustion", "confidence": 0.9,'
            ' "evidence": ["latency/disk correlation"]}],'
            ' "root_cause": "the disk ran out of space",'
            ' "recommended_actions": ["add disk capacity"], "open_questions": []}',
        ]
    )

    result = investigate(provider, incident_dataset, "latency spiked around 00:05")

    assert result.report.root_cause == "the disk ran out of space"
    top = result.report.top_hypothesis()
    assert top is not None
    assert top.confidence == 0.9
    assert result.run.plan is not None
    assert len(result.run.tool_calls) == 2
    assert result.usage.total_tokens >= 0


def test_build_registry_has_every_tool() -> None:
    from dantalion.domains.anomaly import build_registry

    registry = build_registry()
    assert len(registry) == len(ANOMALY_TOOLS)
    assert "detect_spikes" in registry
