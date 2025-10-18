# Dantalion

A model-agnostic, local-first autonomous **investigation agent**.

Point it at local logs and metrics, hand it an alert, and it plans an
investigation, gathers evidence with read-only tools, forms and revises
hypotheses, and writes a structured incident report — against whatever model you
happen to be running locally.

```bash
uv sync --extra dev
uv run dantalion investigate examples/incident.jsonl --model ollama:llama3.1
```

## Why

Most "agent" demos assume a frontier hosted model with reliable tool-calling and
a JSON mode. Local models are not that — they disagree about tool-calling, about
JSON, about context windows. The interesting engineering problem, and the point
of this project, is a genuinely useful multi-step agent that **degrades
gracefully** across that messy reality instead of falling over.

No model is required to develop or test: a scripted provider and recorded
cassettes make the whole suite deterministic and offline.

## Runs against

| spec | backend |
|------|---------|
| `ollama:llama3.1` | local [Ollama](https://ollama.com) daemon |
| `openai:qwen2.5` | any OpenAI-compatible server (vLLM, LM Studio, llama.cpp server, …) |
| `llama-cpp:/models/q4.gguf` | in-process `llama-cpp-python`, with grammar-constrained decoding |
| `replay:run.json` | a recorded cassette, replayed deterministically |

Set a default and connection details via environment (`DANTALION_*`, see
[`.env.example`](.env.example)).

## How it works

```
alert ─▶ domain pack (anomaly: tools + report schema)
           │
           ▼
        agent loop:  plan ─▶ act / observe ─▶ critique     (budgeted, cancellable)
           │                     │
           │                     ▼
           │              read-only tools over an in-memory dataset
           ▼
        structured output:  schema ▶ grammar ▶ prompt ▶ repair ▶ reflection
           │
           ▼
        Investigation report  (ranked hypotheses, evidence, root cause)
```

A pydantic model is the single source of truth for every schema; the agent picks
the strongest output mechanism the model supports and falls back step by step.
See [`DESIGN.md`](DESIGN.md) and the [ADRs](docs/adr) for the reasoning.

## Commands

```bash
uv run dantalion investigate <dataset> --model <spec> --alert "<text>"
uv run dantalion investigate <dataset> --model <spec> --record run.json   # capture a cassette
uv run dantalion investigate <dataset> --model replay:run.json            # replay it, offline
uv run dantalion eval --model <spec>                                      # synthetic-incident suite
uv run dantalion models <spec>                                            # show resolved capabilities
```

`--json` on `investigate`/`eval` emits machine-readable output.

## Library

```python
from dantalion.config import make_provider
from dantalion.domains.anomaly import Dataset, investigate

provider = make_provider("ollama:llama3.1")
result = investigate(provider, Dataset.load("examples/incident.jsonl"),
                     "latency spiked around 00:05")
print(result.report.root_cause)
for h in result.report.hypotheses:
    print(f"{h.confidence:.2f}  {h.statement}")
```

## Development

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
```

`mypy --strict` clean, property-tested where it counts, and green without any
model present. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
