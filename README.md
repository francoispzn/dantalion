# Dantalion

A model-agnostic, local-first autonomous **investigation agent**.

Point it at local logs and metrics, hand it an alert, and it plans an
investigation, gathers evidence with read-only tools, forms and revises
hypotheses, and writes up a structured incident report — all against whatever
model you happen to be running locally.

> **Status:** early. The provider layer and core types are in place; the agent
> loop, structured-output layer, and the anomaly domain land in subsequent
> milestones. See `DESIGN.md` for where this is headed.

## Why

Most "agent" demos assume a frontier hosted model with reliable tool-calling and
a JSON mode. Local models are not that. They disagree about tool-calling, about
JSON, about context windows. The interesting engineering problem — and the point
of this project — is to make a genuinely useful multi-step agent that degrades
gracefully across that messy reality instead of falling over.

## Runs against

- **Ollama** — the reference backend.
- Any **OpenAI-compatible** server (vLLM, LM Studio, llama.cpp's server, …).
- **llama.cpp** in-process, with grammar-constrained decoding.

No model is required to develop or test: a scripted provider plus recorded
cassettes make the whole suite deterministic and offline.

## Quick start

```bash
uv sync --extra dev
uv run dantalion --help
```

Run an investigation against a local Ollama model:

```bash
ollama pull llama3.1
uv run dantalion investigate examples/incident.jsonl --model ollama:llama3.1
```

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

## License

Apache-2.0. See [LICENSE](LICENSE).
