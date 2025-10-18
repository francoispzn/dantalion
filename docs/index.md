# Dantalion

A model-agnostic, local-first autonomous **investigation agent**.

Point it at local logs and metrics with an alert; it plans an investigation,
gathers evidence with read-only tools, forms and revises hypotheses, and writes a
typed incident report — against whatever model you run locally.

```bash
uv sync --extra dev
uv run dantalion investigate examples/incident.jsonl --model ollama:llama3.1
```

- **[Design](design.md)** — the architecture and the constraints behind it.
- **Decisions** — the architecture decision records.

The interesting engineering is the model-agnosticism: local models disagree about
tool-calling, JSON modes, and context sizes, and the job is to make a genuinely
useful multi-step agent that degrades gracefully across all of it.
