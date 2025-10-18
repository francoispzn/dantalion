# Design

The full design document lives at [`DESIGN.md`](https://github.com/francoispzn/dantalion/blob/main/DESIGN.md)
in the repository root. This page is a quick orientation.

## The pieces

- **`providers`** — a narrow `Provider` protocol (`complete`/`stream`/
  `capabilities`) with one adapter per backend (Ollama, OpenAI-compatible,
  in-process llama.cpp, and a scripted mock), a model-name capability registry,
  and token estimation.
- **`structured`** — turning model text into validated pydantic objects, with a
  degradation ladder: schema-constrained → GBNF grammar → prompt → repair →
  reflection.
- **`tools`** — a `@tool` decorator that derives JSON Schema from a typed
  signature, validates arguments, and injects runtime dependencies.
- **`agent`** — the plan → act/observe → critique loop, bounded by a budget and
  cancellable between steps.
- **`memory`** — context-window arithmetic and transcript compaction.
- **`trace`** — spans, a usage/cost meter, and deterministic record/replay.
- **`domains/anomaly`** — the reference domain: dataset loader, read-only analysis
  tools, the `Investigation` report, and `investigate()`.
- **`eval`** — seeded synthetic incidents, scoring, and a runner.

## The one big idea

Local models vary enormously in capability. Two abstractions absorb that:
capability detection (every provider declares what it can do) and a structured
output ladder (the same pydantic type is produced by whichever mechanism the
model supports). See [ADR 0003](adr/0003-structured-output-ladder.md).
