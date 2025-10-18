# Dantalion — design

This document explains what Dantalion is, the constraints that shaped it, and how
the pieces fit together. It is the companion to the code: where the docstrings
explain a module, this explains the *system*.

## 1. What it is

Dantalion is a local-first, model-agnostic autonomous **investigation agent**. Its
reference application is anomaly investigation: given local logs/metrics and an
alert, it plans an investigation, gathers evidence with read-only tools, forms
and revises hypotheses, and produces a typed incident report.

The agent **core is domain-agnostic**. Anomaly investigation is one domain pack
(tools + a report schema) bolted onto a reusable loop; research synthesis or code
review would be other packs over the same core.

## 2. Goals and non-goals

**Goals**

- Run well against *varied, locally-runnable* models — Ollama, llama.cpp, vLLM,
  LM Studio, any OpenAI-compatible server.
- Degrade gracefully as model capability drops (tool-calling → JSON mode →
  grammar → prompt-and-repair).
- Be a real multi-step agent: plan, act, observe, self-critique — not a single
  prompt in a trench coat.
- Be observable and reproducible: structured traces, token accounting, and
  deterministic record/replay.
- Be testable without a model in the loop, so CI is fast and offline.

**Non-goals (v1)**

- Distributed execution, a web UI, fine-tuning, multi-agent swarms.
- A heavyweight data stack: the anomaly tools are hand-rolled over stdlib rather
  than pulling in pandas/numpy.

## 3. The hard problem: model heterogeneity

The defining constraint is that local models disagree about almost everything.
Some call tools natively; some have never seen a tool. Some honour a JSON schema;
some emit prose with a JSON object somewhere inside. Context windows range from
8k to 128k.

Two abstractions absorb this:

1. **Capability detection** (`providers`). Every provider answers
   `capabilities()` — can it call tools, constrain to a JSON schema, constrain to
   a grammar; how big is its context window. Decisions are made against declared
   capability, not hope.
2. **A structured-output ladder** (`structured`). The same pydantic type is
   produced by whichever rung the model supports, falling back step by step:

   | rung | mechanism | when |
   |------|-----------|------|
   | schema | server-side JSON-schema constrained decoding | `json_schema` |
   | grammar | GBNF compiled from the schema (llama.cpp) | `grammar` |
   | prompt | describe the schema, ask for JSON | otherwise |
   | repair | extract + fix fences/commas/truncation | any output |
   | reflection | show the model its validation error, retry | on failure |

## 4. Architecture

```
            ┌─────────────┐
   alert ──▶│   domain    │  anomaly: tools + Investigation schema
            │    pack     │
            └──────┬──────┘
                   │ registry + context (the Dataset)
            ┌──────▼──────┐     ┌──────────────┐
            │    agent    │────▶│  structured  │  schema/grammar/prompt/repair
            │ plan/act/   │     └──────┬───────┘
            │  critique   │            │
            └──────┬──────┘     ┌──────▼───────┐
                   │            │  providers   │  ollama / openai / llama.cpp / mock
            ┌──────▼──────┐     └──────────────┘
            │   memory    │  context compaction
            └─────────────┘
   cross-cutting: trace (spans, usage, record/replay) · config · cli
```

- **`providers`** — the `Provider` protocol (`complete`/`stream`/`capabilities`),
  one adapter per backend, a model-name capability registry, and token
  estimation for when servers don't report usage.
- **`structured`** — the ladder above, plus the JSON extractor/repairer and the
  JSON-Schema→GBNF compiler.
- **`tools`** — a `@tool` decorator that derives JSON Schema from a typed
  signature, validates arguments with pydantic, and supports dependency
  injection so stateful domains stay clean.
- **`agent`** — the plan → act/observe → critique loop, bounded by a `Budget`
  and cancellable between steps.
- **`memory`** — context-window arithmetic and a compactor that condenses the
  middle of a long transcript while preserving the task and recent evidence.
- **`trace`** — nested timed spans, a usage/cost meter, and cassette
  record/replay that makes a run reproducible.
- **`domains/anomaly`** — the reference pack: a dataset loader, ten read-only
  analysis tools, the `Investigation` report, and the `investigate()` entry point.
- **`eval`** — seeded synthetic incidents with planted root causes, scoring, and
  a runner that doubles as a regression test.
- **`config` / `cli`** — `kind:model` spec resolution and a Typer CLI.

## 5. Key decisions

- **Synchronous core.** An investigation is an inherently sequential
  plan/act/observe loop. A blocking interface keeps traces deterministic and
  tests free of an event loop; concurrency, where it would help, can live above
  this layer. (ADR 0002)
- **pydantic as the single source of truth.** Tool schemas, report schemas, and
  grammars are all derived from typed Python. There is no second place for a
  schema to drift. (ADR 0003)
- **Record/replay over mocking the world.** Determinism comes from replaying real
  recorded model I/O, fingerprinted to catch prompt drift, rather than from a web
  of mocks. (ADR 0004)
- **Read-only tools.** The domain tools cannot mutate anything, so an autonomous
  loop is safe by construction.

## 6. How it evolved

The codebase grew in the order a tool like this naturally matures, and the commit
history reflects that progression:

1. provider protocol + the first adapter
2. a second adapter + capability detection
3. the typed tool framework
4. a basic act/observe loop
5. structured output with repair/reflection
6. the GBNF grammar path
7. planner/executor/critic with budgets and cancellation
8. working memory and context compaction
9. tracing, usage accounting, record/replay
10. the anomaly domain pack
11. the evaluation harness
12. config, CLI, docs, CI

## 7. Testing & verification

- `ruff` + `mypy --strict` clean; `pytest` green with coverage.
- Property-based tests for JSON extraction/repair.
- A deterministic end-to-end path: `MockProvider`/replay drives the full agent on
  a synthetic incident and asserts it reaches the planted root cause.
- `dantalion eval` produces a metrics report that CI can regression-check.
- A live smoke test against a real local model is documented but skipped in CI.
