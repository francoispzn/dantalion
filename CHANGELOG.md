# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/).

## 0.1.0

First public release.

### Added

- Model-agnostic provider layer with capability detection: Ollama,
  OpenAI-compatible servers, in-process llama.cpp, and a scripted test double.
- Structured-output layer with a degradation ladder — JSON-schema decoding, a
  JSON-Schema→GBNF compiler for grammar-constrained decoding, and a
  prompt/extract/repair/reflection fallback.
- Typed tool framework: schema derived from typed signatures, argument
  validation, and dependency injection.
- Agent loop with planning, self-critique, token/time/step budgets, cancellation,
  and context compaction.
- Observability: structured trace spans, a usage/cost meter, and deterministic
  record/replay cassettes.
- Anomaly-investigation domain pack: dataset loader (CSV/JSONL), ten read-only
  analysis tools, and a typed incident report.
- Evaluation harness with seeded synthetic incidents and scoring.
- Typer CLI: `investigate`, `eval`, `models`, plus `replay:` provider specs.
