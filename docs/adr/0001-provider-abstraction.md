# 1. A narrow provider protocol

- Status: accepted
- Date: 2024 H2

## Context

The project must run against many local backends (Ollama, llama.cpp, vLLM, LM
Studio, …). If the agent code knew about any of them, "model agnostic" would be a
slogan, not a property.

## Decision

Everything the agent needs from a model is expressed by one small protocol:
`complete`, `stream`, and `capabilities`. Each backend is a thin adapter that
translates to and from a set of provider-neutral message types. Selecting a
backend is a `kind:model` string resolved in one place.

## Consequences

- Adding a backend is a self-contained adapter with no changes elsewhere.
- The agent adapts to declared `capabilities()` rather than hard-coding what a
  model can do.
- Streaming is text-only by design; tool calls and structured output need the
  whole payload, so they use `complete`.
