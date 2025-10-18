# 2. A synchronous core

- Status: accepted
- Date: 2024 H2

## Context

Agent frameworks often default to async. It buys concurrency but costs an event
loop in every test and a more complex tracing story.

## Decision

The core is synchronous. An investigation is a sequential plan → act → observe
loop where each step depends on the last, so there is little concurrency to win
inside one run. HTTP adapters use a blocking client; the in-process llama.cpp
backend is blocking anyway.

## Consequences

- Tests are plain functions; traces are linear and deterministic.
- Where concurrency genuinely helps (e.g. running an eval suite over many
  scenarios), it can be added *around* the core without changing it.
- If a future need demands streaming tool-calls or parallel sub-agents, this is
  the decision to revisit.
