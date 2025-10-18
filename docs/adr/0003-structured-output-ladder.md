# 3. One schema, a ladder of enforcement

- Status: accepted
- Date: 2024 H2

## Context

The agent must extract typed objects (reports, plans, critiques) from models that
range from "has a JSON mode" to "has never emitted valid JSON on purpose". A
second, hand-written schema per model class would inevitably drift.

## Decision

A pydantic model is the single source of truth. From it we derive the JSON Schema
(for schema-constrained servers) and a GBNF grammar (for llama.cpp). At runtime
the strongest available mechanism is used, degrading in order: schema → grammar →
prompt → repair → reflection. The result records which rung it landed on.

## Consequences

- Adding or changing a field is a one-line edit to the model.
- Weak models still return valid objects via repair/reflection, at the cost of
  extra calls — which the recorded `strategy` makes visible in evals.
- The GBNF compiler supports the subset of JSON Schema pydantic emits; property
  order is fixed and all-optional objects are treated as all-required to keep the
  grammar finite.
