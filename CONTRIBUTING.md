# Contributing

Thanks for taking a look. This is a small, opinionated codebase; the conventions
below keep it that way.

## Setup

```bash
uv sync --extra dev
uv run pre-commit install
```

A model is **not** required to develop or test — the suite runs against a scripted
provider and recorded cassettes.

## The checks

Everything CI runs, you can run locally:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

All four must pass. `mypy` runs in `--strict` mode; new code is fully typed.

## Conventions

- **Types are the source of truth.** Tool schemas, report schemas, and grammars
  are derived from pydantic models — don't hand-write a schema next to a model.
- **Tools are read-only.** Domain tools must not mutate state.
- **Adapters stay thin.** Provider-specific code lives in its adapter and
  translates to the neutral types in `dantalion.types`; nothing else learns the
  wire format.
- **Test without a model.** Use `MockProvider` or a cassette. If you change a
  prompt, re-record affected cassettes.
- **Keep commits small** and the message a short, plain statement of what changed.

## Adding a provider

Implement the `Provider` protocol (`complete`, `stream`, `capabilities`), translate
to/from `dantalion.types`, and register the `kind` in `dantalion.providers`. Add a
test that drives it through `httpx.MockTransport` or an injected client.

## Adding a domain

A domain is a `ToolRegistry` of read-only tools plus a pydantic report schema and
an entry point that wires them onto `Agent`. See `dantalion.domains.anomaly`.
