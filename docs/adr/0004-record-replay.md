# 4. Determinism via record/replay

- Status: accepted
- Date: 2024 H2

## Context

An agent that loops over a model is non-deterministic and, in CI, would need a
model present. Mocking every interaction by hand is brittle and drifts from
reality.

## Decision

A `RecordingProvider` wraps any real provider and captures every request/response
pair to a cassette. A `ReplayProvider` serves those responses back, fingerprinting
each incoming request against what was recorded. Replay is strict by default, so a
prompt change is caught as drift instead of silently replaying stale answers.

## Consequences

- CI re-runs real investigations with no model present, fast and offline.
- A captured failure can be debugged byte-for-byte.
- The cassette is also the unit of the `MockProvider` test double, so the same
  idea powers both tests and reproducible runs.
- Cassettes must be re-recorded when prompts intentionally change — by design.
