"""Record and replay of model interactions.

A cassette is a recording of every request the agent made and the response it got
back. Recording one during a live run, then replaying it, turns an inherently
non-deterministic agent into something reproducible: CI can re-run a real
investigation with no model present, and a captured failure can be debugged
offline, byte for byte.

Replay is strict by default — each incoming request is fingerprinted and checked
against what was recorded — so a code change that alters the prompt is caught
immediately as drift rather than silently replaying stale answers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dantalion.errors import ReplayMismatch
from dantalion.providers.base import Capabilities, Provider
from dantalion.types import (
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
)


def request_fingerprint(request: CompletionRequest) -> str:
    """A stable hash of a request, used to detect prompt drift on replay."""
    payload = json.dumps(request.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class Interaction:
    """One recorded request/response pair."""

    request: dict[str, Any]
    response: dict[str, Any]
    fingerprint: str


@dataclass
class Cassette:
    """An ordered recording of model interactions."""

    interactions: list[Interaction] = field(default_factory=list)
    capabilities: dict[str, Any] | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "capabilities": self.capabilities,
                "interactions": [
                    {
                        "request": item.request,
                        "response": item.response,
                        "fingerprint": item.fingerprint,
                    }
                    for item in self.interactions
                ],
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> Cassette:
        data = json.loads(text)
        interactions = [
            Interaction(
                request=item["request"],
                response=item["response"],
                fingerprint=item["fingerprint"],
            )
            for item in data.get("interactions", [])
        ]
        return cls(interactions=interactions, capabilities=data.get("capabilities"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Cassette:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


class RecordingProvider:
    """Wraps a provider and records every interaction onto a cassette."""

    name = "recording"

    def __init__(self, inner: Provider, cassette: Cassette | None = None) -> None:
        self.inner = inner
        self.model = inner.model
        self.cassette = cassette or Cassette()

    def capabilities(self) -> Capabilities:
        caps = self.inner.capabilities()
        self.cassette.capabilities = caps.model_dump()
        return caps

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        response = self.inner.complete(request)
        self.cassette.interactions.append(
            Interaction(
                request=request.model_dump(mode="json"),
                response=response.model_dump(mode="json"),
                fingerprint=request_fingerprint(request),
            )
        )
        return response

    def stream(self, request: CompletionRequest) -> Iterator[CompletionChunk]:
        yield from self.inner.stream(request)


class ReplayProvider:
    """Serves recorded responses, verifying each request matches the recording."""

    name = "replay"

    def __init__(
        self,
        cassette: Cassette,
        *,
        capabilities: Capabilities | None = None,
        strict: bool = True,
    ) -> None:
        self.cassette = cassette
        self._index = 0
        self._strict = strict
        self.model = _recorded_model(cassette)
        if capabilities is not None:
            self._capabilities = capabilities
        elif cassette.capabilities is not None:
            self._capabilities = Capabilities.model_validate(cassette.capabilities)
        else:
            self._capabilities = Capabilities()

    def capabilities(self) -> Capabilities:
        return self._capabilities

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        if self._index >= len(self.cassette.interactions):
            raise ReplayMismatch("cassette is exhausted; the run made more calls than recorded")
        interaction = self.cassette.interactions[self._index]
        position = self._index
        self._index += 1
        if self._strict and request_fingerprint(request) != interaction.fingerprint:
            raise ReplayMismatch(f"request at position {position} does not match the cassette")
        return CompletionResponse.model_validate(interaction.response)

    def stream(self, request: CompletionRequest) -> Iterator[CompletionChunk]:
        response = self.complete(request)
        yield CompletionChunk(
            delta=response.message.content or "",
            finish_reason=response.finish_reason,
            usage=response.usage,
        )


def _recorded_model(cassette: Cassette) -> str:
    if cassette.interactions:
        model = cassette.interactions[0].response.get("model")
        if isinstance(model, str):
            return model
    return "replay"


_record: type[Provider] = RecordingProvider
_replay: type[Provider] = ReplayProvider
