"""Budgets and cancellation: the agent's guard rails.

An autonomous loop that calls a model in a circle needs hard limits, or a
confused model will happily burn tokens and wall-clock forever. A :class:`Budget`
caps steps, tokens, and time; a :class:`CancellationToken` lets a caller pull the
plug between steps. Both are deliberately cooperative — checked at safe points in
the loop — so a run always stops with a coherent partial result rather than a
half-written one.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from dantalion.errors import BudgetExceeded, Cancelled
from dantalion.types import Usage


@dataclass
class Budget:
    """Caps on how much work a single run may consume."""

    max_steps: int = 8
    max_tokens: int | None = None
    max_seconds: float | None = None
    clock: Callable[[], float] = time.monotonic

    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    _started: float | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        self._started = self.clock()

    def add_usage(self, usage: Usage) -> None:
        self.usage += usage

    def tick(self) -> None:
        self.steps += 1

    def elapsed(self) -> float:
        return 0.0 if self._started is None else self.clock() - self._started

    def exceeded(self) -> str | None:
        """Return the name of the first breached limit, or ``None``."""
        if self.steps >= self.max_steps:
            return "max_steps"
        if self.max_tokens is not None and self.usage.total_tokens >= self.max_tokens:
            return "token_budget"
        if self.max_seconds is not None and self.elapsed() >= self.max_seconds:
            return "time_budget"
        return None

    def require(self) -> None:
        """Raise if any limit is breached (for hard enforcement)."""
        reason = self.exceeded()
        if reason is not None:
            raise BudgetExceeded(reason)


@dataclass
class CancellationToken:
    """A one-way flag a caller flips to ask a run to stop."""

    _cancelled: bool = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True

    def require(self) -> None:
        if self._cancelled:
            raise Cancelled("run cancelled")
