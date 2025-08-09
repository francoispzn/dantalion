from __future__ import annotations

import pytest

from dantalion.agent.budget import Budget, CancellationToken
from dantalion.errors import BudgetExceeded, Cancelled
from dantalion.types import Usage


def test_fresh_budget_is_not_exceeded() -> None:
    assert Budget(max_steps=3).exceeded() is None


def test_step_budget_trips() -> None:
    budget = Budget(max_steps=2)
    budget.tick()
    assert budget.exceeded() is None
    budget.tick()
    assert budget.exceeded() == "max_steps"


def test_token_budget_trips() -> None:
    budget = Budget(max_steps=99, max_tokens=100)
    budget.add_usage(Usage(prompt_tokens=60, completion_tokens=50))
    assert budget.exceeded() == "token_budget"


def test_time_budget_within_deadline() -> None:
    ticks = iter([0.0, 1.0])  # start, elapsed below the cap
    budget = Budget(max_steps=99, max_seconds=3.0, clock=lambda: next(ticks))
    budget.start()
    assert budget.exceeded() is None


def test_time_budget_past_deadline() -> None:
    ticks = iter([0.0, 5.0])  # start, elapsed over the cap
    budget = Budget(max_steps=99, max_seconds=3.0, clock=lambda: next(ticks))
    budget.start()
    assert budget.exceeded() == "time_budget"


def test_require_raises_on_breach() -> None:
    budget = Budget(max_steps=1)
    budget.tick()
    with pytest.raises(BudgetExceeded):
        budget.require()


def test_cancellation_token_flow() -> None:
    token = CancellationToken()
    before = token.cancelled
    token.require()  # no-op while not cancelled
    token.cancel()
    after = token.cancelled
    assert before is False
    assert after is True
    with pytest.raises(Cancelled):
        token.require()
