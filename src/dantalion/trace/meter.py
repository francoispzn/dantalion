"""Token and cost accounting.

Local inference is "free" in dollars, but tokens are still the currency of the
context window and of latency, so they are worth counting. The meter accumulates
usage overall and per model, and — for the case where a run mixes local and paid
models — can attach a price sheet to turn tokens into an estimated cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dantalion.types import Usage


@dataclass(frozen=True)
class Price:
    """Per-1000-token prices for one model. Defaults to free."""

    prompt_per_1k: float = 0.0
    completion_per_1k: float = 0.0


@dataclass
class UsageMeter:
    """Running totals of token usage, with optional pricing."""

    usage: Usage = field(default_factory=Usage)
    by_model: dict[str, Usage] = field(default_factory=dict)
    prices: dict[str, Price] = field(default_factory=dict)

    def record(self, model: str, usage: Usage) -> None:
        self.usage += usage
        self.by_model[model] = self.by_model.get(model, Usage()) + usage

    def cost(self) -> float:
        total = 0.0
        for model, usage in self.by_model.items():
            price = self.prices.get(model)
            if price is None:
                continue
            total += usage.prompt_tokens / 1000 * price.prompt_per_1k
            total += usage.completion_tokens / 1000 * price.completion_per_1k
        return total
