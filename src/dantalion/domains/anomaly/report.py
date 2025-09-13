"""The structured deliverable of an investigation.

The whole point of the agent is to end at *this* — not a wall of prose, but a
typed incident report with ranked hypotheses, each tied to evidence and a
calibrated confidence. Because it is a pydantic model it drives the structured
output layer directly, so the same report shape is produced whether the model
has a JSON mode, a grammar, or neither.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Hypothesis(BaseModel):
    """A candidate explanation, ranked by how well the evidence supports it."""

    statement: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)


class Investigation(BaseModel):
    """A complete incident report."""

    summary: str
    timeline: list[str] = Field(default_factory=list)
    hypotheses: list[Hypothesis]
    root_cause: str
    recommended_actions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)

    def top_hypothesis(self) -> Hypothesis | None:
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.confidence)
