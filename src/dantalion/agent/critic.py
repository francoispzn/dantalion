"""The critic: a skeptical second opinion before an answer is final.

Left alone, a model tends to declare victory early. The critic interrupts that:
it is shown the task, the evidence actually gathered, and the proposed answer,
and asked whether the conclusion is genuinely supported. If not, its guidance is
fed back and the executor keeps going. This self-review loop is bounded by the
agent's ``max_reviews`` so it cannot argue with itself forever.
"""

from __future__ import annotations

from collections.abc import Sequence

from dantalion.agent.result import Critique
from dantalion.providers.base import Provider
from dantalion.structured import StructuredResult, structure
from dantalion.types import Message, Role

_CRITIC_SYSTEM = (
    "You are a skeptical reviewer. Decide whether the proposed answer is "
    "genuinely supported by the evidence gathered so far. Set sufficient=false "
    "if the conclusion outruns the evidence, and put one concrete next step in "
    "guidance. Be strict but fair."
)


def review_answer(
    provider: Provider,
    task: str,
    transcript: Sequence[Message],
    answer: str,
    *,
    temperature: float = 0.0,
) -> StructuredResult[Critique]:
    """Judge whether ``answer`` is supported by the evidence in ``transcript``."""
    evidence = _summarise_evidence(transcript)
    messages = [
        Message.system(_CRITIC_SYSTEM),
        Message.user(
            f"Task: {task}\n\n"
            f"Evidence gathered:\n{evidence or '(no tool evidence was gathered)'}\n\n"
            f"Proposed answer:\n{answer}"
        ),
    ]
    return structure(provider, messages, Critique, temperature=temperature)


def _summarise_evidence(transcript: Sequence[Message]) -> str:
    lines = [
        f"- {message.name}: {message.content}"
        for message in transcript
        if message.role is Role.TOOL and message.content
    ]
    return "\n".join(lines)
