from __future__ import annotations

import importlib.util
from typing import Any

import pytest
from pydantic import BaseModel

from dantalion.errors import ProviderError
from dantalion.providers import build_provider
from dantalion.providers.llama_cpp import LlamaCppProvider
from dantalion.structured import structure
from dantalion.types import CompletionRequest, FinishReason, Message, ResponseFormat


class FakeLlama:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.result


def _completion(content: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }


def test_capabilities_offer_grammar_not_schema() -> None:
    caps = LlamaCppProvider(
        "m.gguf", llama=FakeLlama(_completion("hi")), context_window=4096
    ).capabilities()
    assert caps.grammar is True
    assert caps.json_schema is False
    assert caps.context_window == 4096


def test_complete_parses_message_and_usage() -> None:
    provider = LlamaCppProvider("m.gguf", llama=FakeLlama(_completion("hello")))
    out = provider.complete(CompletionRequest(messages=[Message.user("x")]))
    assert out.message.content == "hello"
    assert out.usage.prompt_tokens == 3
    assert out.finish_reason is FinishReason.STOP


def test_grammar_is_passed_to_llama() -> None:
    fake = FakeLlama(_completion("ok"))
    provider = LlamaCppProvider("m.gguf", llama=fake, grammar_factory=lambda g: g)
    request = CompletionRequest(
        messages=[Message.user("x")],
        response_format=ResponseFormat(grammar="root ::= ws value ws"),
    )
    provider.complete(request)
    assert fake.calls[0]["grammar"] == "root ::= ws value ws"


def test_no_choices_raises() -> None:
    provider = LlamaCppProvider("m.gguf", llama=FakeLlama({"choices": []}))
    with pytest.raises(ProviderError):
        provider.complete(CompletionRequest(messages=[Message.user("x")]))


def test_stream_reassembles_deltas() -> None:
    stream = [
        {"choices": [{"delta": {"content": "he"}}]},
        {"choices": [{"delta": {"content": "llo"}, "finish_reason": "stop"}]},
    ]
    provider = LlamaCppProvider("m.gguf", llama=FakeLlama(stream))
    chunks = list(provider.stream(CompletionRequest(messages=[Message.user("x")])))
    assert "".join(c.delta for c in chunks) == "hello"
    assert chunks[-1].finish_reason is FinishReason.STOP


class Report(BaseModel):
    title: str
    score: int


def test_structured_output_uses_grammar_path() -> None:
    fake = FakeLlama(_completion('{"title": "disk", "score": 9}'))
    provider = LlamaCppProvider("m.gguf", llama=fake, grammar_factory=lambda g: g)
    result = structure(provider, [Message.user("report")], Report)
    assert result.value == Report(title="disk", score=9)
    assert result.strategy == "grammar"
    assert fake.calls[0]["grammar"].startswith("root ::=")


def test_build_provider_resolves_llama_cpp_aliases() -> None:
    assert isinstance(build_provider("llama-cpp", "m.gguf"), LlamaCppProvider)
    assert isinstance(build_provider("llamacpp", "m.gguf"), LlamaCppProvider)


@pytest.mark.skipif(
    importlib.util.find_spec("llama_cpp") is not None,
    reason="llama-cpp-python is installed in this environment",
)
def test_missing_dependency_is_a_clean_error() -> None:
    with pytest.raises(ProviderError):
        LlamaCppProvider("m.gguf").complete(CompletionRequest(messages=[Message.user("x")]))
