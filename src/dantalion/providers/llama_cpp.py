"""Adapter for an in-process llama.cpp model via ``llama-cpp-python``.

This is the only provider that runs the model inside the same process rather than
talking to a server, and the only one that can offer true grammar-constrained
decoding (GBNF). That makes it the backstop for the structured-output layer: even
a tiny model with no notion of tools or JSON mode will emit schema-valid output
when its decoding is constrained by a grammar compiled from the target type.

``llama-cpp-python`` is an optional dependency, imported lazily so the rest of the
package works without it. For tests a fake ``llama`` object can be injected, which
also lets the grammar be passed through verbatim instead of compiled.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from dantalion.errors import ProviderError
from dantalion.providers.base import Capabilities, Provider
from dantalion.providers.tokens import estimate_tokens
from dantalion.types import (
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    FinishReason,
    Message,
    Usage,
)

_FINISH_REASONS = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
}


class LlamaCppProvider:
    """Run a local GGUF model in-process, with grammar-constrained decoding."""

    name = "llama-cpp"

    def __init__(
        self,
        model_path: str | None = None,
        *,
        llama: Any = None,
        n_ctx: int = 8192,
        context_window: int | None = None,
        tool_calling: bool = False,
        grammar_factory: Callable[[str], Any] | None = None,
        **llama_kwargs: Any,
    ) -> None:
        self.model = model_path or "llama"
        self._llama = llama
        self._n_ctx = n_ctx
        self._context_window = context_window or n_ctx
        self._tool_calling = tool_calling
        self._grammar_factory = grammar_factory
        self._llama_kwargs = llama_kwargs

    def capabilities(self) -> Capabilities:
        return Capabilities(
            tool_calling=self._tool_calling,
            json_schema=False,
            grammar=True,
            context_window=self._context_window,
            max_output_tokens=min(2048, self._context_window),
        )

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        llama = self._get_llama()
        result = llama.create_chat_completion(**self._kwargs(request, stream=False))
        return self._parse(result)

    def stream(self, request: CompletionRequest) -> Iterator[CompletionChunk]:
        llama = self._get_llama()
        for chunk in llama.create_chat_completion(**self._kwargs(request, stream=True)):
            choice = (chunk.get("choices") or [{}])[0]
            delta = (choice.get("delta") or {}).get("content") or ""
            reason = choice.get("finish_reason")
            finish = _FINISH_REASONS.get(reason) if reason else None
            if delta or finish is not None:
                yield CompletionChunk(delta=delta, finish_reason=finish)

    # -- internals -------------------------------------------------------

    def _kwargs(self, request: CompletionRequest, *, stream: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "messages": [
                {"role": m.role.value, "content": m.content or ""} for m in request.messages
            ],
            "temperature": request.temperature,
            "stream": stream,
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.stop:
            kwargs["stop"] = request.stop
        if request.response_format and request.response_format.grammar:
            kwargs["grammar"] = self._compile_grammar(request.response_format.grammar)
        return kwargs

    def _parse(self, result: dict[str, Any]) -> CompletionResponse:
        choices = result.get("choices") or []
        if not choices:
            raise ProviderError("llama.cpp response contained no choices")
        choice = choices[0]
        content = (choice.get("message") or {}).get("content") or None
        usage_raw = result.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
            completion_tokens=int(usage_raw.get("completion_tokens") or estimate_tokens(content)),
        )
        reason = choice.get("finish_reason")
        return CompletionResponse(
            message=Message.assistant(content),
            model=self.model,
            usage=usage,
            finish_reason=_FINISH_REASONS.get(reason, FinishReason.STOP)
            if reason
            else FinishReason.STOP,
            raw=result,
        )

    def _compile_grammar(self, grammar: str) -> Any:
        if self._grammar_factory is not None:
            return self._grammar_factory(grammar)
        try:
            from llama_cpp import LlamaGrammar
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ProviderError(
                "grammar decoding needs llama-cpp-python; install dantalion[llama-cpp]"
            ) from exc
        return LlamaGrammar.from_string(grammar)

    def _get_llama(self) -> Any:
        if self._llama is not None:
            return self._llama
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ProviderError(
                "in-process inference needs llama-cpp-python; install dantalion[llama-cpp]"
            ) from exc
        self._llama = Llama(model_path=self.model, n_ctx=self._n_ctx, **self._llama_kwargs)
        return self._llama


_: type[Provider] = LlamaCppProvider
