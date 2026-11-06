"""Chat generation: streaming, one-shot, and a session that keeps history."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from app.inference.parameters import GenerationParams
from app.models.ollama_client import get_client, ollama_errors

logger = logging.getLogger(__name__)
NS_PER_S = 1e9  # Ollama reports durations in nanoseconds


def _ns_to_s(value: int | None) -> float:
    return (value or 0) / NS_PER_S


@dataclass(frozen=True)
class GenerationStats:
    """Timing and token statistics for one generation.

    load_s         time Ollama spent loading the model into memory (0 when warm)
    prompt_eval_s  time spent processing the input prompt
    generation_s   time spent producing output tokens
    total_s        Ollama's own total (load + prompt eval + generation + overhead)
    wall_s         client-measured wall clock, includes network/HTTP overhead
    ttft_s         client-measured time to first visible token (streaming only)
    """

    model: str

    tokens_generated: int = 0
    prompt_tokens: int = 0
    generation_s: float = 0.0
    prompt_eval_s: float = 0.0
    load_s: float = 0.0
    total_s: float = 0.0
    wall_s: float = 0.0
    ttft_s: float | None = None

    @property
    def tokens_per_second(self) -> float:
        """Generated tokens / generation duration (excludes load and prompt eval)."""
        if self.generation_s <= 0:
            return 0.0
        return self.tokens_generated / self.generation_s

    @classmethod
    def from_response(
        cls, resp: Any, model: str, wall_s: float, ttft_s: float | None = None
    ) -> "GenerationStats":
        """Build stats from a final stream chunk or a non-streaming response."""
        return cls(
            model=model,
            tokens_generated=getattr(resp, "eval_count", None) or 0,
            prompt_tokens=getattr(resp, "prompt_eval_count", None) or 0,
            generation_s=_ns_to_s(getattr(resp, "eval_duration", None)),
            prompt_eval_s=_ns_to_s(getattr(resp, "prompt_eval_duration", None)),
            load_s=_ns_to_s(getattr(resp, "load_duration", None)),
            total_s=_ns_to_s(getattr(resp, "total_duration", None)),
            wall_s=wall_s,
            ttft_s=ttft_s,
        )



@dataclass
class ChatResult:
    content: str
    stats: GenerationStats


@dataclass
class StreamEvent:
    """Text events carry `text`; the single final event carries `stats`."""

    text: str = ""
    stats: GenerationStats | None = None


def _chat_kwargs(
    model: str,
    messages: list[dict[str, str]],
    params: GenerationParams,
    stream: bool,
    fmt: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": params.to_ollama_options(),
    }
    if fmt is not None:
        kwargs["format"] = fmt
    if params.think is not None:

        kwargs["think"] = params.think
    return kwargs


def generate_once(
    model: str,
    messages: list[dict[str, str]],
    params: GenerationParams | None = None,
    fmt: dict[str, Any] | str | None = None,
) -> ChatResult:
    """Non-streaming generation. `fmt` may be a JSON Schema dict or "json"."""
    params = params or GenerationParams()
    start = perf_counter()
    with ollama_errors(model):
        resp = get_client().chat(**_chat_kwargs(model, messages, params, False, fmt))
    wall = perf_counter() - start
    return ChatResult(resp.message.content or "", GenerationStats.from_response(resp, model, wall))


def stream_chat(
    model: str,
    messages: list[dict[str, str]],
    params: GenerationParams | None = None,
) -> Iterator[StreamEvent]:
    """Streaming generation. Yields text events, then one final stats event."""
    params = params or GenerationParams()
    start = perf_counter()
    ttft: float | None = None
    with ollama_errors(model):
        for chunk in get_client().chat(**_chat_kwargs(model, messages, params, True)):
            text = chunk.message.content or ""
            if text:

                if ttft is None:
                    ttft = perf_counter() - start
                yield StreamEvent(text=text)
            if chunk.done:
                wall = perf_counter() - start
                yield StreamEvent(stats=GenerationStats.from_response(chunk, model, wall, ttft))


@dataclass
class ChatSession:
    """A conversation with one model. Owns the message history."""

    model: str
    params: GenerationParams = field(default_factory=GenerationParams)
    system_prompt: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)
    last_stats: GenerationStats | None = None

    def _messages(self) -> list[dict[str, str]]:
        prefix = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        return prefix + self.history

    def send_stream(self, user_text: str) -> Iterator[str]:
        """Yield response text as it arrives; update history only if it completes."""
        self.history.append({"role": "user", "content": user_text})
        parts: list[str] = []
        completed = False
        try:
            for event in stream_chat(self.model, self._messages(), self.params):
                if event.text:
                    parts.append(event.text)
                    yield event.text

                if event.stats:
                    self.last_stats = event.stats
            self.history.append({"role": "assistant", "content": "".join(parts)})
            completed = True
        finally:
            # On error or early stop, drop the dangling user message so history stays consistent.
            if not completed and self.history and self.history[-1]["role"] == "user":
                self.history.pop()

    def send(self, user_text: str) -> str:
        """Non-streaming convenience wrapper."""
        return "".join(self.send_stream(user_text))

    def clear(self) -> None:
        self.history.clear()
        self.last_stats = None