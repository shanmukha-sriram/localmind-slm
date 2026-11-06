"""Validated generation parameters."""

from typing import Any

from pydantic import BaseModel, Field

from app.core.config import settings


class GenerationParams(BaseModel):
    temperature: float = Field(default=settings.default_temperature, ge=0.0, le=2.0)
    max_tokens: int = Field(default=settings.default_max_tokens, ge=1, le=8192)
    num_ctx: int | None = Field(default=None, ge=256, le=131072)
    seed: int | None = None
    # Reasoning models (e.g. qwen3) can spend max_tokens "thinking". Set False to disable.
    # Leave None for models without thinking support.
    think: bool | None = None

    def to_ollama_options(self) -> dict[str, Any]:
        """Map our names to Ollama's option names, omitting unset values."""
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.seed is not None:
            options["seed"] = self.seed
        return options