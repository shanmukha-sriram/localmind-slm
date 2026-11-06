"""Structured output generation with validation and a bounded retry."""

import json
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel

from app.core.exceptions import StructuredOutputError
from app.inference.generator import GenerationStats, generate_once
from app.inference.parameters import GenerationParams
from app.structured_output.retry_handler import build_retry_messages
from app.structured_output.schemas import SchemaSpec, get_schema
from app.structured_output.validator import validate_output

logger = logging.getLogger(__name__)


@dataclass
class StructuredResult:
    schema_key: str
    model: str
    constrained: bool
    data: BaseModel | None = None
    attempts: int = 0
    raw_outputs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: list[GenerationStats] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.data is not None


    def unwrap(self) -> BaseModel:
        """Return validated data or raise a clear error."""
        if self.data is None:
            raise StructuredOutputError(
                f"Validation failed after {self.attempts} attempt(s): {self.errors[-1]}",
                attempts=self.attempts,
                errors=self.errors,
            )
        return self.data


def _build_messages(spec: SchemaSpec, user_text: str, constrained: bool) -> list[dict[str, str]]:
    system = spec.instruction
    if not constrained:
        # Without schema-constrained decoding, the model must be told the shape in text.
        system += (
            "\nRespond with ONLY a JSON object that matches this JSON Schema:\n"
            + json.dumps(spec.model.model_json_schema())
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user_text}]


def generate_structured(
    model: str,
    schema_key: str,
    user_text: str,
    params: GenerationParams | None = None,
    max_retries: int = 1,
    constrained: bool = True,
) -> StructuredResult:
    """Generate JSON for `schema_key`, validate it, and retry up to `max_retries` times.


    constrained=True  -> pass the JSON Schema to Ollama's `format` (schema-constrained decoding)
    constrained=False -> rely on the prompt alone (lets you observe real validation failures)

    Returns a StructuredResult (never raises on validation failure) so callers such as the
    benchmark can count successes. Call result.unwrap() if you prefer an exception.
    """
    spec = get_schema(schema_key)
    # Low temperature for extraction; generous token cap, since truncated JSON is invalid JSON.
    params = params or GenerationParams(temperature=0.0, max_tokens=1024)
    fmt = spec.model.model_json_schema() if constrained else None
    messages = _build_messages(spec, user_text, constrained)
    result = StructuredResult(schema_key=schema_key, model=model, constrained=constrained)

    for attempt in range(1, max_retries + 2):  # first try + retries
        generated = generate_once(model, messages, params, fmt)
        result.attempts = attempt
        result.raw_outputs.append(generated.content)
        result.stats.append(generated.stats)

        outcome = validate_output(generated.content, spec.model)
        if outcome.ok:
            result.data = outcome.data
            return result

        result.errors.append(f"[{outcome.stage}] {outcome.error}")
        logger.warning("Attempt %d failed validation: %s", attempt, outcome.error)
        messages = build_retry_messages(messages, generated.content, outcome.error or "")

    return result