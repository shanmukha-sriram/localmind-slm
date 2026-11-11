"""Two-stage validation: JSON syntax, then Pydantic schema."""

import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ValidationError

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


@dataclass(frozen=True)
class ValidationOutcome:
    ok: bool
    data: BaseModel | None = None
    stage: Literal["json", "schema"] | None = None  # where it failed
    error: str | None = None


def strip_code_fences(text: str) -> str:
    """Models sometimes wrap JSON in ```json fences when unconstrained."""
    text = text.strip()
    match = _FENCE.match(text)
    return match.group(1) if match else text


def format_validation_error(exc: ValidationError) -> str:
    """Compact, model-friendly error text: one line per problem."""
    lines = []
    for err in exc.errors():
        location = ".".join(str(p) for p in err["loc"]) or "(root)"

        lines.append(f"{location}: {err['msg']}")
    return "; ".join(lines)


def validate_output(raw: str, schema: type[BaseModel]) -> ValidationOutcome:
    """Stage 1: parse JSON. Stage 2: validate against the Pydantic schema."""
    try:
        parsed = json.loads(strip_code_fences(raw))
    except json.JSONDecodeError as exc:
        return ValidationOutcome(False, stage="json", error=f"Invalid JSON: {exc}")
    try:
        instance = schema.model_validate(parsed)
    except ValidationError as exc:
        return ValidationOutcome(False, stage="schema", error=format_validation_error(exc))
    return ValidationOutcome(True, data=instance)
