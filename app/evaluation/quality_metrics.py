"""Rule-based quality scoring. Scores are 0.0-1.0 (None when there are no criteria)."""

import re
from typing import Any

from pydantic import BaseModel


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _has_word(text: str, term: str) -> bool:
    """Whole-word match, so '18' does not match '180'."""
    return re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", text) is not None


def score_label(response: str, expected: str, labels: list[str]) -> float:
    text = _norm(response)
    found = {label.lower() for label in labels if _has_word(text, label)}
    return 1.0 if found == {expected.lower()} else 0.0


def score_answer_any(response: str, answers: list[str]) -> float:
    text = _norm(response)
    return 1.0 if any(_has_word(text, a) for a in answers) else 0.0


def score_contains_all(response: str, keywords: list[str], max_words: int | None = None) -> float:
    text = _norm(response)
    hits = sum(any(alt in text for alt in kw.lower().split("|")) for kw in keywords)
    score = hits / len(keywords)

    if max_words is not None and len(response.split()) > max_words:
        score = min(score, 0.5)
    return score


def _value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict) and "any_of" in expected:
        return any(_value_matches(actual, option) for option in expected["any_of"])
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        items = [str(a).lower() for a in actual]
        return all(any(str(e).lower() in item for item in items) for e in expected)
    if isinstance(expected, str):
        return isinstance(actual, str) and expected.lower() in actual.lower()
    if isinstance(expected, int | float) and not isinstance(expected, bool):
        return isinstance(actual, int | float) and float(actual) == float(expected)
    return actual == expected


def score_json_fields(data: BaseModel | None, expected: dict[str, Any]) -> tuple[float, str]:
    if data is None:
        return 0.0, "no valid structured output"
    dumped = data.model_dump()
    wrong = [k for k, v in expected.items() if not _value_matches(dumped.get(k), v)]
    correct = len(expected) - len(wrong)
    detail = f"{correct}/{len(expected)} fields correct" + (f"; wrong: {wrong}" if wrong else "")
    return correct / len(expected), detail


def evaluate(
    item: dict[str, Any], response: str, structured: BaseModel | None = None
) -> tuple[float | None, str]:
    """Score one response against its prompt's `eval` block. Returns (score, detail)."""
    spec = item.get("eval")
    if not spec:
        return None, "no criteria"
    kind = spec["type"]
    if kind == "label":
        score = score_label(response, spec["expected"], spec["labels"])
    elif kind == "answer_any":
        score = score_answer_any(response, spec["answers"])
    elif kind == "contains_all":
        score = score_contains_all(response, spec["keywords"], spec.get("max_words"))
    elif kind == "json_fields":
        return score_json_fields(structured, spec["expected"])
    else:
        return None, f"unknown eval type '{kind}'"
    return score, kind
