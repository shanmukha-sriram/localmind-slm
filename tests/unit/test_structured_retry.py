import pytest

import app.structured_output.generator as sgen
from app.core.exceptions import StructuredOutputError, UnknownSchemaError
from app.inference.generator import ChatResult, GenerationStats

VALID = (
    '{"category": "Billing", "priority": "High", '
    '"sentiment": "Negative", "summary": "Charged twice."}'
)


def _fake(outputs):
    it = iter(outputs)
    calls = []

    def fake_generate_once(model, messages, params=None, fmt=None):
        calls.append(messages)
        return ChatResult(next(it), GenerationStats(model=model))

    return fake_generate_once, calls


def test_succeeds_first_try(monkeypatch):
    fake, _ = _fake([VALID])
    monkeypatch.setattr(sgen, "generate_once", fake)
    result = sgen.generate_structured("m", "ticket", "text")
    assert result.ok and result.attempts == 1


def test_retry_recovers_and_includes_error_feedback(monkeypatch):
    fake, calls = _fake(["garbage", VALID])
    monkeypatch.setattr(sgen, "generate_once", fake)
    result = sgen.generate_structured("m", "ticket", "text")

    assert result.ok and result.attempts == 2
    assert "invalid" in calls[1][-1]["content"].lower()  # retry prompt contains the error


def test_fails_after_one_retry(monkeypatch):
    fake, _ = _fake(["garbage", "still garbage"])
    monkeypatch.setattr(sgen, "generate_once", fake)
    result = sgen.generate_structured("m", "ticket", "text")
    assert not result.ok and result.attempts == 2 and len(result.errors) == 2
    with pytest.raises(StructuredOutputError):
        result.unwrap()


def test_unknown_schema():
    with pytest.raises(UnknownSchemaError):
        sgen.generate_structured("m", "nope", "text")
