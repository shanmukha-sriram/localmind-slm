import pytest
from pydantic import ValidationError

from app.inference.parameters import GenerationParams


def test_defaults_map_to_ollama_options():
    opts = GenerationParams(temperature=0.3, max_tokens=100).to_ollama_options()
    assert opts == {"temperature": 0.3, "num_predict": 100}


def test_optional_values_included_when_set():
    opts = GenerationParams(num_ctx=2048, seed=7).to_ollama_options()
    assert opts["num_ctx"] == 2048 and opts["seed"] == 7


@pytest.mark.parametrize("bad", [-0.1, 2.5])
def test_temperature_out_of_range_rejected(bad):
    with pytest.raises(ValidationError):
        GenerationParams(temperature=bad)


def test_zero_max_tokens_rejected():
    with pytest.raises(ValidationError):
        GenerationParams(max_tokens=0)
