import pytest

from app.core.exceptions import ModelNotFoundError
from app.inference.generator import generate_once, stream_chat
from app.inference.parameters import GenerationParams
from app.models.ollama_client import list_models
from app.structured_output.generator import generate_structured
from app.structured_output.schemas import SAMPLES, TicketClassification

pytestmark = pytest.mark.integration
HELLO = [{"role": "user", "content": "Say hello in one short sentence."}]


def test_model_discovery_lists_the_test_model(ollama_model):
    assert ollama_model in [m.name for m in list_models()]


def test_generation_returns_text_and_statistics(ollama_model, no_think):
    params = GenerationParams(temperature=0.0, max_tokens=64, think=no_think)
    result = generate_once(ollama_model, HELLO, params)
    assert result.content.strip()
    assert result.stats.tokens_generated > 0
    assert result.stats.generation_s > 0 and result.stats.tokens_per_second > 0


def test_streaming_yields_text_then_final_stats_with_ttft(ollama_model, no_think):
    params = GenerationParams(temperature=0.0, max_tokens=64, think=no_think)
    events = list(stream_chat(ollama_model, HELLO, params))
    assert "".join(e.text for e in events).strip()
    final = events[-1]
    assert final.stats is not None and final.stats.ttft_s is not None
    assert 0 < final.stats.ttft_s <= final.stats.wall_s


def test_structured_output_matches_schema(ollama_model, no_think):
    params = GenerationParams(temperature=0.0, max_tokens=512, think=no_think)
    result = generate_structured(ollama_model, "ticket", SAMPLES["ticket"], params=params)
    assert result.ok, result.errors  # schema validity only, never semantic correctness
    assert isinstance(result.data, TicketClassification)


def test_unknown_model_gives_a_clear_error(ollama_model):
    with pytest.raises(ModelNotFoundError):
        generate_once("definitely-not-a-model:0b", HELLO)
