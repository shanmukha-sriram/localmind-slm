from types import SimpleNamespace

import ollama
import pytest

import app.models.ollama_client as oc
from app.core.exceptions import GenerationError, ModelNotFoundError, OllamaUnavailableError


def _model(name, size_bytes, quant="Q4_K_M"):
    details = SimpleNamespace(parameter_size="3B", quantization_level=quant, family="llama")
    return SimpleNamespace(model=name, size=size_bytes, details=details)


class FakeClient:
    def __init__(self, models):
        self._models = models

    def list(self):
        return SimpleNamespace(models=self._models)


def test_list_models_converts_and_sorts(monkeypatch):
    fake = FakeClient([_model("b:1", 2 * 1024**3), _model("a:1", 1024**3, "Q8_0")])
    monkeypatch.setattr(oc, "get_client", lambda: fake)
    models = oc.list_models()
    assert [m.name for m in models] == ["a:1", "b:1"]
    assert models[0].size_gb == 1.0 and models[0].quantization == "Q8_0"


def test_is_server_up_is_false_when_connection_fails(monkeypatch):
    class Down:
        def list(self):
            raise ConnectionError("refused")

    monkeypatch.setattr(oc, "get_client", lambda: Down())
    assert oc.is_server_up() is False


def test_connection_error_becomes_ollama_unavailable():
    with pytest.raises(OllamaUnavailableError), oc.ollama_errors("m"):
        raise ConnectionError("refused")


def test_404_becomes_model_not_found_with_pull_hint():
    with pytest.raises(ModelNotFoundError, match="ollama pull m"), oc.ollama_errors("m"):
        raise ollama.ResponseError("not found", 404)


def test_other_http_errors_become_generation_error():
    with pytest.raises(GenerationError), oc.ollama_errors("m"):
        raise ollama.ResponseError("boom", 500)
