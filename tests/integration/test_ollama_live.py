import pytest

from app.models.ollama_client import is_server_up, list_models

pytestmark = pytest.mark.integration
pytestmark_skip = pytest.mark.skipif(not is_server_up(), reason="Ollama server not running")


@pytestmark_skip
def test_model_discovery_returns_models():
    models = list_models()
    assert isinstance(models, list)