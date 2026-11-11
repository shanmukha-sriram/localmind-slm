"""Fixtures that skip, with a clear reason, when Ollama or a model is unavailable."""

import os

import pytest

from app.models.ollama_client import is_server_up, list_models


@pytest.fixture(scope="session")
def ollama_model() -> str:
    """A chat model to test with. Override with LOCALMIND_TEST_MODEL; default is the smallest."""
    if not is_server_up():
        pytest.skip("Ollama server is not running (start the Ollama app)")
    models = [m for m in list_models() if "embed" not in m.name]
    if not models:
        pytest.skip("No chat models installed (try: ollama pull llama3.2:3b)")
    wanted = os.getenv("LOCALMIND_TEST_MODEL")
    if wanted:
        if wanted not in {m.name for m in models}:
            pytest.skip(f"LOCALMIND_TEST_MODEL={wanted} is not installed")
        return wanted
    return min(models, key=lambda m: m.size_gb).name


@pytest.fixture(scope="session")
def no_think(ollama_model: str) -> bool | None:
    """Qwen3-style models need thinking disabled so short token budgets are not wasted."""
    return False if ollama_model.startswith("qwen3") else None
