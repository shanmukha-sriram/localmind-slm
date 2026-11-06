"""Thin wrapper around the Ollama Python library.

Responsibilities: one shared client, model discovery, and translating library errors
into our own exceptions so the rest of the app never imports ollama exceptions.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache

import httpx
import ollama

from app.core.config import settings
from app.core.exceptions import GenerationError, ModelNotFoundError, OllamaUnavailableError

logger = logging.getLogger(__name__)
GB = 1024**3


@dataclass(frozen=True)
class ModelInfo:
    name: str
    size_gb: float
    parameter_size: str | None
    quantization: str | None
    family: str | None


@lru_cache(maxsize=1)

def get_client() -> ollama.Client:
    """One shared client reuses the underlying HTTP connection."""
    return ollama.Client(host=settings.ollama_host, timeout=settings.request_timeout_s)


@contextmanager
def ollama_errors(model: str | None = None) -> Iterator[None]:
    """Translate library exceptions into LocalMind exceptions."""
    try:
        yield
    except (ConnectionError, httpx.ConnectError) as exc:
        raise OllamaUnavailableError(
            f"Cannot reach Ollama at {settings.ollama_host}. "
            "Start the Ollama app and check with: ollama list"
        ) from exc
    except ollama.ResponseError as exc:
        if exc.status_code == 404:
            raise ModelNotFoundError(
                f"Model '{model}' is not installed. Run: ollama pull {model}"
            ) from exc
        raise GenerationError(f"Ollama error ({exc.status_code}): {exc.error}") from exc


def is_server_up() -> bool:
    try:
        get_client().list()
        return True
    except (ConnectionError, httpx.ConnectError):
        return False


def list_models() -> list[ModelInfo]:

    """Return locally installed models."""
    with ollama_errors():
        response = get_client().list()
    models = []
    for m in response.models:
        details = m.details
        models.append(
            ModelInfo(
                name=m.model or "",
                size_gb=round((m.size or 0) / GB, 2),
                parameter_size=getattr(details, "parameter_size", None),
                quantization=getattr(details, "quantization_level", None),
                family=getattr(details, "family", None),
            )
        )
    return sorted(models, key=lambda x: x.name)