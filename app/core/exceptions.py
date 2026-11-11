"""Project-specific exceptions so the UI can show clear messages."""


class LocalMindError(Exception):
    """Base class for all LocalMind errors."""


class OllamaUnavailableError(LocalMindError):
    """The Ollama server cannot be reached."""


class ModelNotFoundError(LocalMindError):
    """The requested model is not installed locally."""


class GenerationError(LocalMindError):
    """Ollama was reachable but the request failed."""


class UnknownSchemaError(LocalMindError):
    """The requested structured-output schema does not exist."""


class StructuredOutputError(LocalMindError):
    """Structured generation failed validation after all retries."""

    def __init__(self, message: str, attempts: int, errors: list[str]) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.errors = errors
