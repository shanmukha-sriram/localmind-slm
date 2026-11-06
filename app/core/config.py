"""Central configuration. Values come from environment variables with safe defaults."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Named LOCALMIND_* so it never clashes with Ollama's own OLLAMA_HOST variable,
    # which uses a different format (e.g. "0.0.0.0:11434", no scheme).
    ollama_host: str = os.getenv("LOCALMIND_OLLAMA_HOST", "http://localhost:11434")
    request_timeout_s: float = float(os.getenv("LOCALMIND_TIMEOUT_S", "180"))
    default_temperature: float = 0.7
    default_max_tokens: int = 512
    log_level: str = os.getenv("LOCALMIND_LOG_LEVEL", "INFO")


settings = Settings()