"""Memory measurement: Ollama process RSS (sampled) and model residency (ollama ps)."""

import threading
from dataclasses import dataclass

import psutil

from app.models.ollama_client import get_client, ollama_errors

MB = 1024**2


def ollama_rss_mb() -> float:
    """Combined resident memory of every process whose name starts with 'ollama'.

    On Windows this is the working set. Memory-mapped model weights count only while
    their pages are resident, so treat it as an estimate, not an exact figure.
    """
    total = 0
    for proc in psutil.process_iter(["name"]):
        if (proc.info["name"] or "").lower().startswith("ollama"):
            try:
                total += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    return total / MB


class ResourceMonitor:
    """Context manager sampling Ollama's RAM in a background thread.

    Sampling itself uses a little CPU (observer effect), so keep the interval modest.

    """

    def __init__(self, interval_s: float = 0.25) -> None:
        self.interval_s = interval_s
        self.baseline_mb = 0.0
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._samples.append(ollama_rss_mb())

    def __enter__(self) -> "ResourceMonitor":
        self.baseline_mb = ollama_rss_mb()
        self._samples = [self.baseline_mb]
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._samples.append(ollama_rss_mb())

    @property
    def peak_mb(self) -> float:
        return max(self._samples)

    @property

    def mean_mb(self) -> float:
        return sum(self._samples) / len(self._samples)


@dataclass(frozen=True)
class LoadedModel:
    size_mb: float   # total memory the loaded model occupies, per Ollama
    vram_mb: float   # portion in GPU memory (0 on CPU-only machines)


def loaded_model_memory(model: str) -> LoadedModel | None:
    """Ollama's own view of how much memory `model` uses (None if not loaded)."""
    with ollama_errors(model):
        response = get_client().ps()
    for m in response.models:
        if m.model == model:
            return LoadedModel((m.size or 0) / MB, (m.size_vram or 0) / MB)
    return None