# Engineering decisions

1. **Folded modules.** `streaming.py` lives in `inference/generator.py` (shared request building).
   `model_manager.py` and `model_registry.py` live in `ollama_client.py` (about 15 lines of discovery).
   `evaluator.py` is `quality_metrics.evaluate`. `configs/models.yaml` is omitted because models are
   discovered from Ollama. Avoids empty abstractions.
2. **Added modules.** `benchmarking/experiments.py` and `evaluation/exporter.py` (shared by the CLI and UI).
3. **Entry file `streamlit_app.py`.** A file named `app.py` beside the `app/` package breaks imports.
   Pages live in `frontend/views/`.
4. **Ollama Python library** over raw HTTP: typed responses and streaming handled for us.
5. **`LOCALMIND_*` env vars** avoid clashing with Ollama's own `OLLAMA_HOST`.
6. **JSONL for raw results:** append-safe, interruption-safe, one row per trial.
7. **Rule-based scoring** over LLM-as-judge: reproducible, no second model needed. Cost: it is a coarse proxy.
8. **Constrained decoding by default,** with an unconstrained mode to measure its effect.
9. **Retry changes the prompt** (adds the error), since identical input at T=0 repeats the failure.
10. **Fixed seed in benchmarks, no seed in the temperature experiment** (different purposes).
11. **Explicit recommendation weights,** adjustable in the UI, not a hidden formula.
12. **Missing data is `None`,** never zero (for example, RAM inside Docker).
13. **Dependencies:** declared `httpx` (used directly), removed NumPy (never imported directly).
14. **Docker default is host Ollama;** benchmarks are run natively for accurate memory and speed.