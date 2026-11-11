# Architecture

## Layers
- **Frontend** (`frontend/`): Streamlit pages. Views hold no inference logic; they call `app.*`.
- **Inference** (`app/inference`): `GenerationParams` (validated options), `stream_chat`,
  `generate_once`, `ChatSession` (owns history), `GenerationStats` (timing and tokens).
- **Structured output** (`app/structured_output`): Pydantic schemas, two-stage validator,
  retry builder, generator.
- **Benchmarking** (`app/benchmarking`): runner, latency statistics, throughput, memory monitor,
  comparison and recommendation, experiments.
- **Evaluation** (`app/evaluation`): rule-based quality scoring, exporter, report generator.
- **Ollama client** (`app/models/ollama_client.py`): one shared client, model discovery,
  error translation into project exceptions.

```mermaid
flowchart TB
  V["frontend/views"] --> INF["inference: ChatSession, stream_chat, generate_once"]
  V --> SO["structured_output: schemas, validator, retry, generator"]
  V --> BM["benchmarking: runner, memory, comparison, experiments"]
  V --> EV["evaluation: quality_metrics, exporter, report_generator"]
  SO --> INF
  BM --> INF
  BM --> SO
  BM --> EV
  INF --> OC["models/ollama_client"]
  OC -->|"HTTP localhost:11434"| OLL[("Ollama, llama.cpp, CPU or GPU")]
  BM --> RAW[("results/raw/*.jsonl")]
  EV --> OUT[("results/processed and results/reports")]
```

## Data flows
**Chat:** UI -> `ChatSession.send_stream` -> full history + new message -> `POST /api/chat`
(stream) -> text chunks to the UI -> final chunk carries timing stats -> history updated only if
the reply completed.

**Structured output:** schema -> JSON Schema (`format`) -> generate -> `json.loads` -> Pydantic
`model_validate` -> on failure, one retry with the error fed back -> `StructuredResult`.

**Benchmark:** unload models -> cold request -> warm-up -> warm trials -> one JSONL row per trial
(flushed immediately) -> `analyze` -> summaries -> CSV, JSON, Markdown, charts.

## Optional cloud architecture
A hosted app cannot reach a laptop's Ollama. Use a separate inference server:
```mermaid
flowchart LR
  U[Users] -->|HTTPS and login| P[Reverse proxy with auth]
  P --> S[Streamlit container]
  S -->|private network, authenticated| I[Inference server on CPU or GPU VM]
```
Cost, privacy, security and operations implications: see `docs/setup.md`.