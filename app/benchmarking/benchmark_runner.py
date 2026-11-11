"""The benchmark engine: cold start, warm-up, repeated warm trials, safe raw logging."""

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import yaml

from app.benchmarking.memory import ResourceMonitor, loaded_model_memory
from app.core.exceptions import LocalMindError, ModelNotFoundError, OllamaUnavailableError
from app.evaluation.quality_metrics import evaluate
from app.inference.generator import stream_chat
from app.inference.parameters import GenerationParams
from app.models.ollama_client import list_models, loaded_models, unload_model
from app.structured_output.generator import build_messages
from app.structured_output.schemas import get_schema
from app.structured_output.validator import validate_output
from app.utils import system_info
from app.utils.file_handler import write_json

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "benchmark.yaml"
ProgressFn = Callable[[int, int, str], None]


@dataclass
class BenchmarkConfig:
    models: list[str]
    prompts_path: Path = ROOT / "data" / "benchmark" / "prompts.json"
    results_dir: Path = ROOT / "results" / "raw"
    trials: int = 2
    warmup_runs: int = 1
    temperature: float = 0.0
    max_tokens: int = 256
    structured_max_tokens: int = 512
    num_ctx: int = 4096
    seed: int | None = 42
    sample_interval_s: float = 0.25
    no_think_prefixes: tuple[str, ...] = ("qwen3",)
    categories: list[str] | None = None
    limit_per_category: int | None = None
    run_prefix: str = "run"

    @classmethod
    def from_yaml(cls, models: list[str], path: Path = CONFIG_PATH, **overrides: Any):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        data = {**(data or {}), **overrides}
        if "no_think_prefixes" in data:
            data["no_think_prefixes"] = tuple(data["no_think_prefixes"])
        return cls(models=models, **data)


def load_prompts(config: BenchmarkConfig) -> list[dict[str, Any]]:
    items = json.loads(config.prompts_path.read_text(encoding="utf-8"))
    if config.categories:
        items = [i for i in items if i["category"] in config.categories]
    if config.limit_per_category:
        counts: dict[str, int] = {}
        kept = []

        for item in items:
            counts[item["category"]] = counts.get(item["category"], 0) + 1
            if counts[item["category"]] <= config.limit_per_category:
                kept.append(item)
        items = kept
    return items


def _params(model: str, item: dict[str, Any], config: BenchmarkConfig) -> GenerationParams:
    structured = item["category"] == "structured_json"
    return GenerationParams(
        temperature=config.temperature,
        max_tokens=config.structured_max_tokens if structured else config.max_tokens,
        num_ctx=config.num_ctx,
        seed=config.seed,
        think=False if model.startswith(config.no_think_prefixes) else None,
    )


def run_trial(
    model: str, item: dict[str, Any], config: BenchmarkConfig, phase: str, trial: int
) -> dict[str, Any]:
    """Run one prompt once and return a flat result row (errors become rows, not crashes)."""
    schema_model = None
    fmt = None
    messages = [{"role": "user", "content": item["prompt"]}]
    if item["category"] == "structured_json":
        spec = get_schema(item["schema"])
        schema_model = spec.model
        fmt = spec.model.model_json_schema()
        messages = build_messages(spec, item["prompt"], constrained=True)

    row: dict[str, Any] = {
        "model": model,
        "prompt_id": item["id"],
        "category": item["category"],
        "phase": phase,
        "trial": trial,
        "error": None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "json_valid": None,
        "schema_valid": None,
    }
    parts: list[str] = []
    stats = None
    try:
        with ResourceMonitor(config.sample_interval_s) as monitor:
            for event in stream_chat(model, messages, _params(model, item, config), fmt):
                if event.text:
                    parts.append(event.text)
                if event.stats:
                    stats = event.stats
        loaded = loaded_model_memory(model)
    except (OllamaUnavailableError, ModelNotFoundError):
        raise
    except LocalMindError as exc:
        row["error"] = str(exc)
        return row
    if stats is None:
        row["error"] = "Stream ended without final statistics"
        return row

    response = "".join(parts)
    structured = None
    if schema_model is not None:
        outcome = validate_output(response, schema_model)  # single attempt: raw reliability
        row["json_valid"] = outcome.stage != "json"
        row["schema_valid"] = outcome.ok
        structured = outcome.data

    score, detail = evaluate(item, response, structured)
    ram_ok = monitor.baseline_mb > 0  # 0 means no local ollama process was visible

    row.update(
        ttft_s=stats.ttft_s,
        wall_s=stats.wall_s,
        load_s=stats.load_s,
        prompt_eval_s=stats.prompt_eval_s,
        generation_s=stats.generation_s,
        ollama_total_s=stats.total_s,
        tokens_generated=stats.tokens_generated,
        prompt_tokens=stats.prompt_tokens,
        tokens_per_second=stats.tokens_per_second,
        ram_baseline_mb=monitor.baseline_mb if ram_ok else None,
        ram_peak_mb=monitor.peak_mb if ram_ok else None,
        ram_mean_mb=monitor.mean_mb if ram_ok else None,
        model_mem_mb=loaded.size_mb if loaded else None,
        model_vram_mb=loaded.vram_mb if loaded else None,
        quality_score=score,
        quality_detail=detail,
        response=response,
    )
    return row


def _describe(row: dict[str, Any]) -> str:
    head = f"{row['model']} | {row['phase']:<7} | {row['prompt_id']}"
    if row["error"]:
        return f"{head} | ERROR: {row['error']}"
    q = row["quality_score"]
    return (
        f"{head} | wall {row['wall_s']:.1f}s | {row['tokens_per_second']:.1f} tok/s | "
        f"quality {'n/a' if q is None else f'{q:.2f}'}"
    )


def _benchmark_model(
    model: str,
    prompts: list[dict[str, Any]],
    config: BenchmarkConfig,
    sink: TextIO,
    progress: ProgressFn,
    done: int,
    total: int,
) -> int:
    def record(item: dict[str, Any], phase: str, trial: int) -> None:
        nonlocal done
        row = run_trial(model, item, config, phase, trial)

        sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        sink.flush()  # every trial is on disk immediately
        done += 1
        progress(done, total, _describe(row))

    try:
        for name in loaded_models():  # guarantee a true cold start
            unload_model(name)
        record(prompts[0], "cold", 0)
        for i in range(config.warmup_runs):  # discarded in analysis
            record(prompts[i % len(prompts)], "warmup", 0)
        for trial in range(1, config.trials + 1):  # trial-major order spreads drift
            for item in prompts:
                record(item, "warm", trial)
    except ModelNotFoundError as exc:
        logger.error("Skipping %s: %s", model, exc)
    finally:
        try:
            unload_model(model)  # free RAM before the next model
        except LocalMindError:
            pass
    return done


def run_benchmark(config: BenchmarkConfig, on_progress: ProgressFn | None = None) -> Path:
    """Run the full benchmark. Returns the JSONL path (one raw row per trial)."""
    progress = on_progress or (lambda done, total, msg: None)
    prompts = load_prompts(config)
    if not prompts:
        raise ValueError("No prompts selected; check categories/limit settings.")

    installed = {m.name: m for m in list_models()}

    available = [m for m in config.models if m in installed]
    skipped = [
        {"model": m, "reason": f"not installed (run: ollama pull {m})"}
        for m in config.models
        if m not in installed
    ]
    total = (1 + config.warmup_runs + len(prompts) * config.trials) * len(available)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.results_dir / f"{config.run_prefix}_{run_id}.jsonl"
    meta_path = out_path.with_suffix(".meta.json")

    snap = system_info.snapshot()
    if snap["ram_available_gb"] < 6:
        logger.warning(
            "Only %.1f GB RAM available; close other apps for fair results.",
            snap["ram_available_gb"],
        )
    meta: dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "config": asdict(config),
        "system": snap,
        "models": {m: asdict(installed[m]) for m in available},
        "skipped": skipped,
        "n_prompts": len(prompts),
    }
    write_json(meta, meta_path)

    status = "interrupted"
    try:
        with out_path.open("a", encoding="utf-8") as sink:
            done = 0
            for model in available:
                done = _benchmark_model(model, prompts, config, sink, progress, done, total)
        status = "completed"
    except Exception:
        status = "failed"
        raise
    finally:
        meta["status"] = status
        meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(meta, meta_path)
    return out_path
