"""Temperature and quantization experiments (Milestone 7)."""

import json
import logging
from collections import Counter
from dataclasses import replace
from datetime import datetime
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from app.benchmarking.benchmark_runner import (
    ROOT,
    BenchmarkConfig,
    ProgressFn,
    load_prompts,
    run_benchmark,
    run_trial,
)
from app.benchmarking.comparison import load_runs, summarize_models
from app.core.exceptions import LocalMindError, ModelNotFoundError
from app.models.ollama_client import list_models, loaded_models, unload_model
from app.utils import system_info
from app.utils.file_handler import write_json

logger = logging.getLogger(__name__)
RAW_DIR = ROOT / "results" / "raw"



# ---------------------------------------------------------------- temperature

def run_temperature_experiment(
    model: str,
    temperatures: tuple[float, ...] = (0.0, 0.7),
    repeats: int = 3,
    per_category: int = 2,
    categories: list[str] | None = None,
    on_progress: ProgressFn | None = None,
    results_dir: Path = RAW_DIR,
) -> Path:
    """Run the same prompts at each temperature, `repeats` times, with NO fixed seed.

    Order is repeat -> temperature -> prompt, so drift (thermal throttling etc.)
    affects every temperature equally. Every row is flushed to disk immediately.
    """
    progress = on_progress or (lambda done, total, msg: None)
    installed = {m.name: m for m in list_models()}
    if model not in installed:
        raise ModelNotFoundError(f"Model '{model}' is not installed. Run: ollama pull {model}")

    base = BenchmarkConfig(models=[model], limit_per_category=per_category,
                           categories=categories, seed=None, warmup_runs=0)
    prompts = load_prompts(base)
    if not prompts:
        raise ValueError("No prompts selected.")
    configs = {t: replace(base, temperature=t) for t in temperatures}
    total = 1 + len(prompts) * len(temperatures) * repeats

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / f"temp_{run_id}.jsonl"
    meta_path = out_path.with_suffix(".meta.json")
    meta: dict[str, Any] = {
        "experiment": "temperature", "run_id": run_id, "status": "running",
        "model": model, "temperatures": list(temperatures), "repeats": repeats,
        "prompt_ids": [p["id"] for p in prompts], "seed": None,
        "system": system_info.snapshot(), "model_info": installed[model].__dict__,
    }
    write_json(meta, meta_path)

    status, done = "interrupted", 0
    try:
        for name in loaded_models():
            unload_model(name)
        with out_path.open("a", encoding="utf-8") as sink:

            def record(item: dict[str, Any], temp: float, repeat: int, phase: str) -> None:
                nonlocal done
                row = run_trial(model, item, configs[temp], phase, repeat)
                row.update(experiment="temperature", temperature=temp)
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                done += 1
                progress(done, total, f"T={temp} | {item['id']} | repeat {repeat}")

            record(prompts[0], temperatures[0], 0, "warmup")  # loads the model; discarded
            for repeat in range(1, repeats + 1):
                for temp in temperatures:
                    for item in prompts:
                        record(item, temp, repeat, "warm")
        status = "completed"
    except Exception:

        status = "failed"
        raise
    finally:
        meta.update(status=status, finished_at=datetime.now().isoformat(timespec="seconds"))
        write_json(meta, meta_path)
        try:
            unload_model(model)
        except LocalMindError:
            pass
    return out_path


def _norm(text: Any) -> str:
    return " ".join(text.split()) if isinstance(text, str) else ""


def analyze_temperature(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (by_temperature, per_prompt, raw_rows).

    exact_agreement: share of repeats equal to the most common response (1.0 = identical).
    similarity:      mean pairwise text similarity (SequenceMatcher ratio, 0-1).
    """
    df = load_runs([path])
    empty = pd.DataFrame()
    if df.empty or "temperature" not in df:
        return empty, empty, df
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    ok = df[(df["phase"] == "warm") & df["error"].isna() & df["temperature"].notna()]
    if ok.empty:
        return empty, empty, df

    rows = []

    for (temp, prompt_id), g in ok.groupby(["temperature", "prompt_id"]):
        responses = [_norm(r) for r in g["response"]]
        n = len(responses)
        sims = [SequenceMatcher(None, a, b).ratio() for a, b in combinations(responses, 2)]
        rows.append({
            "temperature": temp, "prompt_id": prompt_id, "category": g["category"].iloc[0],
            "n": n, "unique": len(set(responses)),
            "exact_agreement": Counter(responses).most_common(1)[0][1] / n,
            "similarity": mean(sims) if sims else None,
            "quality": g["quality_score"].mean(),
            "schema_valid_rate": g["schema_valid"].map({True: 1.0, False: 0.0}).mean(),
            "tokens_mean": g["tokens_generated"].mean(),
        })
    per_prompt = pd.DataFrame(rows).sort_values(["temperature", "prompt_id"])
    by_temp = per_prompt.groupby("temperature").agg(
        prompts=("prompt_id", "count"), unique_mean=("unique", "mean"),
        exact_agreement=("exact_agreement", "mean"), similarity=("similarity", "mean"),
        quality=("quality", "mean"), schema_valid_rate=("schema_valid_rate", "mean"),
    ).reset_index()
    return by_temp, per_prompt, df


# --------------------------------------------------------------- quantization

def check_quant_compatibility(variants: list[str]) -> list[str]:
    """Return reasons a comparison is NOT valid (empty list = looks valid).

    Limits: this checks family, parameter size and distinct quantization. It cannot tell
    a base model from an instruct model, so check that both tag names say the same thing.
    """
    problems: list[str] = []
    if len(variants) < 2:

        return ["Select at least two variants."]
    installed = {m.name: m for m in list_models()}
    for v in variants:
        if v not in installed:
            problems.append(f"{v} is not installed (ollama pull {v}).")
    infos = [installed[v] for v in variants if v in installed]
    if len(infos) >= 2:
        if len({i.family for i in infos}) > 1:
            problems.append("Variants come from different model families.")
        if len({i.parameter_size for i in infos}) > 1:
            problems.append("Variants have different parameter sizes.")
        quants = [i.quantization for i in infos]
        if len(set(quants)) < len(quants):
            problems.append(f"Two variants share a quantization level ({quants}); "
                            "there is nothing to compare.")
    return problems


def run_quantization_experiment(
    variants: list[str], trials: int = 2, limit_per_category: int = 3,
    on_progress: ProgressFn | None = None,
) -> Path:
    """Benchmark quantization variants with the standard engine (same settings for all)."""
    problems = check_quant_compatibility(variants)
    if problems:
        raise ValueError(" ".join(problems))
    config = BenchmarkConfig.from_yaml(
        models=variants, trials=trials, limit_per_category=limit_per_category,
        run_prefix="quant",
    )
    return run_benchmark(config, on_progress)



def analyze_quantization(path: Path) -> pd.DataFrame:
    """Benchmark summary joined with disk size and quantization level from the run metadata."""
    df = load_runs([path])
    if df.empty:
        return pd.DataFrame()
    summary = summarize_models(df)
    meta_path = path.with_suffix(".meta.json")
    models = {}
    if meta_path.exists():
        models = json.loads(meta_path.read_text(encoding="utf-8")).get("models", {})
    info = pd.DataFrame([
        {"model": name, "quantization": m.get("quantization"), "disk_gb": m.get("size_gb")}
        for name, m in models.items()
    ], columns=["model", "quantization", "disk_gb"])
    out = summary.merge(info, on="model", how="left")
    out["disk_mb"] = out["disk_gb"] * 1024
    out["quality_vs_best"] = out["avg_quality"] - out["avg_quality"].max()
    return out