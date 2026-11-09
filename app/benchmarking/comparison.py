"""Load raw results, summarize per model/category, and recommend a model."""

from pathlib import Path

import pandas as pd
import json
from app.benchmarking.latency import summarize
from app.benchmarking.throughput import aggregate_tps
from app.utils.file_handler import read_jsonl

NUMERIC = [
    "trial", "ttft_s", "wall_s", "load_s", "prompt_eval_s", "generation_s", "ollama_total_s",
    "tokens_generated", "prompt_tokens", "tokens_per_second", "ram_baseline_mb", "ram_peak_mb",
    "ram_mean_mb", "model_mem_mb", "model_vram_mb", "quality_score",
]
TEXT = ["model", "prompt_id", "category", "phase", "error", "quality_detail", "response",
        "timestamp", "json_valid", "schema_valid"]

# Assumption, not a law: quality matters most. Change these to match your own priorities.
DEFAULT_WEIGHTS = {"quality": 0.40, "speed": 0.20, "latency": 0.15, "memory": 0.10,
                   "reliability": 0.15}


def latest_run(results_dir: Path) -> Path:
    runs = sorted(results_dir.glob("run_*.jsonl"))
    if not runs:
        raise FileNotFoundError(f"No benchmark runs found in {results_dir}")
    return runs[-1]


def load_runs(paths: list[Path]) -> pd.DataFrame:
    rows = []

    for path in paths:
        for row in read_jsonl(Path(path)):
            row["run"] = Path(path).stem
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in NUMERIC + TEXT:
        if col not in df:
            df[col] = None
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Time between request and first token not explained by loading or prompt processing.
    df["ttft_overhead_s"] = df["ttft_s"] - df["load_s"] - df["prompt_eval_s"]
    return df


def _ok_warm(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["phase"] == "warm") & df["error"].isna()]


def summarize_models(df: pd.DataFrame) -> pd.DataFrame:
    """One row per model, warm trials only; cold start reported separately."""
    rows = []
    for model, g in df.groupby("model"):
        warm = g[g["phase"] == "warm"]
        ok = warm[warm["error"].isna()]
        cold = g[(g["phase"] == "cold") & g["error"].isna()]
        wall = summarize(ok["wall_s"])
        structured = ok[ok["category"] == "structured_json"]
        rows.append({
            "model": model,

            "trials_ok": len(ok),
            "trials_failed": len(warm) - len(ok),
            "avg_wall_s": wall["mean"], "median_wall_s": wall["median"],
            "min_wall_s": wall["min"], "max_wall_s": wall["max"], "p95_wall_s": wall["p95"],
            "avg_ttft_s": summarize(ok["ttft_s"])["mean"],
            "avg_ttft_overhead_s": summarize(ok["ttft_overhead_s"])["mean"],
            "avg_tps": summarize(ok["tokens_per_second"])["mean"],
            "agg_tps": aggregate_tps(ok["tokens_generated"].sum(), ok["generation_s"].sum()),
            "avg_ram_peak_mb": summarize(ok["ram_peak_mb"])["mean"],
            "max_ram_peak_mb": ok["ram_peak_mb"].max(),
            "model_mem_mb": ok["model_mem_mb"].max(),
            "model_vram_mb": ok["model_vram_mb"].max(),
            "avg_quality": ok["quality_score"].mean(),
            "schema_success_rate": (
                structured["schema_valid"].astype(float).mean() if len(structured) else None
            ),
            "cold_load_s": cold["load_s"].iloc[0] if len(cold) else None,
            "cold_ttft_s": cold["ttft_s"].iloc[0] if len(cold) else None,
            "cold_wall_s": cold["wall_s"].iloc[0] if len(cold) else None,
        })
    return pd.DataFrame(rows)


def summarize_by_category(df: pd.DataFrame) -> pd.DataFrame:
    return (
        _ok_warm(df)
        .groupby(["model", "category"])
        .agg(n=("prompt_id", "count"), quality=("quality_score", "mean"),
             avg_wall_s=("wall_s", "mean"), avg_tps=("tokens_per_second", "mean"))
        .reset_index()
    )



def _minmax(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """Scale to 0-1 across the tested models. Missing data is treated as worst-case."""
    series = series.astype(float)
    series = series.fillna(series.min() if higher_is_better else series.max())
    lo, hi = series.min(), series.max()
    if pd.isna(lo) or hi == lo:
        return pd.Series(1.0, index=series.index)
    scaled = (series - lo) / (hi - lo)
    return scaled if higher_is_better else 1 - scaled


def recommend(summary: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Rank models by a weighted score. Scores are relative to the models tested."""
    w = weights or DEFAULT_WEIGHTS
    parts = {
        "quality": _minmax(summary["avg_quality"], True),
        "speed": _minmax(summary["agg_tps"], True),
        "latency": _minmax(summary["avg_wall_s"], False),
        "memory": _minmax(summary["avg_ram_peak_mb"], False),
        "reliability": _minmax(summary["schema_success_rate"], True),
    }
    ranked = summary.copy()
    ranked["score"] = sum(w[k] * parts[k] for k in w)
    return ranked.sort_values("score", ascending=False).reset_index(drop=True)


def _best(df: pd.DataFrame, col: str, highest: bool) -> str | None:
    valid = df.dropna(subset=[col])
    if valid.empty:
        return None
    return valid.loc[valid[col].idxmax() if highest else valid[col].idxmin(), "model"]



def explain(ranked: pd.DataFrame) -> str:
    """Plain-language recommendation that names the trade-offs."""
    if ranked.empty:
        return "No successful trials, so no recommendation can be made."
    top = ranked.iloc[0]["model"]
    fastest = _best(ranked, "agg_tps", True)
    best_quality = _best(ranked, "avg_quality", True)
    smallest = _best(ranked, "avg_ram_peak_mb", False)
    text = (f"Highest weighted score: {top}. Fastest generation: {fastest}. "
            f"Highest quality: {best_quality}. Lowest peak RAM: {smallest}.")
    if top != fastest:
        text += (" The top-scoring model is not the fastest: speed is only one of five "
                 "weighted factors, and here quality or reliability outweighed it.")
    text += (" Scores are relative to the models you tested and depend on the weights in "
             "DEFAULT_WEIGHTS, which are an assumption. Check the per-category quality "
             "before trusting a single number.")
    return text

def list_runs(results_dir: Path, prefix: str = "run") -> list[Path]:
    """Raw JSONL files for one kind of run (run, temp or quant), oldest first."""
    return sorted(results_dir.glob(f"{prefix}_*.jsonl"))


def load_model_info(paths: list[Path]) -> pd.DataFrame:
    """Model size and quantization recorded in each run's .meta.json."""
    rows: dict[str, dict] = {}
    for path in paths:
        meta_path = Path(path).with_suffix(".meta.json")
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for name, info in meta.get("models", {}).items():
            rows[name] = {"model": name, "size_gb": info.get("size_gb"),
                          "parameter_size": info.get("parameter_size"),
                          "quantization": info.get("quantization")}
    return pd.DataFrame(list(rows.values()))