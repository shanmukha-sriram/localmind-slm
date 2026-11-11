"""Markdown summary report built from computed summaries."""

import math
from typing import Any

import pandas as pd


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def md_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def build_markdown_report(
    summary: pd.DataFrame,
    by_category: pd.DataFrame,
    ranked: pd.DataFrame,
    explanation: str,
    meta: dict[str, Any] | None = None,
) -> str:
    meta = meta or {}
    system = meta.get("system", {})
    config = meta.get("config", {})
    quality_pivot = by_category.pivot(index="model", columns="category", values="quality")
    quality_pivot = quality_pivot.reset_index()

    parts = [
        "# LocalMind Benchmark Report",
        f"- Run: `{meta.get('run_id', 'combined')}` (status: {meta.get('status', 'n/a')})",
        f"- System: {system.get('cpu', 'n/a')}, {system.get('ram_total_gb', 'n/a')} GB RAM, "
        f"{system.get('os', 'n/a')} (CPU-only inference)",
        f"- Settings: temperature {config.get('temperature', 'n/a')}, "
        f"max_tokens {config.get('max_tokens', 'n/a')}, num_ctx {config.get('num_ctx', 'n/a')}, "
        f"seed {config.get('seed', 'n/a')}, trials {config.get('trials', 'n/a')}",
        "",
        "## Warm-start performance",
        md_table(
            summary,
            [
                "model",
                "trials_ok",
                "trials_failed",
                "avg_wall_s",
                "median_wall_s",
                "min_wall_s",
                "max_wall_s",
                "avg_ttft_s",
                "avg_ttft_overhead_s",
                "agg_tps",
                "avg_tps",
            ],
        ),
        "",
        "## Memory",
        md_table(
            summary,
            ["model", "avg_ram_peak_mb", "max_ram_peak_mb", "model_mem_mb", "model_vram_mb"],
        ),
        "",
        "## Cold start (first request after unloading)",
        md_table(summary, ["model", "cold_load_s", "cold_ttft_s", "cold_wall_s"]),
        "",
        "## Quality (0-1, rule-based) by category",
        md_table(quality_pivot, list(quality_pivot.columns)),
        "",
        "## Reliability and recommendation",
        md_table(ranked, ["model", "avg_quality", "schema_success_rate", "score"]),
        "",
        explanation,
        "",
        "## Caveats",
        "- Quality uses simple rules and a small prompt set; it is a proxy, not ground truth.",
        "- RAM is sampled process memory (working set) and is approximate.",
        "- Results depend on this machine, its power mode and background load.",
    ]
    return "\n".join(parts) + "\n"
