"""Turn raw benchmark runs into summaries and files (shared by the CLI and the UI)."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.benchmarking.comparison import (
    explain,
    load_runs,
    recommend,
    summarize_by_category,
    summarize_models,
)
from app.evaluation.report_generator import build_markdown_report
from app.utils.file_handler import atomic_write_text, write_csv, write_json


@dataclass
class Analysis:
    tag: str
    trials: pd.DataFrame
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_category: pd.DataFrame = field(default_factory=pd.DataFrame)
    ranked: pd.DataFrame = field(default_factory=pd.DataFrame)
    explanation: str = "No trial rows found."
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return self.trials.empty


def analyze(paths: list[Path], weights: dict[str, float] | None = None) -> Analysis:
    tag = paths[0].stem if len(paths) == 1 else "combined"
    df = load_runs(paths)
    if df.empty:
        return Analysis(tag=tag, trials=df)
    summary = summarize_models(df)
    ranked = recommend(summary, weights)
    meta_path = paths[0].with_suffix(".meta.json")
    meta = {}
    if len(paths) == 1 and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return Analysis(tag, df, summary, summarize_by_category(df), ranked, explain(ranked), meta)


def write_artifacts(
    a: Analysis, processed_dir: Path, reports_dir: Path, charts_html: str | None = None
) -> list[Path]:
    """Write CSV/JSON/Markdown (and charts HTML if given). Returns the files written."""
    files = {
        processed_dir / f"{a.tag}_trials.csv": lambda p: write_csv(
            a.trials.drop(columns=["response"]), p
        ),
        processed_dir / f"{a.tag}_summary.csv": lambda p: write_csv(a.summary, p),
        processed_dir / f"{a.tag}_by_category.csv": lambda p: write_csv(a.by_category, p),
        processed_dir / f"{a.tag}_summary.json": lambda p: write_json(
            json.loads(a.summary.to_json(orient="records")), p
        ),
        reports_dir / f"{a.tag}_report.md": lambda p: atomic_write_text(
            p, build_markdown_report(a.summary, a.by_category, a.ranked, a.explanation, a.meta)
        ),
    }
    if charts_html:
        files[reports_dir / f"{a.tag}_charts.html"] = lambda p: atomic_write_text(p, charts_html)
    for path, writer in files.items():
        writer(path)
    return list(files)
