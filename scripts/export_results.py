"""Summarize raw runs and export CSV, JSON, a Markdown report and charts."""

import argparse
import json
from pathlib import Path

from app.benchmarking.comparison import (
    explain, latest_run, load_runs, recommend, summarize_by_category, summarize_models,
)
from app.evaluation.report_generator import build_markdown_report
from app.utils.file_handler import atomic_write_text, write_csv, write_json
from frontend.components.charts import build_figures

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", help="JSONL paths (default: latest run)")
    parser.add_argument("--all", action="store_true", help="merge every run in results/raw")
    args = parser.parse_args()

    raw_dir = ROOT / "results" / "raw"
    if args.all:
        paths = sorted(raw_dir.glob("run_*.jsonl"))
    elif args.runs:
        paths = [Path(p) for p in args.runs]
    else:
        paths = [latest_run(raw_dir)]

    df = load_runs(paths)
    if df.empty:

        print("No trial rows found.")
        return

    summary = summarize_models(df)
    by_category = summarize_by_category(df)
    ranked = recommend(summary)
    explanation = explain(ranked)

    tag = paths[0].stem if len(paths) == 1 else "combined"
    meta_path = paths[0].with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if len(paths) == 1 and meta_path.exists() else {}

    processed, reports = ROOT / "results" / "processed", ROOT / "results" / "reports"
    write_csv(df.drop(columns=["response"]), processed / f"{tag}_trials.csv")
    write_csv(summary, processed / f"{tag}_summary.csv")
    write_csv(by_category, processed / f"{tag}_by_category.csv")
    write_json(json.loads(summary.to_json(orient="records")), processed / f"{tag}_summary.json")
    atomic_write_text(reports / f"{tag}_report.md",
                      build_markdown_report(summary, by_category, ranked, explanation, meta))

    figures = list(build_figures(summary, by_category).values())
    html = "".join(
        fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
        for i, fig in enumerate(figures)
    )
    atomic_write_text(reports / f"{tag}_charts.html", html)

    cols = ["model", "trials_ok", "trials_failed", "avg_wall_s", "avg_ttft_s",
            "avg_ttft_overhead_s", "agg_tps", "avg_ram_peak_mb", "avg_quality",
            "schema_success_rate"]
    print(summary[cols].round(2).to_string(index=False))
    print("\nRanking:\n" + ranked[["model", "score"]].round(3).to_string(index=False))

    print(f"\n{explanation}\n\nFiles written to {processed} and {reports}")


if __name__ == "__main__":
    main()