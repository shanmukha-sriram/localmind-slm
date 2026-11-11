"""Summarize raw runs and export CSV, JSON, a Markdown report and charts."""

import argparse
from pathlib import Path

from app.benchmarking.comparison import latest_run
from app.evaluation.exporter import analyze, write_artifacts
from frontend.components.charts import build_figures, figures_html

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

    a = analyze(paths)
    if a.empty:
        print("No trial rows found.")
        return
    html = figures_html(build_figures(a.summary, a.by_category))
    processed, reports = ROOT / "results" / "processed", ROOT / "results" / "reports"

    write_artifacts(a, processed, reports, html)

    cols = [
        "model",
        "trials_ok",
        "trials_failed",
        "avg_wall_s",
        "avg_ttft_s",
        "avg_ttft_overhead_s",
        "agg_tps",
        "avg_ram_peak_mb",
        "avg_quality",
        "schema_success_rate",
    ]
    print(a.summary[cols].round(2).to_string(index=False))
    print("\nRanking:\n" + a.ranked[["model", "score"]].round(3).to_string(index=False))
    print(f"\n{a.explanation}\n\nFiles written to {processed} and {reports}")


if __name__ == "__main__":
    main()
