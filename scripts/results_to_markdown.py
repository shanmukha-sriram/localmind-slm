"""Print Markdown tables from saved results so docs contain only measured numbers."""

import argparse
from pathlib import Path

from app.benchmarking.comparison import list_runs
from app.benchmarking.experiments import analyze_quantization, analyze_temperature
from app.evaluation.exporter import analyze
from app.evaluation.report_generator import md_table

ROOT = Path(__file__).resolve().parents[1]
PERF = ["model", "trials_ok", "trials_failed", "avg_wall_s", "median_wall_s", "avg_ttft_s",
        "agg_tps", "avg_ram_peak_mb", "avg_quality", "schema_success_rate"]
QUANT = ["model", "quantization", "disk_gb", "model_mem_mb", "agg_tps", "avg_wall_s",
         "avg_quality", "schema_success_rate"]


def _existing(df, columns):
    return [c for c in columns if c in df.columns]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["run", "temp", "quant"], default="run")
    parser.add_argument("--path", help="a specific .jsonl file (default: newest of that kind)")
    args = parser.parse_args()

    raw = ROOT / "results" / "raw"
    path = Path(args.path) if args.path else next(iter(reversed(list_runs(raw, args.kind))), None)
    if path is None:
        print(f"No '{args.kind}' results found in {raw}")
        return

    print(f"<!-- source: {path.name} -->\n")

    if args.kind == "run":
        a = analyze([path])
        print(md_table(a.summary, _existing(a.summary, PERF)), "\n")
        print(md_table(a.ranked, ["model", "avg_quality", "schema_success_rate", "score"]), "\n")
        print(a.explanation)
    elif args.kind == "temp":
        by_temp, _, _ = analyze_temperature(path)
        print(md_table(by_temp, list(by_temp.columns)))
    else:
        table = analyze_quantization(path)
        print(md_table(table, _existing(table, QUANT)))


if __name__ == "__main__":
    main()