"""Run the temperature or quantization experiment from the terminal."""

import argparse

from app.benchmarking.experiments import (
    analyze_quantization,
    analyze_temperature,
    run_quantization_experiment,
    run_temperature_experiment,
)
from app.core.exceptions import LocalMindError
from app.core.logging_config import setup_logging

QUANT_COLS = [
    "model",
    "quantization",
    "disk_gb",
    "model_mem_mb",
    "avg_ram_peak_mb",
    "agg_tps",
    "avg_wall_s",
    "avg_ttft_s",
    "avg_quality",
    "quality_vs_best",
    "schema_success_rate",
]


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("temperature")
    t.add_argument("--model", required=True)
    t.add_argument("--repeats", type=int, default=3)
    t.add_argument("--per-category", type=int, default=2)
    t.add_argument("--categories", nargs="+")
    q = sub.add_parser("quant")
    q.add_argument("--variants", nargs="+", required=True)
    q.add_argument("--trials", type=int, default=2)
    q.add_argument("--limit-per-category", type=int, default=3)
    args = parser.parse_args()

    def progress(done: int, total: int, message: str) -> None:
        print(f"[{done}/{total}] {message}")

    try:
        if args.command == "temperature":
            path = run_temperature_experiment(
                args.model,
                repeats=args.repeats,
                per_category=args.per_category,
                categories=args.categories,
                on_progress=progress,
            )
            by_temp, _, _ = analyze_temperature(path)
            print("\n" + by_temp.round(3).to_string(index=False))
        else:
            path = run_quantization_experiment(
                args.variants,
                trials=args.trials,
                limit_per_category=args.limit_per_category,
                on_progress=progress,
            )
            table = analyze_quantization(path)
            print(
                "\n" + table[[c for c in QUANT_COLS if c in table]].round(3).to_string(index=False)
            )
    except (LocalMindError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return
    except KeyboardInterrupt:
        print("\nInterrupted. Rows written so far are saved in results\\raw.")
        return
    print(f"\nRaw results: {path}")


if __name__ == "__main__":
    main()
