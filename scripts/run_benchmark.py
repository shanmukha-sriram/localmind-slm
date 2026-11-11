"""Run the benchmark across models (Milestone 5/6)."""

import argparse

from app.benchmarking.benchmark_runner import BenchmarkConfig, run_benchmark
from app.core.exceptions import LocalMindError
from app.core.logging_config import setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--limit-per-category", type=int)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--max-tokens", type=int)
    args = parser.parse_args()

    overrides = {
        "trials": args.trials,
        "limit_per_category": args.limit_per_category,
        "categories": args.categories,
        "max_tokens": args.max_tokens,
    }
    config = BenchmarkConfig.from_yaml(
        models=args.models, **{k: v for k, v in overrides.items() if v is not None}
    )

    def progress(done: int, total: int, message: str) -> None:
        print(f"[{done}/{total}] {message}")

    try:
        path = run_benchmark(config, progress)
    except LocalMindError as exc:
        print(f"[ERROR] {exc}\nPartial results (if any) are in {config.results_dir}")
        return
    except KeyboardInterrupt:
        print(f"\nInterrupted. Partial results are saved in {config.results_dir}")
        return
    print(f"\nDone. Raw results: {path}")


if __name__ == "__main__":
    main()
