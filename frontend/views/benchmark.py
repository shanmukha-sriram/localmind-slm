"""Page 4: Benchmark Models."""

from pathlib import Path

import streamlit as st

from app.benchmarking.benchmark_runner import BenchmarkConfig, load_prompts, run_benchmark
from app.core.exceptions import LocalMindError
from app.evaluation.exporter import analyze
from frontend.components.progress import progress_callback
from frontend.components.sidebar import installed_models
from frontend.components.tables import show_table

SUMMARY_COLS = [
    "model",
    "trials_ok",
    "trials_failed",
    "avg_wall_s",
    "median_wall_s",
    "avg_ttft_s",
    "avg_ttft_overhead_s",
    "agg_tps",
    "avg_ram_peak_mb",
    "avg_quality",
    "schema_success_rate",
]


def render() -> None:
    st.title("⏱️ Benchmark Models")
    models = installed_models()
    if not models:
        st.warning("No models available.")
        return

    all_prompts = load_prompts(BenchmarkConfig(models=[]))
    categories = sorted({p["category"] for p in all_prompts})
    selected = st.multiselect("Models", [m.name for m in models], default=[models[0].name])
    chosen_cats = st.multiselect("Categories", categories, default=categories)
    c1, c2, c3 = st.columns(3)
    limit = c1.slider("Prompts per category (0 = all)", 0, 6, 2)

    trials = c2.slider("Trials", 1, 5, 2)
    max_tokens = c3.select_slider("Max output tokens", [64, 128, 256, 384, 512], value=256)

    config = BenchmarkConfig.from_yaml(
        models=selected,
        trials=trials,
        max_tokens=max_tokens,
        categories=chosen_cats or None,
        limit_per_category=limit or None,
    )
    n_prompts = len(load_prompts(config))
    st.caption(
        f"{n_prompts} prompts x {trials} trials + cold start + warm-up = "
        f"{n_prompts * trials + 1 + config.warmup_runs} requests per model."
    )
    st.warning(
        "Don't touch any widget while a run is in progress: Streamlit restarts the "
        "script and stops the run (finished trials stay saved). For long runs use "
        "`python -m scripts.run_benchmark`."
    )

    if st.button("Start benchmark", type="primary", disabled=not selected):
        try:
            path = run_benchmark(config, progress_callback())
        except LocalMindError as exc:
            st.error(str(exc))
        else:
            st.session_state["last_benchmark"] = str(path)
            st.success(f"Finished: {path.name}")

    last = st.session_state.get("last_benchmark")
    if last and Path(last).exists():
        st.subheader("Results")
        a = analyze([Path(last)])
        if a.empty:
            st.info("No successful rows (check that the models exist).")
        else:
            show_table(a.summary, SUMMARY_COLS)
            if (a.summary["trials_failed"] > 0).any():
                st.warning("Some trials failed. Failed trials are excluded from averages.")

            st.caption("See Model Comparison for charts and the recommendation.")
