"""Page 5: Model Comparison."""

import streamlit as st

from app.benchmarking.benchmark_runner import ROOT
from app.benchmarking.comparison import DEFAULT_WEIGHTS, list_runs, load_model_info
from app.evaluation.exporter import analyze
from frontend.components.charts import build_figures
from frontend.components.tables import show_table

RAW = ROOT / "results" / "raw"
PERF_COLS = [
    "model",
    "trials_ok",
    "trials_failed",
    "avg_wall_s",
    "median_wall_s",
    "min_wall_s",
    "max_wall_s",
    "p95_wall_s",
    "avg_ttft_s",
    "avg_ttft_overhead_s",
    "agg_tps",
    "avg_tps",
]
MEM_COLS = ["model", "avg_ram_peak_mb", "max_ram_peak_mb", "model_mem_mb", "model_vram_mb"]
COLD_COLS = ["model", "cold_load_s", "cold_ttft_s", "cold_wall_s"]


def render() -> None:
    st.title("📊 Model Comparison")
    runs = list_runs(RAW, "run")
    if not runs:
        st.info("No benchmark runs yet. Run one on the Benchmark page.")
        return
    names = [r.name for r in runs]
    chosen = st.multiselect("Benchmark runs", names, default=names[-1:])
    if not chosen:
        return
    if len(chosen) > 1:
        st.caption("Runs are merged by model name. Only merge runs made with the same settings.")
    paths = [RAW / n for n in chosen]

    with st.expander("Recommendation weights (an assumption you can change)"):
        raw_w = {
            k: st.slider(k.capitalize(), 0.0, 1.0, v, 0.05) for k, v in DEFAULT_WEIGHTS.items()
        }
    total = sum(raw_w.values())
    if total == 0:
        st.error("At least one weight must be above zero.")
        return
    weights = {k: v / total for k, v in raw_w.items()}

    a = analyze(paths, weights)
    if a.empty:
        st.warning("The selected runs contain no trial rows.")
        return

    st.subheader("Models tested")
    show_table(load_model_info(paths))
    st.subheader("Warm-start performance")
    show_table(a.summary, PERF_COLS)
    st.caption(
        "avg_ttft_overhead_s = TTFT minus load minus prompt evaluation: time not "
        "explained by the model itself."
    )
    if (a.summary["trials_failed"] > 0).any():
        st.warning("Some trials failed and are excluded from the averages above.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Memory")
        show_table(a.summary, MEM_COLS)
    with c2:
        st.subheader("Cold start")
        show_table(a.summary, COLD_COLS)

    st.subheader("Charts")
    figs = build_figures(a.summary, a.by_category)
    left, right = st.columns(2)
    left.plotly_chart(figs["latency"])

    right.plotly_chart(figs["tokens_per_second"])
    left.plotly_chart(figs["memory"])
    right.plotly_chart(figs["reliability"])
    st.plotly_chart(figs["quality"])

    st.subheader("Quality by category")
    show_table(a.by_category)

    st.subheader("Recommendation")
    show_table(
        a.ranked,
        [
            "model",
            "avg_quality",
            "agg_tps",
            "avg_wall_s",
            "avg_ram_peak_mb",
            "schema_success_rate",
            "score",
        ],
        decimals=3,
    )
    st.info(a.explanation)
