"""Page 6: Experiments (temperature and quantization)."""

from pathlib import Path

import streamlit as st

from app.benchmarking.benchmark_runner import ROOT, BenchmarkConfig, load_prompts
from app.benchmarking.comparison import list_runs
from app.benchmarking.experiments import (
    analyze_quantization,
    analyze_temperature,
    check_quant_compatibility,
    run_quantization_experiment,
    run_temperature_experiment,
)
from app.core.exceptions import LocalMindError
from frontend.components.charts import quantization_figures, temperature_chart
from frontend.components.progress import progress_callback
from frontend.components.sidebar import installed_models
from frontend.components.tables import show_table

RAW = ROOT / "results" / "raw"
QUANT_COLS = ["model", "quantization", "disk_gb", "model_mem_mb", "avg_ram_peak_mb", "agg_tps",
              "avg_wall_s", "avg_ttft_s", "avg_quality", "quality_vs_best", "schema_success_rate"]


def render() -> None:
    st.title("🧪 Experiments")
    models = installed_models()
    if not models:
        st.warning("No models available.")
        return
    names = [m.name for m in models]
    temp_tab, quant_tab = st.tabs(["Temperature", "Quantization"])
    with temp_tab:
        _temperature(names)
    with quant_tab:
        _quantization(names)


def _pick(prefix: str, key: str) -> Path | None:
    files = list_runs(RAW, prefix)
    if not files:
        st.info("No saved experiments of this kind yet.")
        return None
    name = st.selectbox("Saved experiment", [f.name for f in reversed(files)], key=key)
    return RAW / name


def _temperature(names: list[str]) -> None:
    st.write("Same model, same prompts, no fixed seed, repeated at temperature 0 and 0.7.")
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("Model", names, key="temp_model")
    repeats = c2.slider("Repeats per prompt", 2, 5, 3)
    per_cat = c3.slider("Prompts per category", 1, 3, 2)
    n = len(load_prompts(BenchmarkConfig(models=[], limit_per_category=per_cat)))
    st.caption(f"{n * 2 * repeats + 1} requests. Don't touch widgets while it runs.")
    if st.button("Run temperature experiment", type="primary"):
        try:
            run_temperature_experiment(model, repeats=repeats, per_category=per_cat,
                                       on_progress=progress_callback())
            st.success("Finished.")
        except LocalMindError as exc:
            st.error(str(exc))

    path = _pick("temp", "temp_file")
    if path is None:
        return
    by_temp, per_prompt, rows = analyze_temperature(path)
    if by_temp.empty:
        st.warning("This experiment has no successful rows.")
        return
    show_table(by_temp, decimals=3)
    st.plotly_chart(temperature_chart(by_temp))
    st.caption("exact_agreement 1.0 means every repeat was identical. If temperature 0 is below "
               "1.0 here, that is a real result: temperature 0 reduces variation but does not "
               "guarantee identical output.")
    with st.expander("Per-prompt results"):
        show_table(per_prompt, decimals=3)
    st.subheader("Compare actual responses")
    pid = st.selectbox("Prompt", sorted(per_prompt["prompt_id"].unique()))
    sel = rows[(rows["phase"] == "warm") & rows["error"].isna() & (rows["prompt_id"] == pid)]
    for temp, group in sel.groupby("temperature"):
        st.markdown(f"**Temperature {temp}**")
        for _, r in group.sort_values("trial").iterrows():
            st.text(f"#{int(r['trial'])}: {r['response']}")


def _quantization(names: list[str]) -> None:
    st.write("Compare variants of the same model at different precisions. Pull them first "
             "(see the Day 4 guide) and check the tags on the Ollama library page.")
    variants = st.multiselect("Variants (same model, different quantization)", names)
    ready = False
    if len(variants) >= 2:
        try:
            problems = check_quant_compatibility(variants)
        except LocalMindError as exc:
            problems = [str(exc)]
        for p in problems:
            st.error(p)
        ready = not problems
        if ready:
            st.success("Same family and size, distinct quantization levels. Also confirm that "
                       "all tags are the same kind (all instruct).")
    c1, c2 = st.columns(2)
    trials = c1.slider("Trials", 1, 3, 2, key="q_trials")
    limit = c2.slider("Prompts per category", 1, 6, 3, key="q_limit")
    if st.button("Run quantization experiment", type="primary", disabled=not ready):
        try:
            run_quantization_experiment(variants, trials, limit, progress_callback())
            st.success("Finished.")
        except (LocalMindError, ValueError) as exc:
            st.error(str(exc))

    path = _pick("quant", "quant_file")
    if path is None:
        return
    table = analyze_quantization(path)
    if table.empty:
        st.warning("This experiment has no successful rows.")
        return
    show_table(table, QUANT_COLS, decimals=3)
    for fig in quantization_figures(table).values():
        st.plotly_chart(fig)
    st.caption("Small quality differences on this prompt set are within noise. Higher "
               "precision is not guaranteed to be slower, and lower precision is not "
               "guaranteed to be faster: compare the measured columns.")