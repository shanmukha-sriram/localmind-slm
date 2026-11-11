"""Page 7: Reports."""

from pathlib import Path

import streamlit as st

from app.benchmarking.benchmark_runner import ROOT
from app.benchmarking.comparison import list_runs
from app.evaluation.exporter import analyze, write_artifacts
from frontend.components.charts import build_figures, figures_html

RAW = ROOT / "results" / "raw"
PROCESSED = ROOT / "results" / "processed"
REPORTS = ROOT / "results" / "reports"
MIME = {
    ".csv": "text/csv",
    ".json": "application/json",
    ".md": "text/markdown",
    ".html": "text/html",
    ".jsonl": "application/x-ndjson",
}


def render() -> None:
    st.title("📄 Reports")
    runs = list_runs(RAW, "run")
    if not runs:
        st.info("No benchmark runs yet.")
        return
    names = [r.name for r in runs]
    chosen = st.multiselect("Benchmark runs", names, default=names[-1:])
    paths = [RAW / n for n in chosen]

    if st.button("Generate report files", type="primary", disabled=not paths):
        a = analyze(paths)
        if a.empty:
            st.warning("No trial rows in the selected runs.")
        else:
            html = figures_html(build_figures(a.summary, a.by_category))

            files = write_artifacts(a, PROCESSED, REPORTS, html)
            st.session_state["report_files"] = [str(f) for f in files + paths]
            st.success(f"Wrote {len(files)} files to results/processed and results/reports.")

    files = [f for f in map(Path, st.session_state.get("report_files", [])) if f.exists()]
    if files:
        st.subheader("Downloads")
        for f in files:
            st.download_button(
                f"Download {f.name}",
                data=f.read_bytes(),
                file_name=f.name,
                mime=MIME.get(f.suffix, "application/octet-stream"),
                key=f.name,
            )
        report = next((f for f in files if f.suffix == ".md"), None)
        if report:
            with st.expander("Report preview", expanded=True):
                st.markdown(report.read_text(encoding="utf-8"))
