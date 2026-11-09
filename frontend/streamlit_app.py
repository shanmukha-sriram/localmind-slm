"""Streamlit entry point. Run from the project root:

    streamlit run frontend/streamlit_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # lets `import app` and `import frontend` work

import streamlit as st

from app.core.logging_config import setup_logging
from frontend.components.sidebar import render_sidebar
from frontend.views import benchmark, chat, comparison, experiments, home, reports, structured

st.set_page_config(page_title="LocalMind", page_icon="🧠", layout="wide")
setup_logging()


def _home() -> None:
    home.render(PAGES)


PAGES = {
    "home": st.Page(_home, title="Home", icon="🏠", url_path="home", default=True),
    "chat": st.Page(chat.render, title="Local Chat", icon="💬", url_path="chat"),
    "structured": st.Page(structured.render, title="Structured Output", icon="🧩",
                          url_path="structured"),
    "benchmark": st.Page(benchmark.render, title="Benchmark Models", icon="⏱️",
                         url_path="benchmark"),
    "comparison": st.Page(comparison.render, title="Model Comparison", icon="📊",
                          url_path="comparison"),
    "experiments": st.Page(experiments.render, title="Experiments", icon="🧪",
                           url_path="experiments"),
    "reports": st.Page(reports.render, title="Reports", icon="📄", url_path="reports"),
}

navigation = st.navigation(list(PAGES.values()))
render_sidebar()
navigation.run()