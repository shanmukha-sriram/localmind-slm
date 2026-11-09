"""Page 1: Home."""

from dataclasses import asdict

import pandas as pd
import streamlit as st

from frontend.components.sidebar import installed_models
from app.utils import system_info
from app.models.ollama_client import is_server_up


def render(pages: dict) -> None:
    st.title("🧠 LocalMind")
    st.write(
        "Run small language models on your own computer through Ollama: chat, generate "
        "validated structured JSON, and benchmark models using real measurements."
    )
    left, right = st.columns(2)
    with left:
        st.subheader("System")
        snap = system_info.snapshot()
        st.table(pd.Series(snap, name="value").astype(str))
        st.caption("GPU offload per model appears in benchmark data as model_vram_mb "
                   "(0 means CPU-only).")
    with right:
        st.subheader("Ollama")
        if is_server_up():
            st.success("Server running")
            models = installed_models()
            if models:
                st.dataframe(pd.DataFrame([asdict(m) for m in models]), hide_index=True)
            else:
                st.info("No models installed. Run: ollama pull llama3.2:3b")
        else:
            st.error("Ollama is not reachable. Start the Ollama app and click Refresh models.")

    st.subheader("Quick links")
    cols = st.columns(3)
    labels = [("chat", "Local Chat"), ("structured", "Structured Output"),
              ("benchmark", "Benchmark Models"), ("comparison", "Model Comparison"),
              ("experiments", "Experiments"), ("reports", "Reports")]
    for i, (key, label) in enumerate(labels):
        with cols[i % 3]:
            st.page_link(pages[key], label=label)