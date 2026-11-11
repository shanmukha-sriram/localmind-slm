"""Sidebar status panel and shared model discovery."""

import streamlit as st

from app.core.exceptions import LocalMindError
from app.models.ollama_client import ModelInfo, is_server_up, list_models
from app.utils import system_info


@st.cache_data(ttl=15, show_spinner=False)
def _cached_models() -> list[ModelInfo]:
    return list_models()


def installed_models() -> list[ModelInfo]:
    """Installed models, or [] with a visible error if Ollama is unreachable."""
    try:
        return _cached_models()
    except LocalMindError as exc:
        st.error(str(exc))
        return []


def render_sidebar() -> None:
    with st.sidebar:
        st.divider()
        st.caption("Ollama status")
        if is_server_up():
            st.success("Server running")
        else:
            st.error("Server not reachable. Start the Ollama app.")
        if st.button("Refresh models"):
            _cached_models.clear()
            st.rerun()
        snap = system_info.snapshot()
        st.caption(f"Free RAM: {snap['ram_available_gb']} of {snap['ram_total_gb']} GB")
        if snap["ram_available_gb"] < 6:
            st.warning("Low free RAM. Close other apps before benchmarking.")
