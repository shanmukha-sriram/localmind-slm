"""Chat display helpers."""

import streamlit as st

from app.inference.generator import ChatSession, GenerationStats


def render_history(session: ChatSession) -> None:
    for message in session.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def stats_caption(stats: GenerationStats) -> None:
    ttft = f"{stats.ttft_s:.2f}s" if stats.ttft_s is not None else "n/a"
    st.caption(
        f"{stats.model} | {stats.tokens_generated} tokens | "
        f"{stats.tokens_per_second:.1f} tok/s | TTFT {ttft} | "
        f"load {stats.load_s:.2f}s | wall {stats.wall_s:.1f}s"
    )