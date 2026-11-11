"""Page 2: Local Chat with streaming."""

import streamlit as st

from app.core.exceptions import LocalMindError
from app.inference.generator import ChatSession
from app.inference.parameters import GenerationParams
from frontend.components.chat import render_history, stats_caption
from frontend.components.sidebar import installed_models


def render() -> None:
    st.title("💬 Local Chat")
    models = installed_models()
    if not models:
        st.warning("No models available. Start Ollama and pull a model.")
        return

    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("Model", [m.name for m in models])
    temperature = c2.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    max_tokens = int(c3.number_input("Max output tokens", 16, 4096, 512, step=16))
    no_think = st.checkbox(
        "Disable thinking (for Qwen3-style models)",
        value=model.startswith("qwen3"),
        key=f"nothink_{model}",
    )

    if "chat" not in st.session_state:
        st.session_state.chat = ChatSession(model=model)
    chat: ChatSession = st.session_state.chat
    chat.model = model
    chat.params = GenerationParams(
        temperature=temperature, max_tokens=max_tokens, think=False if no_think else None
    )

    if st.button("Clear chat"):
        chat.clear()
        st.rerun()

    render_history(chat)

    if prompt := st.chat_input("Message the model"):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                st.write_stream(chat.send_stream(prompt))
            except LocalMindError as exc:
                st.error(str(exc))
            else:
                if chat.last_stats:
                    stats_caption(chat.last_stats)
