"""Progress callback that matches the backend's ProgressFn signature."""

from collections.abc import Callable

import streamlit as st


def progress_callback() -> Callable[[int, int, str], None]:
    bar = st.progress(0.0)
    text = st.empty()

    def update(done: int, total: int, message: str) -> None:
        bar.progress(min(done / max(total, 1), 1.0))
        text.caption(f"[{done}/{total}] {message}")

    return update