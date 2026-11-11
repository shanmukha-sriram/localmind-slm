"""Table display helper."""

import pandas as pd
import streamlit as st


def show_table(df: pd.DataFrame, columns: list[str] | None = None, decimals: int = 2) -> None:
    if df.empty:
        st.info("No data to show yet.")
        return
    cols = [c for c in (columns or list(df.columns)) if c in df.columns]
    st.dataframe(df[cols].round(decimals), hide_index=True)
