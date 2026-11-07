"""Plotly figures built from summary DataFrames (reused by the Streamlit app on Day 4)."""

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


def latency_chart(summary: pd.DataFrame) -> Figure:
    long = summary.melt(id_vars="model", value_vars=["avg_wall_s", "avg_ttft_s"],
                        var_name="metric", value_name="seconds")
    return px.bar(long, x="model", y="seconds", color="metric", barmode="group",
                  title="Average latency (lower is better)")


def tps_chart(summary: pd.DataFrame) -> Figure:
    return px.bar(summary, x="model", y="agg_tps",
                  title="Generation throughput, tokens/sec (higher is better)")


def memory_chart(summary: pd.DataFrame) -> Figure:
    long = summary.melt(id_vars="model", value_vars=["avg_ram_peak_mb", "model_mem_mb"],
                        var_name="metric", value_name="MB")
    return px.bar(long, x="model", y="MB", color="metric", barmode="group",
                  title="Memory (lower is better)")


def quality_chart(by_category: pd.DataFrame) -> Figure:
    fig = px.bar(by_category, x="category", y="quality", color="model", barmode="group",
                 title="Rule-based quality by category (higher is better)")
    fig.update_yaxes(range=[0, 1])
    return fig



def reliability_chart(summary: pd.DataFrame) -> Figure:
    fig = px.bar(summary, x="model", y="schema_success_rate",
                 title="Structured output: schema-valid on first attempt")
    fig.update_yaxes(range=[0, 1])
    return fig


def build_figures(summary: pd.DataFrame, by_category: pd.DataFrame) -> dict[str, Figure]:
    return {
        "latency": latency_chart(summary),
        "tokens_per_second": tps_chart(summary),
        "memory": memory_chart(summary),
        "quality": quality_chart(by_category),
        "reliability": reliability_chart(summary),
    }