import pandas as pd
import pytest

from app.benchmarking.comparison import recommend
from app.benchmarking.latency import summarize
from app.benchmarking.throughput import aggregate_tps


def test_summarize_basic():
    s = summarize([1, 2, 3, 4, 100])
    assert s["n"] == 5 and s["mean"] == pytest.approx(22.0)
    assert s["median"] == 3 and s["min"] == 1 and s["max"] == 100 and s["p95"] == 100


def test_summarize_ignores_none_and_nan():
    assert summarize([1, None, float("nan"), 3])["mean"] == 2


def test_summarize_empty():
    s = summarize([])
    assert s["n"] == 0 and s["mean"] is None


def test_aggregate_tps_is_token_weighted():
    # 10 tok in 1 s and 100 tok in 20 s: mean of ratios is 7.5, true throughput is 110/21
    assert aggregate_tps(110, 21) == pytest.approx(110 / 21)


def test_aggregate_tps_zero_duration():
    assert aggregate_tps(10, 0) == 0.0



def test_recommendation_is_not_just_the_fastest():
    summary = pd.DataFrame([
        {"model": "fast", "avg_quality": 0.4, "agg_tps": 30, "avg_wall_s": 2,
         "avg_ram_peak_mb": 2000, "schema_success_rate": 0.5},
        {"model": "good", "avg_quality": 0.9, "agg_tps": 15, "avg_wall_s": 4,
         "avg_ram_peak_mb": 3000, "schema_success_rate": 1.0},
    ])
    ranked = recommend(summary)
    assert ranked.iloc[0]["model"] == "good"
    assert ranked.iloc[0]["score"] == pytest.approx(0.55)  # 0.40 quality + 0.15 reliability