import json

import pandas as pd

from app.benchmarking.comparison import load_runs, summarize_models
from app.utils.file_handler import write_csv, write_json


def test_csv_and_json_roundtrip(tmp_path):
    df = pd.DataFrame({"model": ["a"], "x": [1.5]})
    write_csv(df, tmp_path / "t.csv")
    write_json({"k": 1}, tmp_path / "t.json")
    assert pd.read_csv(tmp_path / "t.csv").equals(df)
    assert json.loads((tmp_path / "t.json").read_text()) == {"k": 1}


def test_summary_separates_cold_warm_and_errors(tmp_path):
    base = {"model": "m", "prompt_id": "p", "category": "qa", "error": None}
    rows = [
        {**base, "phase": "cold", "load_s": 3.0, "ttft_s": 4.0, "wall_s": 6.0},
        {**base, "phase": "warm", "wall_s": 2.0, "ttft_s": 0.5, "load_s": 0.0,
         "prompt_eval_s": 0.1, "generation_s": 2.0, "tokens_generated": 20,
         "tokens_per_second": 10.0, "ram_peak_mb": 3000, "quality_score": 1.0},
        {**base, "phase": "warm", "error": "boom"},
    ]
    path = tmp_path / "run_x.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    s = summarize_models(load_runs([path])).iloc[0]
    assert s["trials_ok"] == 1 and s["trials_failed"] == 1
    assert s["avg_wall_s"] == 2.0 and s["cold_load_s"] == 3.0
    assert s["avg_ttft_overhead_s"] == 0.4  # 0.5 - 0.0 - 0.1