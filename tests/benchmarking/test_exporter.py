import json

from app.evaluation.exporter import analyze, write_artifacts


def test_analyze_and_write_artifacts(tmp_path):
    base = {"model": "m", "error": None, "phase": "warm", "wall_s": 2.0, "ttft_s": 0.5,
            "load_s": 0.0, "prompt_eval_s": 0.1, "generation_s": 2.0, "tokens_generated": 20,
            "tokens_per_second": 10.0, "ram_peak_mb": 3000, "model_mem_mb": 2500,
            "quality_score": 1.0, "response": "ok"}
    rows = [{**base, "prompt_id": "qa-1", "category": "question_answering"},
            {**base, "prompt_id": "js-1", "category": "structured_json",
             "json_valid": True, "schema_valid": True}]
    path = tmp_path / "run_x.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    a = analyze([path])
    assert not a.empty and a.ranked.iloc[0]["model"] == "m"
    files = write_artifacts(a, tmp_path / "p", tmp_path / "r", "<div>charts</div>")
    names = {f.name for f in files}
    assert {"run_x_summary.csv", "run_x_report.md", "run_x_charts.html"} <= names
    assert all(f.exists() for f in files)


def test_analyze_empty_file(tmp_path):
    path = tmp_path / "run_empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert analyze([path]).empty