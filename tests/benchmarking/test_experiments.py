import json

import pytest

import app.benchmarking.experiments as exp
from app.models.ollama_client import ModelInfo

PROMPTS = [
    {"id": "qa-1", "category": "question_answering", "prompt": "Capital of France?"},
    {"id": "qa-2", "category": "question_answering", "prompt": "Capital of Italy?"},
]


def _write_rows(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _row(temp, response, repeat, prompt_id="qa-1", quality=1.0):
    return {
        "model": "m",
        "prompt_id": prompt_id,
        "category": "qa",
        "phase": "warm",
        "trial": repeat,
        "error": None,
        "temperature": temp,
        "response": response,
        "quality_score": quality,
        "tokens_generated": 5,
    }


def test_analysis_separates_identical_and_varied_outputs(tmp_path):
    rows = [_row(0.0, "Paris", i) for i in (1, 2, 3)]
    rows += [_row(0.7, r, i) for i, r in enumerate(["Paris", "Paris.", "The capital is Paris"], 1)]
    path = tmp_path / "temp_x.jsonl"
    _write_rows(path, rows)
    by_temp, per_prompt, _ = exp.analyze_temperature(path)
    cold = by_temp[by_temp["temperature"] == 0.0].iloc[0]
    hot = by_temp[by_temp["temperature"] == 0.7].iloc[0]
    assert cold["exact_agreement"] == 1.0 and cold["unique_mean"] == 1.0

    assert hot["exact_agreement"] == pytest.approx(1 / 3) and hot["unique_mean"] == 3.0
    assert hot["similarity"] < cold["similarity"] == 1.0


def test_analysis_ignores_warmup_and_errors(tmp_path):
    rows = [
        _row(0.0, "Paris", 1),
        {**_row(0.0, "x", 2), "phase": "warmup"},
        {**_row(0.0, "y", 3), "error": "boom"},
    ]
    path = tmp_path / "temp_x.jsonl"
    _write_rows(path, rows)
    _, per_prompt, _ = exp.analyze_temperature(path)
    assert per_prompt.iloc[0]["n"] == 1


def test_experiment_order_and_rows(monkeypatch, tmp_path):
    def fake_trial(model, item, config, phase, trial):
        return {
            "model": model,
            "prompt_id": item["id"],
            "category": item["category"],
            "phase": phase,
            "trial": trial,
            "error": None,
            "response": "x",
        }

    monkeypatch.setattr(exp, "run_trial", fake_trial)
    monkeypatch.setattr(exp, "load_prompts", lambda cfg: PROMPTS)
    monkeypatch.setattr(exp, "list_models", lambda: [ModelInfo("m", 1.0, "3B", "Q4_K_M", "f")])
    monkeypatch.setattr(exp, "loaded_models", lambda: [])
    monkeypatch.setattr(exp, "unload_model", lambda name: None)
    path = exp.run_temperature_experiment("m", repeats=2, results_dir=tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["phase"] == "warmup" and len(rows) == 1 + 2 * 2 * 2
    warm = [r for r in rows if r["phase"] == "warm"]
    assert [r["temperature"] for r in warm[:4]] == [0.0, 0.0, 0.7, 0.7]  # temp-major in a repeat
    assert json.loads(path.with_suffix(".meta.json").read_text())["status"] == "completed"


def test_quant_compatibility(monkeypatch):

    infos = {
        "a-q4": ModelInfo("a-q4", 2.0, "3B", "Q4_K_M", "qwen2"),
        "a-q8": ModelInfo("a-q8", 3.3, "3B", "Q8_0", "qwen2"),
        "b": ModelInfo("b", 2.0, "3B", "Q4_K_M", "llama"),
        "c": ModelInfo("c", 1.0, "1B", "Q8_0", "qwen2"),
    }
    monkeypatch.setattr(exp, "list_models", lambda: list(infos.values()))
    assert exp.check_quant_compatibility(["a-q4", "a-q8"]) == []
    assert exp.check_quant_compatibility(["a-q4", "b"])  # same quant level AND different family
    assert any("parameter" in p for p in exp.check_quant_compatibility(["a-q4", "c"]))
    assert any("not installed" in p for p in exp.check_quant_compatibility(["a-q4", "ghost"]))
    assert exp.check_quant_compatibility(["a-q4"])
