import json

import pytest

import app.benchmarking.benchmark_runner as br
from app.benchmarking.memory import LoadedModel
from app.core.exceptions import OllamaUnavailableError
from app.inference.generator import GenerationStats, StreamEvent
from app.models.ollama_client import ModelInfo

PROMPTS = [
    {"id": "qa-t1", "category": "question_answering", "prompt": "Capital of France?",
     "eval": {"type": "answer_any", "answers": ["paris"]}},
    {"id": "qa-t2", "category": "question_answering", "prompt": "Capital of Italy?",
     "eval": {"type": "answer_any", "answers": ["rome"]}},
]


def _fake_stream(model, messages, params=None, fmt=None):
    yield StreamEvent(text="Paris")
    yield StreamEvent(stats=GenerationStats(
        model=model, tokens_generated=10, generation_s=2.0, prompt_eval_s=0.1,
        load_s=0.0, total_s=2.3, wall_s=2.5, ttft_s=0.4))


@pytest.fixture
def config(monkeypatch, tmp_path):
    monkeypatch.setattr(br, "stream_chat", _fake_stream)
    monkeypatch.setattr(br, "list_models", lambda: [ModelInfo("m1", 1.0, "3B", "Q4_K_M", "x")])
    monkeypatch.setattr(br, "unload_model", lambda name: None)
    monkeypatch.setattr(br, "loaded_models", lambda: [])
    monkeypatch.setattr(br, "loaded_model_memory", lambda model: LoadedModel(2000.0, 0.0))

    prompts_file = tmp_path / "prompts.json"
    prompts_file.write_text(json.dumps(PROMPTS), encoding="utf-8")
    return br.BenchmarkConfig(models=["m1"], prompts_path=prompts_file,
                              results_dir=tmp_path / "raw", trials=2, warmup_runs=1)


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_cold_warmup_and_warm_rows_are_recorded(config):
    path = br.run_benchmark(config)
    rows = _rows(path)
    assert [r["phase"] for r in rows] == ["cold", "warmup"] + ["warm"] * 4
    warm = [r for r in rows if r["phase"] == "warm"]
    assert all(r["tokens_per_second"] == pytest.approx(5.0) for r in warm)  # 10 tok / 2.0 s
    assert {r["prompt_id"]: r["quality_score"] for r in warm} == {"qa-t1": 1.0, "qa-t2": 0.0}
    assert json.loads(path.with_suffix(".meta.json").read_text())["status"] == "completed"


def test_missing_model_is_reported_not_faked(config):
    config.models = ["ghost"]
    path = br.run_benchmark(config)
    assert _rows(path) == []
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    assert meta["skipped"][0]["model"] == "ghost"


def test_server_down_aborts_and_marks_failed(config, monkeypatch):
    def boom(*args, **kwargs):
        raise OllamaUnavailableError("down")


    monkeypatch.setattr(br, "stream_chat", boom)
    with pytest.raises(OllamaUnavailableError):
        br.run_benchmark(config)
    meta_file = next(config.results_dir.glob("*.meta.json"))
    assert json.loads(meta_file.read_text())["status"] == "failed"


def test_load_prompts_limit_per_category(config):
    config.limit_per_category = 1
    assert len(br.load_prompts(config)) == 1