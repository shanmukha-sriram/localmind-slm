import json
from collections import Counter

from app.benchmarking.benchmark_runner import ROOT, BenchmarkConfig, _params
from app.structured_output.schemas import SCHEMAS

EVAL_TYPES = {"label", "answer_any", "contains_all", "json_fields"}


def test_yaml_values_and_overrides_are_applied(tmp_path):
    cfg_file = tmp_path / "b.yaml"
    cfg_file.write_text("trials: 4\nmax_tokens: 100\nno_think_prefixes: [foo]\n", encoding="utf-8")
    cfg = BenchmarkConfig.from_yaml(["m"], path=cfg_file, trials=7)
    assert cfg.trials == 7  # override beats file
    assert cfg.max_tokens == 100  # file beats default
    assert cfg.no_think_prefixes == ("foo",)  # list converted to tuple for str.startswith


def test_shipped_config_file_loads():
    cfg = BenchmarkConfig.from_yaml(["m"])
    assert cfg.temperature == 0.0 and cfg.seed == 42 and cfg.num_ctx == 4096


def test_thinking_is_disabled_only_for_configured_prefixes():
    cfg = BenchmarkConfig(models=[])
    item = {"category": "question_answering"}
    assert _params("qwen3:4b", item, cfg).think is False
    assert _params("llama3.2:3b", item, cfg).think is None


def test_structured_prompts_get_the_larger_token_budget():
    cfg = BenchmarkConfig(models=[])

    assert (
        _params("m", {"category": "structured_json"}, cfg).max_tokens == cfg.structured_max_tokens
    )
    assert _params("m", {"category": "reasoning"}, cfg).max_tokens == cfg.max_tokens


def test_shipped_dataset_is_well_formed():
    items = json.loads((ROOT / "data" / "benchmark" / "prompts.json").read_text(encoding="utf-8"))
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)), "prompt ids must be unique"
    assert 30 <= len(items) <= 50
    assert len(set(Counter(i["category"] for i in items).values())) == 1, "balanced categories"
    for item in items:
        assert item["prompt"].strip()
        assert item["eval"]["type"] in EVAL_TYPES
        if item["category"] == "structured_json":
            assert item["schema"] in SCHEMAS
