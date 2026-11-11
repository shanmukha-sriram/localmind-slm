from types import SimpleNamespace

import pytest

from app.inference.generator import GenerationStats


def test_nanoseconds_converted_and_tps_computed():
    resp = SimpleNamespace(
        eval_count=50,
        eval_duration=2_000_000_000,
        prompt_eval_count=10,
        prompt_eval_duration=500_000_000,
        load_duration=1_500_000_000,
        total_duration=4_000_000_000,
    )
    s = GenerationStats.from_response(resp, "m", wall_s=4.2, ttft_s=0.9)
    assert s.generation_s == pytest.approx(2.0)
    assert s.load_s == pytest.approx(1.5)
    assert s.tokens_per_second == pytest.approx(25.0)  # 50 tokens / 2 s


def test_missing_fields_do_not_crash_and_tps_is_zero():
    s = GenerationStats.from_response(SimpleNamespace(), "m", wall_s=1.0)
    assert s.tokens_generated == 0 and s.tokens_per_second == 0.0
