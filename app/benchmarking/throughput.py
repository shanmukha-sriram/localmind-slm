"""Throughput helpers."""


def aggregate_tps(total_tokens: float, total_generation_s: float) -> float:
    """Token-weighted throughput: sum(tokens) / sum(seconds).

    Unlike the mean of per-trial tokens/sec, long generations count proportionally more.
    """
    if total_generation_s <= 0:
        return 0.0
    return total_tokens / total_generation_s
