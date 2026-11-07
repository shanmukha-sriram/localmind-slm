"""Descriptive statistics for timing data."""

import math
from collections.abc import Iterable
from statistics import mean, median, stdev


def summarize(values: Iterable[float | None]) -> dict[str, float | int | None]:
    """n, mean, median, min, max, p95 (nearest-rank) and stdev, ignoring None/NaN."""
    vals = sorted(v for v in values if v is not None and not math.isnan(v))
    if not vals:
        return dict.fromkeys(("mean", "median", "min", "max", "p95", "stdev"), None) | {"n": 0}
    n = len(vals)
    return {
        "n": n,
        "mean": mean(vals),
        "median": median(vals),
        "min": vals[0],
        "max": vals[-1],
        "p95": vals[math.ceil(0.95 * n) - 1],
        "stdev": stdev(vals) if n > 1 else 0.0,
    }