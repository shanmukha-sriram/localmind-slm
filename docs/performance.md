# Performance

## Performance results

Machine: Windows 11, Intel Core CPU, 16GB RAM, CPU-only execution (Best Performance power plan). Date: 2026-11-11.

### Warm-start performance
| model | trials_ok | trials_failed | avg_wall_s | median_wall_s | min_wall_s | max_wall_s | avg_ttft_s | avg_ttft_overhead_s | agg_tps | avg_tps |
|---|---|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | 96 | 0 | 2.03 | 1.19 | 0.26 | 5.74 | 0.62 | 0.06 | 12.04 | 15.53 |
| qwen2.5:3b | 78 | 0 | 1.88 | 1.09 | 0.26 | 5.78 | 0.59 | 0.12 | 13.37 | 17.01 |
| qwen3:4b | 96 | 0 | 20.07 | 19.08 | 4.21 | 39.08 | 1.13 | 0.19 | 8.76 | 8.83 |

### Memory (MB)
| model | avg_ram_peak_mb | max_ram_peak_mb | model_mem_mb | model_vram_mb |
|---|---|---|---|---|
| llama3.2:3b | 76.61 | 79.96 | 2448.87 | 0.00 |
| qwen2.5:3b | 74.73 | 75.40 | 2067.92 | 0.00 |
| qwen3:4b | 77.19 | 80.51 | 3038.51 | 0.00 |

### Cold start (first request after unloading)
| model | cold_load_s | cold_ttft_s | cold_wall_s |
|---|---|---|---|
| llama3.2:3b | 4.46 | 5.63 | 7.97 |
| qwen2.5:3b | 7.21 | 8.52 | 11.30 |
| qwen3:4b | 7.10 | 8.57 | 21.67 |

### Quality (0-1, rule-based) by category
| model | classification | extraction | question_answering | reasoning | structured_json | summarization |
|---|---|---|---|---|---|---|
| llama3.2:3b | 1.00 | 1.00 | 1.00 | 0.69 | 1.00 | 1.00 |
| qwen2.5:3b | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 |
| qwen3:4b | 0.38 | 1.00 | 1.00 | 0.88 | 0.94 | 0.50 |

### Reliability and recommendation
| model | avg_quality | schema_success_rate | score |
|---|---|---|---|
| qwen2.5:3b | 0.99 | 1.00 | 1.00 |
| llama3.2:3b | 0.95 | 1.00 | 0.78 |
| qwen3:4b | 0.78 | 1.00 | 0.15 |

Highest weighted score: `qwen2.5:3b`. Fastest generation: `qwen2.5:3b`. Highest quality: `qwen2.5:3b`. Lowest peak RAM: `qwen2.5:3b`. Scores are relative to the models tested and depend on the weights in DEFAULT_WEIGHTS.

## Observations
- **Speed vs Quality Trade-off:** `qwen2.5:3b` and `llama3.2:3b` both deliver exceptional performance with aggregate throughputs of **13.37** and **12.04 tokens/sec** respectively, whereas `qwen3:4b` incurs heavily prolonged wall times (**20.07s** average) due to internal reasoning/thinking overhead on CPU.
- **Per-Category Winners:** `qwen2.5:3b` dominates across nearly all categories, scoring perfect `1.00` marks in classification, extraction, question answering, reasoning, and structured JSON, alongside a stellar `0.96` in summarization. `llama3.2:3b` closely follows with `1.00` across most categories but dips slightly on reasoning (`0.69`). `qwen3:4b` suffers lower scores on classification and summarization due to strict rule-based formatting mismatches with its chain-of-thought outputs.
- **Schema Reliability:** All three models attained a flawless **1.00** schema success rate when paired with structured decoding.

## Caveat on tokens/sec across models
Different models use different tokenizers, meaning "one token" does not represent the exact same amount of text or semantic weight across architectures. Therefore, models must be compared on wall time per task and functional quality alongside raw tokens/sec.