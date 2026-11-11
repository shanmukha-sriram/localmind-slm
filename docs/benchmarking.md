# Benchmark methodology

## Metrics (what each one is)
| Metric | Meaning | Source |
|---|---|---|
| Load time | Disk -> memory, ~0 when already loaded | Ollama `load_duration` |
| Prompt eval time | Processing the input tokens | `prompt_eval_duration` |
| Generation time | Producing output tokens | `eval_duration` |
| Tokens/sec | generated tokens / generation seconds | computed |
| TTFT | Request sent -> first visible text | client clock |
| Wall latency | Request sent -> last chunk | client clock |
| TTFT overhead | TTFT - load - prompt eval | computed |
| RAM peak | Max summed working set of `ollama*` processes, sampled | psutil |
| Model memory / VRAM | Ollama's own residency figures | `ps` API |

All Ollama durations are nanoseconds, divided by 1e9.

## Fairness controls
Same prompts, hardware and limits; temperature 0; seed 42; `num_ctx` 4096; `max_tokens` 256
(512 for JSON); thinking disabled for Qwen3-style models; models unloaded before each run;
one discarded warm-up; trial-major ordering to spread drift; cold start recorded separately.

## Data handling
One JSONL row per trial, flushed immediately. Timestamped files, never overwritten. A `.meta.json`
holds config, system snapshot and run status (`completed`, `interrupted`, `failed`). Failed
trials are rows with an `error`, excluded from averages and counted in `trials_failed`.

## Statistics
Mean, median, min, max, p95, stdev per model. `avg_tps` is the mean of per-trial rates;
`agg_tps` is token-weighted (sum tokens / sum seconds). Warm trials only; cold start separate.

## Quality scoring
Rule-based: label match, whole-word answers, keyword coverage, field-by-field JSON checks. It
is reproducible but is a proxy.

## Recommendation
Weighted min-max score over quality, speed, latency, memory and reliability. The weights are an
assumption, and adjustable in the UI. Scores are relative to the models tested.

## Sources of unreliability
Thermal throttling and power plan; background processes; OS file cache; sampled RAM;
small prompt set; rule-based scoring; floating-point nondeterminism; observer effect of the
RAM sampler.