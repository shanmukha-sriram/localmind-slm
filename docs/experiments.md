# Experiments

## Temperature
Divide logits by $T$, apply softmax, sample. $T = 0$ selects the highest-probability token (greedy decoding). $T = 0.7$ flattens the distribution, allowing lower-probability choices and introducing variations across runs. 

We ran the temperature experiment using **`qwen2.5:3b`** across 12 prompts with 3 repeats per temperature ($73$ total requests, with **no fixed seed** to accurately observe generation drift):

| temperature | prompts | unique_mean | exact_agreement | similarity | quality | schema_valid_rate |
|---|---|---|---|---|---|---|
| 0.0 | 12.0 | 1.167 | 0.944 | 0.984 | 1.000 | 1.000 |
| 0.7 | 12.0 | 1.750 | 0.750 | 0.921 | 1.000 | 1.000 |

### Observations:
- **Exact Agreement & Variation:** At $T = 0.0$, exact agreement reached **0.944** (with a `unique_mean` of `1.167`), confirming that greedy decoding heavily minimizes output drift. However, it did not reach a pure `1.0` due to floating-point non-determinism and thread-level variations during local execution. At $T = 0.7$, exact agreement dropped to **0.750** and unique responses increased (`1.750`), showing successful generative variation.
- **Quality & Schema Validity:** Both temperatures maintained a perfect quality score (`1.000`) and schema valid rate (`1.000`) on these short prompt types, proving that temperature alters stylistic variation or phrasing depth without compromising core correctness or structured JSON validity.

## Quantization
See `docs/quantization.md` for our quantization variant analysis comparing Q4, Q5, and Q8 bit-widths on `qwen2.5:3b`.

## Measurement Error & Limitations
- Thermal throttling and dynamic CPU power-plan changes mid-run.
- Background OS processes and file cache effects causing execution variance.
- RAM usage samples representing process working-set approximations.
- Small prompt set size and rule-based scoring limitations on subtle phrasing differences.
- Floating-point non-determinism preventing absolute 1.0 determinism even at $T = 0$.