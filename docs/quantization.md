# Quantization

- **Quantization** stores weights in fewer bits (about 4-5 average bits for Q4/Q5, 8 for Q8) to reduce model file size and memory footprint.
- **GGUF** is llama.cpp's single-file format containing metadata, tokenizer, and quantized tensors.
- **Q4_K_M / Q5_K_M** use block-wise "K-quant" schemes with medium variants, while **Q8_0** uses a simple 8-bit scheme.
- Lower precision usually means less memory and faster token generation due to memory bandwidth limits on CPU decoding, but **not always**: unpacking low-bit weights requires compute, so throughput must be measured empirically.

## Valid Comparisons
Comparing variants from the same base model, size, and instruction tuning (`qwen2.5:3b-instruct`), varying only the quantization level (`Q4_K_M`, `Q5_K_M`, and `Q8_0`).

## What we tested
We evaluated three quantization variants of **`qwen2.5:3b-instruct`**:
1. **`q2.5:3b-instruct-q4_K_M`** (~1.8 GB disk footprint)
2. **`q2.5:3b-instruct-q5_K_M`** (~2.07 GB disk footprint)
3. **`q2.5:3b-instruct-q8_0`** (~3.06 GB disk footprint)

## Results

| model | quantization | disk_gb | model_mem_mb | avg_ram_peak_mb | agg_tps | avg_wall_s | avg_ttft_s | avg_quality | quality_vs_best | schema_success_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| q2.5:3b-instruct-q4_K_M | Q4_K_M | 1.80 | 2067.92 | 55.57 | 12.82 | 1.67 | 0.44 | 0.99 | -0.014 | 1.00 |
| q2.5:3b-instruct-q5_K_M | Q5_K_M | 2.07 | 2349.17 | 55.88 | 11.98 | 1.96 | 0.56 | 1.00 | 0.000 | 1.00 |
| q2.5:3b-instruct-q8_0 | Q8_0 | 3.06 | 3360.69 | 56.17 | 7.66 | 3.11 | 0.99 | 1.00 | 0.000 | 1.00 |

**Observation:** Lower bit-widths (`Q4_K_M` and `Q5_K_M`) achieved significantly higher generation throughput (**12.82** and **11.98 tokens/sec**, respectively) compared to the heavier `Q8_0` variant (**7.66 tokens/sec**) due to reduced memory bandwidth overhead on CPU inference. Meanwhile, rule-based quality and schema success rates remained nearly identical (`1.00`), proving that lower precision is ideal for efficient local deployment without sacrificing output structure or validity.