# Model Selection

## Hardware this project was developed on
* **CPU:** Intel Core Processor (CPU-only, Best Performance power plan)
* **RAM:** 16.0 GB Total System RAM
* **GPU:** Intel Integrated Graphics (CPU-only execution via Ollama / llama.cpp)
* **Free Disk Space:** 50+ GB available on local SSD

## Rules of thumb
- **Weights must fit in RAM (CPU)** or VRAM (GPU), with adequate headroom for the KV cache, the OS, and background processes.
- **A 4-bit quantized model** needs roughly 0.5–0.6 GB of RAM per billion parameters for weights, plus context overhead.
- **CPU decoding** is limited by memory bandwidth, meaning larger models scale down proportionally in token generation speed.
- **Layer offloading** or running mismatched architectures that do not fit entirely in memory will severely degrade performance.

## Candidates tested
1. **`llama3.2:3b`** (~2.0 GB size, 3 Billion parameters, Q4_K_M quantization): Chosen as the primary lightweight general-purpose candidate. It offers fast token generation and robust instruction-following suitable for local CPU inference.
2. **`qwen3:4b`** (~4.7 GB size, 4 Billion parameters, standard instruction tag): Chosen to test reasoning capabilities and structured output reliability on a slightly larger parameter scale, despite internal chain-of-thought processing overhead on CPU.

## Decision
Based on our measured benchmark suite, **`llama3.2:3b`** is the recommended default model for general interactive use and low-latency structured output tasks on this hardware profile, achieving a weighted score of `1.00` with **12.32 tokens/sec** throughput. Larger models like **`qwen3:4b`** should only be selected if deep internal reasoning or specialized domain knowledge outweighs speed constraints on powerful dedicated hardware.