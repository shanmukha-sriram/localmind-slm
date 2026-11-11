# Limitations

- **Hardware dependence:** every number comes from one CPU-only laptop. Another machine can rank models differently.
- **Model-specific behavior:** prompts, chat templates and Qwen3's no-think mode favor some models. Results describe these exact tags and versions.
- **Benchmark variability:** thermal throttling, background load and caching add noise. Few trials per prompt means wide uncertainty and no significance testing.
- **Small, synthetic prompt set:** 36 prompts, 6 per category, is a probe, not a benchmark suite.
- **Rule-based quality:** cheap and reproducible, but can mark a correct answer wrong because of phrasing and cannot judge writing quality.
- **Token counts:** tokenizers differ across models, so tokens/sec is not strictly comparable between them.
- **RAM figures:** sampled process working set, approximate with memory-mapped weights; unavailable in Docker.
- **Quantization trade-offs:** quality differences are often below this test's resolution; not every model has interchangeable variants.
- **Structured output:** valid does not mean correct; truncation can still break JSON.
- **Local deployment:** a hosted app cannot reach a local Ollama; Streamlit has no built-in authentication; one model server handles limited concurrency.