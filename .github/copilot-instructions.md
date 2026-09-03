# NNIS repository agent instructions

Before making repository changes, fetch and read the persistent NNIS agent roadmap:

```bash
git fetch origin agent/sovereignty-roadmap && \
git show origin/agent/sovereignty-roadmap:.agent/NNIS_SOVEREIGNTY_ROADMAP.yaml
```

For model-runtime, model-I/O, dtype, KV, attention, batching, sampling, serving, benchmark, memory, throughput, quality, or cross-repository ML work, also read:

```bash
git show origin/agent/sovereignty-roadmap:.agent/ML_MATURITY_5_OF_5.yaml
```

For compressed/quantized KV, codebook/PQ/LUT scoring, sparse outlier residuals, direct compressed-attention kernels, dynamic KV precision tiers, or PHR-Lite KV routing, also read:

```bash
git show origin/agent/sovereignty-roadmap:.agent/DA_LUC_RESEARCH_PROGRAM.yaml
```

Treat `AGENTS.md` at repository root as mandatory bootstrap policy. Reread the roadmap and applicable overlays at every session start, before a new major task, after benchmark promotion/rejection, after strategy or ML-priority changes, and before performance/representation/model-runtime/DA-LUC PR or merge decisions.

If the roadmap or any applicable overlay is unavailable, fail closed for major architectural, performance-promotion, representation-format, compressed-KV, model-runtime, or merge decisions. Do not substitute guesses for missing roadmap state.

DA-LUC is research-only until its gates pass. Do not turn nominal index width into a compression claim, call a path "zero dequantization" when conversion still occurs, or report an 8x-16x KV target without exact effective-bits/value accounting including codebooks, residuals, metadata and padding. Compressed-KV promotion requires quality, memory, latency and tokens/s evidence together.

PHR-Lite is limited to optional KV routing; it does not replace the tokenizer, next-token objective, LM head, or language-model architecture.

A `5/5` maturity label is valid only after the overlay's real-model, real-GPU, quality, latency, throughput, memory, interoperability and exact-head evidence gates are actually closed.
