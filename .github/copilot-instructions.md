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

Treat `AGENTS.md` at repository root as mandatory bootstrap policy. Reread the roadmap and applicable ML overlay at every session start, before a new major task, after benchmark promotion/rejection, after strategy or ML-priority changes, and before performance/representation/model-runtime PR or merge decisions.

If the roadmap or applicable ML overlay is unavailable, fail closed for major architectural, performance-promotion, representation-format, model-runtime, or merge decisions. Do not substitute guesses for missing roadmap state.

A `5/5` maturity label is valid only after the overlay's real-model, real-GPU, quality, latency, throughput, memory, interoperability and exact-head evidence gates are actually closed.
