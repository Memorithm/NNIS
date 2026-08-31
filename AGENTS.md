# NNIS Agent Bootstrap Contract

This repository has a persistent agent-oriented execution roadmap that is intentionally kept off `main`.

## Mandatory first step

Before any autonomous coding, performance work, representation change, architectural decision, PR creation, or merge decision, read the current roadmap from:

`origin/agent/sovereignty-roadmap:.agent/NNIS_SOVEREIGNTY_ROADMAP.yaml`

Recommended command:

```bash
git fetch origin agent/sovereignty-roadmap && \
git show origin/agent/sovereignty-roadmap:.agent/NNIS_SOVEREIGNTY_ROADMAP.yaml
```

If the roadmap cannot be fetched or read, fail closed: do not make a major architectural, performance-promotion, representation-format, or merge decision. Read-only diagnosis is allowed.

## Mandatory ML maturity overlay

For any model-runtime, model I/O, dtype, KV, attention, batching, sampling, serving, benchmark, memory, throughput, quality, or cross-repository ML work, also read:

`origin/agent/sovereignty-roadmap:.agent/ML_MATURITY_5_OF_5.yaml`

Recommended command:

```bash
git fetch origin agent/sovereignty-roadmap && \
git show origin/agent/sovereignty-roadmap:.agent/ML_MATURITY_5_OF_5.yaml
```

The ML maturity file turns 5/5 into an evidence-backed exit criterion. Never promote random/tiny structural fixtures into model-quality evidence, kernel microbenchmarks into serving performance, or one Llama-like path into general model support.

## Mandatory reread points

Reread the roadmap and, for ML work, the ML maturity overlay:

1. at the start of every agent session on this repository;
2. before selecting the next major task or roadmap phase;
3. after any user instruction that changes strategy, sovereignty goals, invariants, optimization priorities, or ML maturity priorities;
4. after any physical benchmark that promotes or rejects a candidate;
5. before opening or merging any performance, kernel-selection, weight-representation, format, model-runtime, serving, or ElasticAutoTuner PR.

## Mandatory roadmap maintenance

Update the roadmap branch and the ML overlay when applicable when:

- a candidate is promoted or rejected;
- a benchmark/reference baseline changes;
- a new invariant or sovereignty constraint is introduced;
- a roadmap or ML maturity phase changes status;
- an audited ML gap is closed, regresses, or is re-scoped;
- an important negative result changes the next action.

Do not merge the roadmap or ML maturity overlay itself into `main` unless the user explicitly requests it.

## Core constraints that must never be bypassed

The roadmap is authoritative for current details, but these principles always apply:

- correctness and declared semantics dominate optimization;
- do not fabricate performance or scientific novelty;
- do not promote from microbenchmarks alone;
- incompatible benchmark environments fail closed;
- kernel elasticity and weight-representation elasticity are separate axes;
- logical weights must not be silently changed to win benchmarks;
- preserve NNIS Rust 1.77 MSRV unless explicitly revised;
- keep high-level NVIDIA software such as TensorRT/TensorRT-LLM optional references, not mandatory NNIS runtime dependencies;
- required CI and hardware evidence must correspond to the exact head being promoted.

This file is a bootstrap pointer, not the roadmap itself. The off-main sovereignty roadmap plus ML maturity overlay are the persistent sources of current agent strategy and ML execution priorities.
