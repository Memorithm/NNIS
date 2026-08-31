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

## Mandatory reread points

Reread the roadmap:

1. at the start of every agent session on this repository;
2. before selecting the next major task or roadmap phase;
3. after any user instruction that changes strategy, sovereignty goals, invariants, or optimization priorities;
4. after any physical benchmark that promotes or rejects a candidate;
5. before opening or merging any performance, kernel-selection, weight-representation, format, or ElasticAutoTuner PR.

## Mandatory roadmap maintenance

Update the roadmap branch when:

- a candidate is promoted or rejected;
- a benchmark/reference baseline changes;
- a new invariant or sovereignty constraint is introduced;
- a roadmap phase changes status;
- an important negative result changes the next action.

Do not merge the roadmap itself into `main` unless the user explicitly requests it.

## Core constraints that must never be bypassed

The roadmap is authoritative for current details, but these principles always apply:

- correctness and declared semantics dominate optimization;
- do not fabricate performance or scientific novelty;
- do not promote from microbenchmarks alone;
- incompatible benchmark environments fail closed;
- kernel elasticity and weight-representation elasticity are separate axes;
- logical weights must not be silently changed to win benchmarks;
- preserve NNIS Rust 1.77 MSRV unless explicitly revised;
- keep high-level NVIDIA software such as TensorRT/TensorRT-LLM optional references, not mandatory NNIS runtime dependencies.

This file is a bootstrap pointer, not the roadmap itself. The off-main roadmap is the persistent source of current agent strategy and experimental state.
