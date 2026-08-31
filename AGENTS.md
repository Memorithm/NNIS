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

## Mandatory kernel-agent integration overlay

For any Forge-driven kernel search, generated CUDA candidate, CUDA-agent-style optimization loop, cross-repository kernel verification/measurement envelope, or numerical-policy comparison, also read:

`origin/agent/sovereignty-roadmap:.agent/KERNEL_AGENT_INTEGRATION_PROGRAM.yaml`

Recommended command:

```bash
git fetch origin agent/sovereignty-roadmap && \
git show origin/agent/sovereignty-roadmap:.agent/KERNEL_AGENT_INTEGRATION_PROGRAM.yaml
```

NNIS remains the owner of runtime/kernel semantics, reference paths, environment-compatible benchmark evidence, end-to-end requalification, and final promotion. Forge may propose/search/select candidates but may not self-authorize NNIS runtime changes. Baseline and candidate numerical semantics must be explicit and comparable; TF32, mixed precision, relaxed tolerances, or other precision-policy changes are distinct campaigns rather than hidden speedups. Compilation success never substitutes for correctness, and isolated kernel speed never substitutes for real-model end-to-end evidence.

## Mandatory DA-LUC research overlay

For any compressed/quantized KV representation, codebook/PQ/LUT scoring, sparse outlier residual, direct compressed-attention kernel, dynamic KV precision tiering, or PHR-Lite KV-routing work, also read:

`origin/agent/sovereignty-roadmap:.agent/DA_LUC_RESEARCH_PROGRAM.yaml`

Recommended command:

```bash
git fetch origin agent/sovereignty-roadmap && \
git show origin/agent/sovereignty-roadmap:.agent/DA_LUC_RESEARCH_PROGRAM.yaml
```

DA-LUC is a research program, not a novelty or performance claim. The verifiable target is direct consumption of an explicit compressed representation without dense K/V materialization. Do not describe this as "zero dequantization" when scalar/register conversion still occurs. Any claimed KV compression ratio must use measured effective bits/value including codebooks, scales/zero-points, residual values and indices/bitmaps, metadata, alignment, and padding. A nominal 8x-16x target is never a result by itself.

PHR-Lite is restricted to optional KV precision/retention/page/offload/eviction routing after simpler routing baselines exist. It does not replace tokenization, next-token prediction, the LM head, or the model objective in the NNIS research program.

If DA-LUC work is applicable and the research overlay cannot be read, fail closed for representation design, kernel promotion, runtime selection, performance claims, or merge decisions. Read-only diagnosis is allowed.

## Mandatory reread points

Reread the roadmap and, when applicable, the ML maturity, kernel-agent integration, and DA-LUC overlays:

1. at the start of every agent session on this repository;
2. before selecting the next major task or roadmap phase;
3. after any user instruction that changes strategy, sovereignty goals, invariants, optimization priorities, or ML maturity priorities;
4. after any physical benchmark that promotes or rejects a candidate;
5. before opening or merging any performance, kernel-selection, weight-representation, KV-format, model-runtime, serving, Forge/kernel-agent, DA-LUC, PHR-Lite, or ElasticAutoTuner PR.

## Mandatory roadmap maintenance

Update the roadmap branch and the applicable overlays when:

- a candidate is promoted or rejected;
- a benchmark/reference baseline changes;
- a new invariant or sovereignty constraint is introduced;
- a roadmap or ML maturity phase changes status;
- an audited ML gap is closed, regresses, or is re-scoped;
- a kernel-agent backend contract or candidate-promotion boundary changes;
- a DA-LUC phase, representation contract, quality budget, or routing hypothesis changes status;
- an important negative result changes the next action.

Do not merge the roadmap or research overlays themselves into `main` unless the user explicitly requests it.

## Core constraints that must never be bypassed

The roadmap is authoritative for current details, but these principles always apply:

- correctness and declared semantics dominate optimization;
- do not fabricate performance or scientific novelty;
- do not promote from microbenchmarks alone;
- incompatible benchmark environments fail closed;
- kernel elasticity and weight-representation elasticity are separate axes;
- logical weights must not be silently changed to win benchmarks;
- compressed KV must retain a dense reference path for semantic/quality qualification;
- K and V may be represented asymmetrically only through explicit versioned plans;
- exact compressed-KV storage accounting must include every representation overhead;
- quality, memory, decode latency, and tokens/s must be reported together before compressed-KV runtime promotion;
- preserve NNIS Rust 1.77 MSRV unless explicitly revised;
- keep high-level NVIDIA software such as TensorRT/TensorRT-LLM optional references, not mandatory NNIS runtime dependencies;
- required CI and hardware evidence must correspond to the exact head being promoted.

This file is a bootstrap pointer, not the roadmap itself. The off-main sovereignty roadmap plus the applicable ML maturity, kernel-agent integration, and DA-LUC overlays are the persistent sources of current agent strategy and research priorities.
