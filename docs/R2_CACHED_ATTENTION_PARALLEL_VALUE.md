# R2 cached-attention parallel-value plan

R2 has a full-model CUDA profile for the promoted E1.1 parent on the physical
Jetson AGX Thor. That evidence made cached attention a measured target rather
than a speculative fusion. The parallel-value implementation subsequently
passed both isolated and end-to-end physical gates and is now promoted for
`MinLatency` as an **explicit plan**. The historical serial path remains the
runtime default and correctness oracle.

## Nsight Systems evidence

Profile context:

- exact NNIS head: `fab2bcb2ccd6a4d3966f41dfd964708769c8aeef`;
- clean tracked worktree;
- run context: `r2-nsys-e1-1-20260831T184211Z`;
- device: NVIDIA Jetson AGX Thor, MAXN, fixed CPU/GPU/NVD/EMC clocks;
- Nsight Systems: 2025.3.2.367-253236224375v0;
- parent plan: E1.1 all-f32 LM-head GEMV64;
- representation: all-f32;
- fusion plan: baseline;
- prompt IDs: `[22007, 6463, 314]`;
- greedy decode steps: 32;
- qualified greedy prefix passed.

The trace contains 35 complete transformer passes: three prompt positions plus
32 generated positions. Kernel counts match the graph exactly:

- `nnis_gemm_f32`: 7,350 = `35 * 30 * 7`;
- `nnis_cached_attention_decode_f32`: 1,050 = `35 * 30`;
- `nnis_project_kn_f32`: 35 LM-head launches;
- top-k kernels: 32 launches each.

CUDA GPU kernel summary from that trace:

| Kernel family | GPU time | Share | Instances | Average | Median |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nnis_gemm_f32` | 393.825792 ms | 62.8% | 7,350 | 53.5817 us | 41.216 us |
| `nnis_cached_attention_decode_f32` | 194.850528 ms | 31.1% | 1,050 | 185.5719 us | 184.800 us |
| `nnis_project_kn_f32` | 21.690304 ms | 3.5% | 35 | 619.723 us | 595.936 us |
| `nnis_weighted_rmsnorm_f32` | 8.000064 ms | 1.3% | 2,135 | 3.7471 us | 3.648 us |

The percentages above are percentages of captured GPU kernel time, not wall
clock. The profiled wall-clock generation time must not be treated as a clean
benchmark because Nsight instrumentation is active.

## Structural problem

The historical correctness-first cached-attention kernel launches one block per
query head with a block size of **one thread**. For SmolLM2 that means nine CUDA
threads total per layer. Each thread serially performs:

1. the 64-element query/key FMA chain;
2. online max/sum softmax state;
3. all 64 value/output component updates for every active KV position.

The value/output components are independent once the scalar softmax weights for
a position are known.

## Parallel-value implementation

`F32CachedAttentionDecodeParallelValue` keeps lane zero as the sole owner of the
score and online-softmax chain, preserving the same increasing-dimension score
FMA order and the same increasing-position softmax order. A 64-thread block then
updates the 64 independent value/output components in parallel, with a barrier
before advancing to the next KV position.

The implementation intentionally does **not**:

- parallel-reduce the query/key dot product;
- reassociate the score FMA chain;
- change cache layout;
- change the number of attention launches;
- alter weight representation or projection selection.

## Physical isolated gate at `kv_rows=35`

Exact candidate infrastructure head:
`1af93c133e5d81d08746de525a4e986902d6105c`.

Run context: `r2-attention-pv-20260831T195542Z`.

- NVIDIA Thor, compute capability 11.0;
- MAXN and fixed CPU/GPU/NVD/EMC clocks;
- 20 warmups, 100 measured iterations;
- query heads 9, KV heads 3, head dimension 64;
- 576 output values checked bitwise;
- bitwise equality: **true**;
- reference median: `0.19043199717998505 ms`;
- candidate median: `0.08195199817419052 ms`;
- candidate/reference latency ratio: `0.4303478374841301`;
- reference/candidate speed ratio: `2.323701696390834`.

This is isolated evidence only, not an end-to-end speed claim.

## Physical prefix-length sweep

The same fail-closed deterministic fixture was then evaluated across the active
prefix lengths below. Every tested case remained bitwise equal, and the
parallel-value path was faster at every tested length.

| KV rows | Reference median (ms) | Candidate median (ms) | Ref/candidate speed ratio |
| ---: | ---: | ---: | ---: |
| 1 | 0.01228800043463707 | 0.008224000222980976 | 1.494163436462433 |
| 2 | 0.016416000202298164 | 0.010304000228643417 | 1.5931676861442994 |
| 4 | 0.026688000187277794 | 0.014399999752640724 | 1.8533333781747914 |
| 8 | 0.049056001007556915 | 0.0225600004196167 | 2.1744680893224198 |
| 16 | 0.09016000106930733 | 0.04095999896526337 | 2.201171956712417 |
| 24 | 0.13312000036239624 | 0.05777600035071373 | 2.3040708867752517 |
| 35 | 0.19041600078344345 | 0.08188799768686295 | 2.325322466811169 |

No threshold is therefore introduced into the plan. The physically qualified
geometry is simply 64 threads per query head, and runtime integration fails
closed if a model head dimension differs from 64.

## End-to-end ABBA gate and promotion

Exact integration head and physical gate head:
`befc70790485385bce81b33eb4956fc7de3984f9`.

Run context: `r2-attention-e2e-20260831T204137Z`.

Two A/B/B/A rounds used the pinned SmolLM2 checkpoint, 32 greedy decode steps,
2 warmups and 5 measured iterations per run. A and B were identical except for
the explicit attention axis:

- A: E1.1 all-f32 LM-head GEMV64 + serial cached attention;
- B: E1.1 all-f32 LM-head GEMV64 + parallel-value64 cached attention.

Measured aggregate result:

- parent generation median: `688.191749 ms`;
- parallel-value generation median: `597.418587 ms`;
- generation latency reduction: `13.190097401182288%`;
- generation throughput gain: `15.19423131038271%`;
- parent request median: `692.670355 ms`;
- parallel-value request median: `601.6822599999999 ms`;
- request latency reduction: `13.135843672709224%`;
- request throughput gain: `15.1222831465897%`;
- all four candidate generation medians were below all four parent medians;
- minimum parent minus maximum candidate generation separation: `86.597579 ms`;
- complete greedy trajectory: identical;
- environment fingerprint: compatible;
- tracked worktree: clean.

This is a credible end-to-end improvement relative to the observed ABBA
variation. The plan is therefore **promoted for `MinLatency`** in the sovereignty
roadmap.

## Explicit plan boundary

`F32AttentionPlan` v1 remains a separate execution-policy axis. Existing
constructors retain baseline serial attention. Callers must explicitly select
the 64-thread parallel-value plan; this promotion does not silently change the
global runtime default.

The existing constructor name `r2_parallel_value_candidate()` is retained for
API/schema compatibility. Its name does not override the evidence-backed
promotion state recorded by the roadmap and this document.

No memory saving is claimed. The launch count is unchanged; the improvement is
from intra-block parallelism on the physically qualified `head_dim=64` path.
