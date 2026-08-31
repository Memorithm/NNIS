# R2 cached-attention parallel-value candidate

R2 now has a full-model CUDA profile for the promoted E1.1 parent on the
physical Jetson AGX Thor. That evidence makes cached attention a measured target
rather than a speculative fusion.

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

The current correctness-first cached-attention kernel launches one block per
query head with a block size of **one thread**. For SmolLM2 that means nine CUDA
threads total per layer. Each thread serially performs:

1. the 64-element query/key FMA chain;
2. online max/sum softmax state;
3. all 64 value/output component updates for every active KV position.

The value/output components are independent once the scalar softmax weights for
a position are known.

## Candidate

`F32CachedAttentionDecodeParallelValue` is candidate-only. It keeps lane zero as
the sole owner of the score and online-softmax chain, preserving the same
increasing-dimension score FMA order and the same increasing-position softmax
order. A 64-thread block then updates the 64 independent value/output components
in parallel, with a barrier before advancing to the next KV position.

The candidate intentionally does **not**:

- parallel-reduce the query/key dot product;
- reassociate the score FMA chain;
- change cache layout;
- change the number of attention launches;
- alter the production decoder path or any plan.

The isolated gate requires bitwise-equal f32 output for the deterministic
SmolLM2-shaped fixture before reporting performance.

## Physical isolated gate

Run only from a clean exact candidate checkout with the Thor fingerprint held
stable:

```bash
cd /root/NNIS && \
test -z "$(git status --porcelain --untracked-files=no)" && \
export NNIS_BENCH_RUN_CONTEXT_ID="r2-attention-pv-$(date -u +%Y%m%dT%H%M%SZ)" && \
export NNIS_BENCH_ENVIRONMENT_LABEL="native-thor" && \
export NNIS_ATTENTION_KV_ROWS=35 && \
export NNIS_PROFILE_WARMUPS=20 && \
export NNIS_PROFILE_ITERATIONS=100 && \
cargo run --locked --release -p nnis-bench --example cached_attention_parallel_value
```

`kv_rows=35` is the first gate because it matches the longest active prefix in
the qualifying 3+32-token profile. Longer-context sweeps may follow only after
this exact target passes correctness and shows a useful isolated signal.

## Promotion boundary

An isolated win does not promote anything. If the physical gate is strong and
bitwise equal, a later PR may add an explicit versioned attention-kernel plan and
a fingerprint-compatible end-to-end ABBA campaign. The production decoder must
remain on the existing kernel until that later gate succeeds.
