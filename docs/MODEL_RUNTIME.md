# NNIS decoder model runtime

This document describes the first model-runtime execution contract. It is a
correctness contract, not a performance claim and not a Hugging Face format
compatibility claim.

## Supported execution shape

The model representation can describe `f32` and packed-bf16 device tensors. The
first executable decoder path is intentionally narrow but supports both standard
multi-head attention and grouped-query attention:

- decoder-only pre-norm transformer;
- `f32` weights and activations;
- query-head count equal to or an integer multiple of KV-head count;
- even rotary head dimension;
- per-channel RMSNorm;
- Llama-style rotate-half RoPE;
- causal autoregressive attention through the owned KV cache;
- SiLU/SwiGLU MLP;
- deterministic greedy generation.

Unsupported combinations fail explicitly rather than being silently coerced.

## One-stream ownership model

`Model` owns immutable weights and compiled kernel families. Each
`InferenceSession` owns one CUDA stream, one device-resident KV cache and all
mutable activation/scratch buffers used by that sequence.

Safe session methods require `&mut self`. Internally they may call NNIS's
`unsafe enqueue_*` operations, but only under these conditions:

1. every referenced allocation is owned by the model, the session or the
   current safe call;
2. all dependent GPU operations are submitted to the session's single stream;
3. later writes to a buffer are ordered after all earlier reads on that same
   stream;
4. no buffer or cache is exposed to unsynchronized host or other-stream access
   while the graph is outstanding;
5. KV append handles retain the source and destination allocations through the
   append completion event;
6. safe method boundaries synchronize before returning host-visible results or
   releasing call-owned host memory.

The low-level rule that an async buffer must not be modified before dependent
work completes therefore means *no unordered modification*. A later operation
on the same stream is an ordered dependency and is allowed by the high-level
session ownership discipline.

## Decoder token pipeline

One token is processed as:

```text
embedding gather
-> weighted RMSNorm
-> Q/K/V projections
-> position-aware RoPE on Q/K
-> append K/V into device-resident KV cache
-> cached causal attention
-> output projection
-> residual
-> weighted RMSNorm
-> gate/up projections
-> SiLU(gate) * up
-> down projection
-> residual
-> final RMSNorm
-> LM-head projection
```

Prefill uploads input IDs once and selects successive prompt IDs on device.
Fixed-length greedy generation performs top-1 on device, records the generated
ID on device, and feeds the same ID into the next embedding lookup. There is no
host roundtrip between transformer stages. When EOS stopping is requested, the
runtime copies only the one top-1 token ID to the host at each generation step
to decide whether to stop; activations, weights and KV state remain on device.
The EOS token itself is processed before the session returns, so `position()`
continues to describe the complete generated sequence length.

## KV cache

The cache layout is:

```text
[layer][head][capacity][head_dim]
```

Each layer has an independent logical length. Appending one token copies only
that token's K/V suffix into the reserved capacity; existing cache data is not
recopied. Overflow is an explicit error. `reset` changes logical lengths and
reuses the allocations without requiring a whole-cache clear.

## Explicit model directory format

The first loader consumes an NNIS-specific directory. `model.json` contains a
versioned manifest plus tensor metadata. Tensor files contain raw little-endian
`f32` or packed-bf16 words. Paths must be relative and cannot contain parent
traversal.

Required canonical tensor names are:

```text
token_embedding
layers.N.input_norm
layers.N.q_proj
layers.N.k_proj
layers.N.v_proj
layers.N.o_proj
layers.N.post_attention_norm
layers.N.gate_proj
layers.N.up_proj
layers.N.down_proj
final_norm
lm_head
```

Projection matrices are stored in NNIS's internal row-major GEMM orientation:
`[input_width, output_width]`. A checkpoint-specific converter is responsible
for transposing source-framework weights when necessary. The generic runtime
does not advertise direct Safetensors or Hugging Face loading.

## Validation boundary

CPU CI proves Rust formatting, compilation, strict Clippy and CPU-visible unit
tests. Tests named `*_on_gpu` skip when CUDA is unavailable, so a normal hosted
CI success is not evidence of GPU execution. The release-quality model claim
requires a separate `NNIS_REQUIRE_GPU=1` exact-SHA run and reference-logit
comparison on actual CUDA hardware.
