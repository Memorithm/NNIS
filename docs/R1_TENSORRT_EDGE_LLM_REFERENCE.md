# R1 TensorRT Edge-LLM reference on Jetson AGX Thor

R1 establishes a trusted NVIDIA runtime reference for NNIS on the physical
Jetson AGX Thor used by the SmolLM2 qualification campaign.

The qualified reference is **TensorRT Edge-LLM v0.10.0**, pinned to upstream
commit `71dd1bae032e70771265917ec74d3ff4cad07a10`.

This document records what has been demonstrated and, equally importantly, what
has **not** been demonstrated.

## Why Edge-LLM is the Thor reference

The locally installed TensorRT-LLM environment was not accepted as a clean
reference: it was an editable install backed by a dirty local source checkout,
its import path contained conflicting TensorRT Python distributions, and the
runtime import failed before model execution.

TensorRT Edge-LLM v0.10.0 was instead checked out cleanly and built on the Thor
against the JetPack TensorRT installation. The NNIS runtime remains independent
of TensorRT Edge-LLM; this is a qualification reference only.

## Exact source-model identity

Both runtimes derive from the same pinned source checkpoint:

- repository: `HuggingFaceTB/SmolLM2-135M`;
- revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`;
- `model.safetensors` SHA-256:
  `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`;
- source weight dtype: BF16;
- tokenizer SHA-256:
  `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`.

The tokenizer file used by Edge-LLM was byte-identical to the tokenizer already
qualified by the NNIS SmolLM2 fixture.

The checkpoint resolves as `LlamaForCausalLM` / `model_type=llama`, with:

- hidden size `576`;
- intermediate size `1536`;
- 30 decoder layers;
- 9 query heads;
- 3 KV heads;
- head dimension `64`;
- vocabulary size `49152`;
- tied input/output embeddings.

This successful execution is evidence for this **exact checkpoint and workload**.
It is not a claim that NNIS or Edge-LLM generally supports every Llama-family
checkpoint.

## Edge-LLM engine

The v0.10.0 direct builder accepted the exact checkpoint and produced an engine
with the following explicit build boundary:

```text
dense mode:               fp16
max batch size:           1
max input length:         64
max KV-cache capacity:    128
engine size:              5,066,980 bytes
runtime weight bindings:  273
binding destination dtype F16
```

The source checkpoint remains BF16. Edge-LLM converts/binds the runtime weights
as F16 for this engine path.

## Exact semantic workload

The reference workload is:

```text
prompt text:     "Gravity is"
prompt IDs:      [22007, 6463, 314]
batch size:      1
decode steps:    32
sampling:        temperature=0, top_k=1
chat template:   disabled
generation add:  disabled
context reuse:   bypassed
```

The prompt round-trips through the pinned tokenizer exactly to the three NNIS
prompt IDs.

The Edge-LLM generated trajectory was:

```text
[260, 3075, 338, 6650, 260, 2591, 284, 260,
 8872, 1592, 30, 198, 198, 504, 8872, 314,
 253, 8304, 282, 260, 2591, 30, 657, 314,
 253, 19284, 1248, 338, 21837, 260, 2591, 30]
```

It matched **32/32** tokens in the qualified NNIS R2 campaign
`r2-attention-e2e-20260831T204137Z`. The R2 summary plus all eight A/B reports
contained that same trajectory.

This is exact greedy-trajectory equality for one qualified workload. It is **not**
bitwise-logit equality or general numerical equivalence.

## Physical reference-performance campaign

The autonomous Edge-LLM reference campaign used five independent processes with
two warmup requests per process under the following physical regime:

- Jetson AGX Thor Developer Kit;
- JetPack `7.1-b112`;
- CUDA `13.0`;
- TensorRT `10.13.3.9`;
- MAXN;
- CPU fixed at `2.601 GHz`;
- GPU GPC fixed at `1.575 GHz`;
- GPU NVD fixed at `1.692 GHz`;
- EMC fixed at `4.266 GHz`;
- no competing CUDA process at campaign start.

All five measured processes reproduced the same semantic output.

| Metric | Median | Min | Max |
| --- | ---: | ---: | ---: |
| Prefill GPU time | `7.090816 ms` | `6.538080 ms` | `10.778720 ms` |
| Generation throughput, NVIDIA definition | `350.212463 tok/s` | `344.463806` | `357.621155` |
| Generation stage total GPU time | `91.373108 ms` | `89.480164 ms` | `92.898003 ms` |
| Generation stage median decode step | `2.920064 ms` | `2.858208 ms` | `2.935584 ms` |
| Peak unified memory | `657,317,888 B` | `654,647,296 B` | `658,460,672 B` |

### NVIDIA throughput definition

Edge-LLM reports 32 generated tokens but 31 `llm_generation` GPU-stage
executions because the first generated token is produced by the prefill pass.
Its native generation throughput is therefore:

```text
32 generated tokens / cumulative GPU time of the 31 llm_generation stages
```

The committed evidence preserves that definition verbatim. It must not be
silently substituted for another serving-throughput definition.

## Cross-runtime comparison is deliberately blocked

The current qualified NNIS baseline reports logical F32 execution weights while
this Edge-LLM engine binds weights as F16. The memory observations are also not
the same metric: Edge-LLM reports process peak unified memory/RSS on the iGPU.

Therefore the committed evidence sets both of these to `false`:

```text
cross_runtime_speed_comparison_allowed
cross_runtime_memory_comparison_allowed
```

No NNIS/Edge-LLM speedup, slowdown, performance parity, or memory ratio is
justified by this R1 evidence. R1 first establishes a reproducible NVIDIA
reference line. A comparative performance claim requires a later experiment
with explicitly aligned precision/representation and metric semantics.

## Machine-readable evidence

The versioned dossier is:

```text
evidence/r1_tensorrt_edge_llm_v0_10_0_smollm2_thor.json
```

Validate it with:

```bash
python3 tools/validate_external_reference_evidence.py \
  evidence/r1_tensorrt_edge_llm_v0_10_0_smollm2_thor.json
```

The validator fails closed on schema drift, malformed provenance, trajectory
mismatch, campaign aggregation drift, throughput-definition drift, or an unsafe
cross-runtime comparability flag.
