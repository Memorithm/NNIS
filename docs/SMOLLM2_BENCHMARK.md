# SmolLM2-135M end-to-end benchmark

This benchmark is deliberately separate from NNIS kernel microbenchmarks and
from the SmolLM2 numerical-validation policy.

It answers a narrower product question: how long does fixed-length greedy
inference take on one CUDA GPU when model loading is excluded and a fresh
sequence/session is used for each measured generation?

No performance claim is valid until both backends have been run on the same
physical GPU under a recorded software environment.

## Pinned model

Both harnesses use the same trained checkpoint:

- repository: `HuggingFaceTB/SmolLM2-135M`
- revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- `model.safetensors` SHA-256:
  `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
- source weights: BF16
- execution weights for this comparison: f32
- default input IDs: `[22007, 6463, 314]` (`"Gravity is"` in the pinned tokenizer)
- the already-qualified two-token greedy prefix is `[260, 3075]`

The NNIS harness refuses a model directory whose `provenance.json` does not
match this source. The Transformers harness downloads the exact revision and
checks the model file hash before loading it.

## What is timed

The primary comparison is `generation`:

- model load is excluded;
- a fresh NNIS `InferenceSession` is created before each timed NNIS sample;
- NNIS session creation is timed and reported separately;
- the timed region includes prompt prefill plus fixed-length greedy decode;
- NNIS uses `GenerationConfig::greedy`, so token selection and token feedback
  remain device-resident during the graph;
- the Transformers harness uses `use_cache=True`, keeps each argmax token on
  CUDA, feeds it back without a per-step `.item()`, and synchronizes CUDA only
  at timing boundaries;
- both paths intentionally execute the final generated token through the
  decoder, matching NNIS fixed-length generation semantics;
- all samples are host wall-clock durations around synchronized end-to-end
  operations. They must not be described as individual CUDA-kernel latency.

The reported `generated_tokens_per_second_median` is:

```text
decode_steps / median_generation_seconds
```

Because the timed region also includes prefill, this is an end-to-end generated
throughput metric for the exact configured prompt and decode length. It is not
an isolated steady-state decode-token rate.

## NNIS

First produce the pinned model directory with `tools/smollm2_135m_fixture.py`
as described in `SMOLLM2_VALIDATION.md`. Then run a release build on the target
GPU:

```bash
NNIS_REQUIRE_GPU=1 \
cargo run --locked --release -p nnis-bench --example smollm2_e2e -- \
  --model /tmp/smollm2-135m/model \
  --device 0 \
  --decode-steps 32 \
  --warmups 2 \
  --iterations 5 \
  > /tmp/nnis-smollm2-e2e.json
```

The report contains:

- the NNIS git commit and dirty state;
- GPU identity, compute capability and driver/NVRTC versions;
- model shape and exact checkpoint provenance;
- free/total CUDA memory before model load, after model load and with one fresh
  session resident;
- CUDA free-memory deltas after model and session creation;
- session-construction samples;
- synchronized generation samples and distribution statistics;
- deterministic generated IDs;
- whether the qualified two-token greedy prefix was checked.

### Controlled Jetson Thor runs

Jetson AGX Thor uses dynamic power/frequency management. Benchmark evidence
intended for backend comparison should therefore retain the current power mode
and clock state. Before each compared run, capture at least:

```bash
sudo nvpmodel -q
sudo /usr/bin/jetson_clocks --show
```

For a controlled maximum-performance experiment, select the intended nvpmodel
mode before the CUDA context is created and, when appropriate for the test,
apply `jetson_clocks`. Do not silently change power mode or clock policy inside
the benchmark harness. If fixed clocks are used for one backend they must be
used for the other backend as well.

`tegrastats` may be recorded alongside longer experiments to retain GPU/EMC
frequency, thermal and power/throttling context. A throughput result obtained
under a different power/clock/thermal state is a different experiment, even on
the same physical Thor.

## Transformers CUDA baseline

Use a Python environment whose PyTorch build actually supports the target GPU.
Do not use the CPU-only environment used to construct the correctness oracle.
The script records the actual Torch, Transformers and CUDA versions instead of
pretending they are interchangeable across installations.

On Jetson AGX Thor/CUDA 13.0, keep the correctness oracle untouched and create a
separate GPU environment. One supported setup is to install the CUDA 13.0
PyTorch wheel first, then install the pinned Python-side benchmark dependencies:

```bash
python3 -m venv /tmp/nnis-smollm2-gpu-venv
source /tmp/nnis-smollm2-gpu-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r tools/requirements-smollm2-benchmark-gpu.txt
```

Verify that the installed build is really CUDA-enabled before collecting any
Transformers evidence:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
x = torch.ones(4096, device="cuda")
print(float((x * x).sum()))
PY
```

Then run:

```bash
python tools/bench_smollm2_transformers.py \
  --device 0 \
  --decode-steps 32 \
  --warmups 2 \
  --iterations 5 \
  > /tmp/transformers-smollm2-e2e.json
```

By default the script leaves Transformers' attention implementation selection
untouched and records the resolved implementation. For controlled experiments,
`--attn-implementation eager`, `sdpa`, or `flash_attention_2` may be requested,
provided the environment supports it. Results from different attention
implementations must be labeled as different benchmark configurations.

The Transformers report also captures CUDA memory before/after model load and
peak allocated/reserved memory during measured generation.

## Comparison rules

A comparison is admissible only when all of the following match or are
explicitly reported as a difference:

1. physical GPU and device ordinal;
2. exact checkpoint revision and SHA-256;
3. f32 execution weights;
4. input token IDs;
5. fixed decode length;
6. warmup and measured iteration counts;
7. no per-token host decision in either fixed-length path;
8. successful qualified greedy prefix check for the default probe;
9. clean NNIS git state, or the dirty state is explicitly retained in evidence;
10. actual Torch/Transformers/CUDA and NNIS driver/NVRTC versions are retained;
11. Jetson power mode, clock policy and relevant thermal/throttling state are
    matched or explicitly retained as a difference.

Do not compare NNIS GPU timing with the Transformers CPU reference used for
numerical validation. Do not infer quality from tokens/second, and do not infer
performance from the numerical-validation logs.

## Memory interpretation

NNIS memory snapshots use CUDA `cuMemGetInfo`. The reported free-byte deltas are
observational process/device memory signals only; they are **not** allocation
ownership, model byte size, or proof of physical residency. This distinction is
especially important on integrated/unified-memory devices such as NVIDIA Thor.
