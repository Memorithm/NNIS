# R2 SiLU-multiply fusion candidate

R2 has physical evidence that CUDA launch/submission overhead is measurable on
Jetson AGX Thor, but that control does not justify arbitrary decoder fusion. This
candidate targets a concrete repeated sequence in the current SmolLM2 path:

1. `activated = SiLU(gate)`;
2. `gated = activated * up`.

The candidate computes `gated = SiLU(gate) * up` in one kernel.

## Structural hypothesis

For the pinned SmolLM2-135M shape (`intermediate_size = 1536`, 30 layers), an
integrated candidate removes exactly:

- one CUDA launch per decoder layer, or 30 launches per token;
- one explicit write and one explicit read of the 1536-element `activated`
  intermediate per layer;
- 12,288 logical intermediate bytes per layer and 368,640 logical intermediate
  bytes per token (`1536 * 4 bytes * 2 * 30`);
- the 1536-element `activated` f32 workspace allocation from each fused session,
  equal to 6,144 logical bytes before allocator alignment or other overhead.

The traffic byte count is logical buffer traffic, not measured DRAM traffic.
Cache behavior may change the physical memory traffic, and CUDA free-memory
observations must not be substituted for the logical workspace accounting.

## Physical isolated gate

The isolated candidate was executed on the Jetson AGX Thor in MAXN with fixed
CPU/GPU/EMC clocks, no competing CUDA process and one explicit environment
fingerprint:

- exact commit: `143a5d5870cebb539b1ab68de0e3025755dcc26a`;
- run context: `r2-silu-fusion-20260831T174722Z`;
- elements: 1536;
- warmups: 20;
- measured iterations: 100;
- reference median: `0.007215999998152256 ms`;
- fused candidate median: `0.004832000005990267 ms`;
- candidate/reference latency ratio: `0.6696230608685642`;
- reference/candidate median speed ratio: `1.49337748120996`;
- all 1536 f32 outputs: bitwise identical.

This is approximately a 33.04% reduction in the isolated sequence median. It is
strong enough to justify an explicit runtime candidate and an end-to-end gate.
It is **not** an end-to-end decoder speedup claim.

## Runtime integration boundary

The next stage adds a versioned `F32FusionPlan` separate from projection and
weight-representation plans. The historical separate two-launch path remains the
default. The R2 candidate is explicit, pinned to the physically evaluated block
size 256, and removes the `activated` session workspace only when selected.

The generic SmolLM2 report comparator must fail closed when fusion plans differ.
A dedicated A/B/B/A campaign is used because the fusion-plan difference is the
intentional experimental variable.

## End-to-end gate

The end-to-end parent and candidate keep all other axes identical:

- projection: promoted E1.1 all-f32 LM-head GEMV64;
- representation: all f32;
- A: separate SiLU then multiply;
- B: fused SiLU-multiply block 256;
- same pinned SmolLM2 fixture, prompt, decode length and benchmark parameters;
- exact-head clean-worktree evidence;
- one fingerprint-compatible run context;
- identical greedy token trajectory required.

Run the dedicated gate documented in `docs/R2_SILU_MULTIPLY_FUSION_E2E.md`.
Even a successful ABBA does not silently change the default runtime: promotion
requires interpreting the end-to-end margin relative to observed variation and
recording the decision in the sovereignty roadmap.
