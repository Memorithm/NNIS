# Portable cross-layer KV reuse-plan consumer v1 (DSV41-2)

Status: software consumer contract, backend-neutral core plus CPU binding.
Roadmap item `NNIS_DSV41_2_cross_layer_KV_reuse_plan_consumer_without_policy_invention`
of `deepseek_v41_kv_runtime_program_2026_09_24` (off-main sovereignty roadmap).
Builds on DSV41-0 (`nnis_core::replay_state`, #226/#227) and the DSV41-1 CPU
recent-window replay reference (`nnis_cpu::replay`, #230).

## Ownership

The model/domain owner supplies the reuse plan. NNIS validates and executes
it; it never infers, extends, chains, or repairs reuse, and it does not pick
which layers share.

## `nnis-core::kv_reuse_plan`

- `CrossLayerKvReusePlanV1::new(plan_id, model_id, plan_schema_version, layers)`
  where each `KvLayerReuseV1 { key, value }` is `Own` or `ReuseFrom(layer)`.
- Validation (fail closed, never rewritten):
  - canonical, non-empty, at most 128-byte `plan_id` and `model_id`;
  - non-zero plan schema version; 1..=4096 layers;
  - `ReuseFrom(p)` requires `p` strictly earlier than the consuming layer
    (`NonCausalReuse` otherwise, including layer 0 and self reuse);
  - the producer must own the same component (`ChainedReuse` otherwise), so
    chains and key/value cross-reuse are never inferred.
- Key and value are resolved independently; asymmetry exists only when the
  plan declares it.
- Queries: `owner(layer, component)`, `owning_layers(component)`,
  `owned_count(component)` (a logical count), `layer(layer)`.

## `nnis-cpu::kv_reuse`

- `CpuCrossLayerKvBindingV1::new(plan, key_sources, value_sources)` binds
  exactly one `CpuReplaySourceV1` to each owning (layer, component) pair.
  Missing, duplicate, or undeclared (reusing-layer) sources fail closed, and
  all bound sources must share one logical position range.
- `source`, `replay_window`, and `replay_recent_window` resolve a layer to its
  declared owner and reuse the DSV41-1 identity-checked bit-exact replay.

## Evidence

Host-only unit tests in both modules cover asymmetric resolution, no-reuse
plans, self/forward/first-layer reuse, chained and cross-component reuse,
identity/version/size bounds, out-of-range queries, bit-exact owner reads for
reusing layers, foreign-identity requests, missing/duplicate/undeclared
sources, and range mismatch. No GPU, model, or physical run was performed.

## Claim boundary

- Logical sharing only: fewer bound sources is not physical memory release or a
  memory-saving result.
- No quality, latency, throughput, or model-support claim; no DeepSeek
  equivalence claim; no plan is shipped for any real model.
- No WGPU or CUDA execution; FP4/INT4 KV formats are out of scope (DSV41-3).
