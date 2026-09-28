#!/usr/bin/env bash
set -euo pipefail

# nnis-wgpu is the only crate allowed to depend on wgpu. It must not reach
# vendor-specific NNIS crates or CUDA/NVML bindings in any edge kind.
tree="$(cargo tree --locked -p nnis-wgpu --edges normal,build,dev --prefix none)"

for forbidden in nnis-sys nnis-rt nnis-jit nnis-kernels nnis-model nnis-bench cudarc cust nvml-wrapper nvml-wrapper-sys; do
  if printf '%s\n' "$tree" | grep -Eq "^$forbidden( |$)"; then
    echo "nnis-wgpu portable boundary depends on forbidden crate: $forbidden" >&2
    exit 1
  fi
done

cargo check --locked -p nnis-wgpu --all-targets
cargo test --locked -p nnis-wgpu --test capabilities
# Adapter-dependent test: logs an explicit SKIP when no adapter exists (CI has
# no GPU). A pass without an adapter, or on a software adapter, is not
# hardware evidence.
cargo test --locked -p nnis-wgpu --test adapter_add_f32 -- --nocapture
cargo test --locked -p nnis-wgpu --test portable_memory -- --nocapture
cargo test --locked -p nnis-wgpu --test numerical_parity -- --nocapture
cargo test --locked -p nnis-wgpu --test graph_parity -- --nocapture
cargo test --locked -p nnis-wgpu --test replay_parity -- --nocapture
cargo test --locked -p nnis-wgpu --test fp4_decode_parity -- --nocapture
cargo test --locked -p nnis-wgpu --test speculative_parity -- --nocapture
RUSTDOCFLAGS="${RUSTDOCFLAGS:-} -D warnings" cargo doc --locked -p nnis-wgpu --no-deps

echo "nnis-wgpu portable dependency boundary: OK"
