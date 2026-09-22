#!/usr/bin/env bash
set -euo pipefail

# Include test/example/build dependencies: a normal-only tree could hide a
# vendor dependency in the very qualification intended to prove portability.
tree="$(cargo tree --locked -p nnis-cpu --edges normal,build,dev --prefix none)"

for forbidden in nnis-sys nnis-rt nnis-jit nnis-kernels nnis-model nnis-bench; do
  if printf '%s\n' "$tree" | grep -Eq "^$forbidden( |$)"; then
    echo "nnis-cpu portable boundary depends on forbidden crate: $forbidden" >&2
    exit 1
  fi
done

cargo check --locked -p nnis-core -p nnis-cpu --all-targets
cargo test --locked -p nnis-core -p nnis-cpu --all-targets
cargo test --locked -p nnis-core --release --test graph_contract
cargo test --locked -p nnis-cpu --release --test numerical_reference --test graph_execution
RUSTDOCFLAGS="${RUSTDOCFLAGS:-} -D warnings" cargo doc --locked -p nnis-core -p nnis-cpu --no-deps
cargo run --locked -p nnis-cpu --example portable_buffers
cargo run --locked -p nnis-cpu --example portable_f32_graph
cargo run --locked -p nnis-cpu --example portable_graph_plan

echo "nnis-cpu portable dependency boundary: OK"
