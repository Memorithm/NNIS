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

cargo check --locked -p nnis-cpu --all-targets
cargo test --locked -p nnis-cpu --all-targets
cargo run --locked -p nnis-cpu --example portable_buffers
cargo run --locked -p nnis-cpu --example portable_f32_graph

echo "nnis-cpu portable dependency boundary: OK"
