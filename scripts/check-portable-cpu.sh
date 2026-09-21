#!/usr/bin/env bash
set -euo pipefail

tree="$(cargo tree -p nnis-cpu --edges normal --prefix none)"

for forbidden in nnis-sys nnis-rt nnis-jit nnis-kernels nnis-model nnis-bench; do
  if printf '%s\n' "$tree" | grep -Eq "^$forbidden( |$)"; then
    echo "nnis-cpu portable boundary depends on forbidden crate: $forbidden" >&2
    exit 1
  fi
done

cargo check -p nnis-cpu --all-targets
cargo test -p nnis-cpu --all-targets

echo "nnis-cpu portable dependency boundary: OK"
