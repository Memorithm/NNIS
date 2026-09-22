#!/usr/bin/env bash
set -euo pipefail

tree="$(cargo tree -p nnis-core --edges normal --prefix none)"

for forbidden in nnis-sys nnis-rt nnis-jit nnis-kernels nnis-model nnis-bench; do
  if printf '%s\n' "$tree" | grep -Eq "^$forbidden( |$)"; then
    echo "nnis-core portable boundary depends on forbidden crate: $forbidden" >&2
    exit 1
  fi
done

cargo check -p nnis-core --all-targets
cargo test -p nnis-core --all-targets

echo "nnis-core portable dependency boundary: OK"
