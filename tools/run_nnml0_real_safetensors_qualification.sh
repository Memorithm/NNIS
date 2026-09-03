#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${1:-/tmp/nnis-nnml0-smollm2-135m}"
SOURCE_REPO="HuggingFaceTB/SmolLM2-135M"
SOURCE_REVISION="93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
SOURCE_MODEL_SHA256="80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
BASE_URL="https://huggingface.co/${SOURCE_REPO}/resolve/${SOURCE_REVISION}"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "required command not found: $1" >&2
        exit 2
    fi
}

require_clean_worktree() {
    if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
        echo "refusing qualification from a dirty worktree" >&2
        exit 2
    fi
}

require_command git
require_command cargo
require_command curl
require_command sha256sum

cd "$ROOT"
HEAD_SHA="$(git rev-parse HEAD)"
require_clean_worktree

mkdir -p "$MODEL_DIR"

download_if_missing() {
    local name="$1"
    local destination="$MODEL_DIR/$name"
    if [[ -f "$destination" ]]; then
        return 0
    fi
    local temporary="${destination}.partial"
    rm -f "$temporary"
    curl --fail --location --retry 3 --retry-delay 2 \
        --output "$temporary" "${BASE_URL}/${name}?download=true"
    mv "$temporary" "$destination"
}

download_if_missing config.json
download_if_missing model.safetensors
require_clean_worktree

echo "${SOURCE_MODEL_SHA256}  ${MODEL_DIR}/model.safetensors" | sha256sum --check --strict

echo "NNIS_HEAD=${HEAD_SHA}"
echo "SOURCE=${SOURCE_REPO}@${SOURCE_REVISION}"
echo "MODEL_DIR=${MODEL_DIR}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
fi

cargo run --locked -p nnis-model --example nnml0_real_safetensors -- --model "$MODEL_DIR"
