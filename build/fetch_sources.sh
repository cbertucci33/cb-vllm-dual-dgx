#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 EXACT_VLLM_SOURCE_COMMIT EMPTY_CHECKOUT_DIR" >&2
  exit 2
fi

vllm_commit=$1
out=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/versions.env"
if [[ ! $vllm_commit =~ ^[0-9a-f]{40}$ ]]; then
  echo "vLLM source commit must be a full 40-character commit ID" >&2
  exit 2
fi
mkdir -p "$out"
if find "$out" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "checkout directory must be empty: $out" >&2
  exit 1
fi

fetch() {
  local name=$1 url=$2 commit=$3
  git init --quiet "$out/$name"
  git -C "$out/$name" remote add origin "$url"
  git -C "$out/$name" fetch --quiet --depth 1 origin "$commit"
  git -C "$out/$name" checkout --quiet --detach FETCH_HEAD
  local actual
  actual=$(git -C "$out/$name" rev-parse HEAD)
  [[ "$actual" == "$commit" ]] || {
    echo "$name checkout is $actual, expected $commit" >&2
    return 1
  }
}

fetch flashinfer https://github.com/flashinfer-ai/flashinfer.git "$FLASHINFER_COMMIT"
git -C "$out/flashinfer" submodule update --init --recursive --depth 1
fetch sparkinfer https://github.com/local-inference-lab/b12x.git "$SPARKINFER_COMMIT"
fetch exllamav3 https://github.com/turboderp-org/exllamav3.git "$EXLLAMAV3_COMMIT"
fetch b12x https://github.com/local-inference-lab/b12x.git "$B12X_COMMIT"
fetch cutlass https://github.com/NVIDIA/cutlass.git "$VLLM_CUTLASS_COMMIT"
fetch flashkda https://github.com/vllm-project/FlashKDA.git "$FLASHKDA_COMMIT"
git -C "$out/flashkda" submodule update --init --recursive --depth 1

check_submodule() {
  local repo=$1 path=$2 expected=$3 actual
  actual=$(git -C "$repo" -C "$path" rev-parse HEAD)
  [[ "$actual" == "$expected" ]] || {
    echo "$repo/$path is $actual, expected $expected" >&2
    return 1
  }
}
check_submodule "$out/flashinfer" 3rdparty/cccl "$FLASHINFER_CCCL_COMMIT"
check_submodule "$out/flashinfer" 3rdparty/cutlass "$FLASHINFER_CUTLASS_COMMIT"
check_submodule "$out/flashinfer" 3rdparty/nixl "$FLASHINFER_NIXL_COMMIT"
check_submodule "$out/flashinfer" 3rdparty/spdlog "$FLASHINFER_SPDLOG_COMMIT"
check_submodule "$out/flashkda" cutlass "$FLASHKDA_CUTLASS_COMMIT"

printf 'vLLM source commit argument: %s\n' "$vllm_commit"
printf 'pinned native checkouts verified under %s\n' "$out"
