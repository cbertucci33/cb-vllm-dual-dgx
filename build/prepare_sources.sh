#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir/.." rev-parse --show-toplevel)
source "$repo_root/build/versions.env"

if [[ $# -ne 6 ]]; then
  echo "usage: $0 FLASHINFER_REPO SPARKINFER_REPO EXLLAMAV3_REPO B12X_REPO CUTLASS_REPO OUTPUT_DIR" >&2
  exit 2
fi

flashinfer_repo=$1
sparkinfer_repo=$2
exllamav3_repo=$3
b12x_repo=$4
cutlass_repo=$5
output_dir=$6
mkdir -p "$output_dir"

require_commit() {
  local repo=$1
  local expected=$2
  local actual
  actual=$(git -C "$repo" rev-parse HEAD)
  if [[ $actual != "$expected" ]]; then
    echo "source mismatch: $repo is $actual, expected $expected" >&2
    exit 1
  fi
  if [[ -n $(git -C "$repo" status --porcelain) ]]; then
    echo "source is not clean: $repo" >&2
    exit 1
  fi
}

require_commit "$flashinfer_repo" "$FLASHINFER_COMMIT"
require_commit "$sparkinfer_repo" "$SPARKINFER_COMMIT"
require_commit "$exllamav3_repo" "$EXLLAMAV3_COMMIT"
require_commit "$b12x_repo" "$B12X_COMMIT"
require_commit "$cutlass_repo" "$VLLM_CUTLASS_COMMIT"

vllm_commit=$(git -C "$repo_root" rev-parse HEAD)
if [[ -n $(git -C "$repo_root" status --porcelain) ]]; then
  echo "vLLM source is not clean" >&2
  exit 1
fi

git -C "$repo_root" archive --format=tar.gz --prefix=vllm/ \
  -o "$output_dir/vllm-$vllm_commit.tar.gz" "$vllm_commit"
git -C "$flashinfer_repo" archive --format=tar.gz --prefix=flashinfer/ \
  -o "$output_dir/flashinfer-$FLASHINFER_COMMIT.tar.gz" "$FLASHINFER_COMMIT"
git -C "$sparkinfer_repo" archive --format=tar.gz --prefix=sparkinfer/ \
  -o "$output_dir/sparkinfer-$SPARKINFER_COMMIT.tar.gz" "$SPARKINFER_COMMIT"
git -C "$exllamav3_repo" archive --format=tar.gz --prefix=exllamav3/ \
  -o "$output_dir/exllamav3-$EXLLAMAV3_COMMIT.tar.gz" "$EXLLAMAV3_COMMIT"
git -C "$b12x_repo" archive --format=tar.gz --prefix=b12x/ \
  -o "$output_dir/b12x-$B12X_COMMIT.tar.gz" "$B12X_COMMIT"
git -C "$cutlass_repo" archive --format=tar.gz --prefix=cutlass/ \
  -o "$output_dir/cutlass-$VLLM_CUTLASS_COMMIT.tar.gz" "$VLLM_CUTLASS_COMMIT"

cp "$repo_root/build/patches/sparkinfer-cutlass-dsl.patch" "$output_dir/"
cp "$repo_root/build/patches/exllamav3-aarch64.patch" "$output_dir/"
cp "$repo_root/build/patches/b12x-cutlass-dsl.patch" "$output_dir/"
(
  cd "$output_dir"
  sha256sum ./*.tar.gz ./*.patch > source-sha256.txt
)

printf '%s\n' "$vllm_commit" > "$output_dir/vllm-commit.txt"
echo "prepared source archives for vLLM $vllm_commit"
