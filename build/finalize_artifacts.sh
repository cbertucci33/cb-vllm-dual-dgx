#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 ARTIFACT_DIR" >&2
  exit 2
fi

artifact_dir=$1
mapfile -t core_extensions < <(
  find "$artifact_dir" -maxdepth 1 -type f -name '_C_stable_libtorch*.so'
)
if [[ ${#core_extensions[@]} -ne 1 ]]; then
  echo "expected one vLLM core extension, found ${#core_extensions[@]}" >&2
  exit 1
fi

shopt -s nullglob
wheels=("$artifact_dir"/*.whl)
if [[ ${#wheels[@]} -lt 4 ]]; then
  echo "expected pinned dependency wheels in $artifact_dir" >&2
  exit 1
fi

if [[ ! -f "$artifact_dir/flashinfer-topk-sm121.so" ]]; then
  echo "matching FlashInfer SM121 TopK module is missing" >&2
  exit 1
fi

(
  cd "$artifact_dir"
  sha256sum ./*.whl ./_C_stable_libtorch*.so \
    ./flashinfer-topk-sm121.so > artifact-sha256.txt
)
