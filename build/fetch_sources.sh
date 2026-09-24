#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

output_dir=$(realpath -m "$1")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/versions.env"

mkdir -p "$output_dir"
if find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "source checkout directory is not empty: $output_dir" >&2
  exit 1
fi

fetch_commit() {
  local url=$1
  local commit=$2
  local destination=$3
  git init --quiet "$destination"
  git -C "$destination" remote add origin "$url"
  git -C "$destination" fetch --quiet --depth 1 origin "$commit"
  git -C "$destination" checkout --quiet --detach FETCH_HEAD
  [[ $(git -C "$destination" rev-parse HEAD) == "$commit" ]]
}

fetch_commit https://github.com/flashinfer-ai/flashinfer.git \
  "$FLASHINFER_COMMIT" "$output_dir/flashinfer"
git -C "$output_dir/flashinfer" submodule update --init --recursive --depth 1

fetch_commit https://github.com/local-inference-lab/b12x.git \
  "$SPARKINFER_COMMIT" "$output_dir/sparkinfer"
fetch_commit https://github.com/turboderp-org/exllamav3.git \
  "$EXLLAMAV3_COMMIT" "$output_dir/exllamav3"
fetch_commit https://github.com/local-inference-lab/b12x.git \
  "$B12X_COMMIT" "$output_dir/b12x"
fetch_commit https://github.com/NVIDIA/cutlass.git \
  "$VLLM_CUTLASS_COMMIT" "$output_dir/cutlass"

declare -A flashinfer_submodules=(
  [3rdparty/cccl]="$FLASHINFER_CCCL_COMMIT"
  [3rdparty/cutlass]="$FLASHINFER_CUTLASS_COMMIT"
  [3rdparty/spdlog]="$FLASHINFER_SPDLOG_COMMIT"
)
for path in "${!flashinfer_submodules[@]}"; do
  actual=$(git -C "$output_dir/flashinfer/$path" rev-parse HEAD)
  expected=${flashinfer_submodules[$path]}
  if [[ $actual != "$expected" ]]; then
    echo "FlashInfer submodule $path is $actual, expected $expected" >&2
    exit 1
  fi
done

echo "fetched all pinned source checkouts into $output_dir"
