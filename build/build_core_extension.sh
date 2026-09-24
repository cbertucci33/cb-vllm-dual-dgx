#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SOURCE_ARCHIVE_DIR BUILD_DEPS_DIR OUTPUT_DIR" >&2
  exit 2
fi

source_dir=$1
build_deps_dir=$2
output_dir=$3
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/versions.env"

python_bin=/usr/bin/python3
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
mkdir -p "$output_dir"

(cd "$source_dir" && sha256sum -c source-sha256.txt)
tar -xzf "$source_dir"/vllm-*.tar.gz -C "$work_dir"
tar -xzf "$source_dir"/cutlass-*.tar.gz -C "$work_dir"
(cd "$build_deps_dir" && sha256sum -c build-deps-sha256.txt)

"$python_bin" -m pip install --no-deps "$build_deps_dir"/cmake-*.whl
python_path=$(
  "$python_bin" -c 'import os, sys; print(os.pathsep.join(sys.path))'
)
export TORCH_CUDA_ARCH_LIST MAX_JOBS NVCC_THREADS
export CMAKE_BUILD_PARALLEL_LEVEL=$VLLM_CORE_MAX_JOBS

cmake -S "$work_dir/vllm" -B "$work_dir/vllm-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_TARGET_DEVICE=cuda \
  -DVLLM_PYTHON_EXECUTABLE="$python_bin" \
  -DVLLM_PYTHON_PATH="$python_path" \
  -DVLLM_CUTLASS_SRC_DIR="$work_dir/cutlass" \
  -DVLLM_BUILD_CORE_EXTENSION_ONLY=ON \
  -DNVCC_THREADS="$NVCC_THREADS" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build "$work_dir/vllm-build" --target _C_stable_libtorch \
  --parallel "$VLLM_CORE_MAX_JOBS"

mapfile -t core_extensions < <(
  find "$work_dir/vllm-build" -type f -name '_C_stable_libtorch*.so'
)
if [[ ${#core_extensions[@]} -ne 1 ]]; then
  echo "expected one built vLLM core extension, found ${#core_extensions[@]}" >&2
  exit 1
fi
cp "${core_extensions[0]}" "$output_dir/"
