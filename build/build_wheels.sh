#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 SOURCE_ARCHIVE_DIR FLASHINFER_WHEEL BUILD_DEPS_DIR OUTPUT_DIR" >&2
  exit 2
fi

source_dir=$1
flashinfer_wheel=$2
build_deps_dir=$3
output_dir=$4
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/versions.env"

python_bin=/usr/bin/python3
work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
mkdir -p "$output_dir"

(cd "$source_dir" && sha256sum -c source-sha256.txt)
tar -xzf "$source_dir"/sparkinfer-*.tar.gz -C "$work_dir"
tar -xzf "$source_dir"/exllamav3-*.tar.gz -C "$work_dir"
tar -xzf "$source_dir"/b12x-*.tar.gz -C "$work_dir"

(cd "$work_dir/sparkinfer" && patch -p1 < "$source_dir/sparkinfer-cutlass-dsl.patch")
(cd "$work_dir/exllamav3" && patch -p1 < "$source_dir/exllamav3-aarch64.patch")
(cd "$work_dir/b12x" && patch -p1 < "$source_dir/b12x-cutlass-dsl.patch")

flashinfer_manifest=$(dirname "$flashinfer_wheel")/flashinfer-wheel.sha256
if [[ ! -f $flashinfer_manifest ]]; then
  echo "FlashInfer wheel manifest is missing: $flashinfer_manifest" >&2
  exit 1
fi
(cd "$(dirname "$flashinfer_wheel")" && sha256sum -c flashinfer-wheel.sha256)
(cd "$build_deps_dir" && sha256sum -c build-deps-sha256.txt)

export TORCH_CUDA_ARCH_LIST FLASHINFER_CUDA_ARCH_LIST MAX_JOBS NVCC_THREADS
export BUILD_NVEP=0 BUILD_NIXL_EP=0 BUILD_NCCL_EP=0 FLASHINFER_BUILD_NO_PIP=1
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS

"$python_bin" -m pip install --no-deps \
  "$build_deps_dir"/cmake-*.whl \
  "$build_deps_dir"/wheel-*.whl
"$python_bin" -m pip install --no-deps --force-reinstall \
  "$build_deps_dir"/nvidia_cutlass_dsl-*.whl \
  "$build_deps_dir"/nvidia_cutlass_dsl_libs_*.whl
cp "$build_deps_dir"/nvidia_cutlass_dsl-*.whl "$output_dir/"
cp "$build_deps_dir"/nvidia_cutlass_dsl_libs_*.whl "$output_dir/"

"$python_bin" -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir "$output_dir" "$work_dir/sparkinfer"
"$python_bin" -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir "$output_dir" "$work_dir/exllamav3"
"$python_bin" -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir "$output_dir" "$work_dir/b12x"

cp "$flashinfer_wheel" \
  "$output_dir/flashinfer_python-${FLASHINFER_VERSION}-py3-none-any.whl"
