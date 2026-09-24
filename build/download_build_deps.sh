#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

output_dir=$1
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/versions.env"

mkdir -p "$output_dir"
if find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "build dependency directory is not empty: $output_dir" >&2
  exit 1
fi

packages=(
  "cmake==$CMAKE_VERSION"
  "wheel==$WHEEL_VERSION"
  "setuptools==$SETUPTOOLS_VERSION"
  "packaging==$PACKAGING_VERSION"
  "apache-tvm-ffi==$APACHE_TVM_FFI_VERSION"
  "nvidia-cutlass-dsl==$NVIDIA_CUTLASS_DSL_VERSION"
  "nvidia-cutlass-dsl-libs-base==$NVIDIA_CUTLASS_DSL_VERSION"
  "nvidia-cutlass-dsl-libs-core==$NVIDIA_CUTLASS_DSL_VERSION"
  "nvidia-cutlass-dsl-libs-cu12==$NVIDIA_CUTLASS_DSL_VERSION"
  "nvidia-cutlass-dsl-libs-cu13==$NVIDIA_CUTLASS_DSL_VERSION"
)
for package in "${packages[@]}"; do
  /usr/bin/python3 -m pip download --no-deps --dest "$output_dir" "$package"
done

(
  cd "$output_dir"
  sha256sum ./*.whl > build-deps-sha256.txt
)
