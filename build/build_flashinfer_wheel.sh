#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 FLASHINFER_SOURCE_DIR BUILD_DEPS_DIR OUTPUT_DIR" >&2
  exit 2
fi

source_dir=$(realpath "$1")
build_deps_dir=$(realpath "$2")
output_dir=$(realpath -m "$3")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/versions.env"

actual_commit=$(git -C "$source_dir" rev-parse HEAD)
if [[ $actual_commit != "$FLASHINFER_COMMIT" ]]; then
  echo "FlashInfer source is $actual_commit, expected $FLASHINFER_COMMIT" >&2
  exit 1
fi
if [[ -n $(git -C "$source_dir" status --porcelain) ]]; then
  echo "FlashInfer source is not clean: $source_dir" >&2
  exit 1
fi
(cd "$build_deps_dir" && sha256sum -c build-deps-sha256.txt)

mkdir -p "$output_dir"
if find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "FlashInfer wheel directory is not empty: $output_dir" >&2
  exit 1
fi

/usr/bin/python3 -m pip install --no-deps \
  "$build_deps_dir"/setuptools-*.whl \
  "$build_deps_dir"/packaging-*.whl \
  "$build_deps_dir"/apache_tvm_ffi-*.whl \
  "$build_deps_dir"/wheel-*.whl

export BUILD_NVEP=0 BUILD_NIXL_EP=0 BUILD_NCCL_EP=0
export FLASHINFER_BUILD_NO_PIP=1
export SOURCE_DATE_EPOCH
SOURCE_DATE_EPOCH=$(git -C "$source_dir" show -s --format=%ct HEAD)

/usr/bin/python3 -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir "$output_dir" "$source_dir"

mapfile -t wheels < <(find "$output_dir" -maxdepth 1 -type f \
  -name "flashinfer_python-${FLASHINFER_VERSION}-*.whl")
if [[ ${#wheels[@]} -ne 1 ]]; then
  echo "expected one FlashInfer $FLASHINFER_VERSION wheel, found ${#wheels[@]}" >&2
  exit 1
fi
sha256sum "${wheels[0]}" > "$output_dir/flashinfer-wheel.sha256"
echo "built ${wheels[0]}"
