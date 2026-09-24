#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 FLASHINFER_SOURCE_DIR OUTPUT_DIR" >&2
  exit 2
fi

source_dir=$(realpath "$1")
output_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/versions.env"

if command -v git >/dev/null 2>&1; then
  actual_commit=$(git -C "$source_dir" rev-parse HEAD)
  if [[ -n $(git -C "$source_dir" status --porcelain) ]]; then
    echo "FlashInfer source is not clean: $source_dir" >&2
    exit 1
  fi
else
  actual_commit=${FLASHINFER_SOURCE_COMMIT:-}
fi
if [[ $actual_commit != "$FLASHINFER_COMMIT" ]]; then
  echo "FlashInfer source is $actual_commit, expected $FLASHINFER_COMMIT" >&2
  exit 1
fi

work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
mkdir -p "$output_dir"

export FLASHINFER_CUDA_ARCH_LIST
export FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_NVCC_THREADS="$NVCC_THREADS"
export FLASHINFER_WORKSPACE_BASE="$work_dir/workspace"
export MAX_JOBS=1
export PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}"

SOURCE_DIR="$source_dir" OUTPUT_DIR="$output_dir" /usr/bin/python3 - <<'PY'
import os
import shutil
from pathlib import Path

from flashinfer.jit import build_jit_specs
from flashinfer.jit import env as jit_env
from flashinfer.jit.topk import gen_topk_module

source = Path(os.environ["SOURCE_DIR"])
output = Path(os.environ["OUTPUT_DIR"])
build = Path(os.environ["FLASHINFER_WORKSPACE_BASE"]) / "topk-build"

jit_env.FLASHINFER_CSRC_DIR = source / "csrc"
jit_env.FLASHINFER_INCLUDE_DIR = source / "include"
jit_env.CUTLASS_INCLUDE_DIRS = [
    source / "3rdparty/cutlass/include",
    source / "3rdparty/cutlass/tools/util/include",
]
jit_env.SPDLOG_INCLUDE_DIR = source / "3rdparty/spdlog/include"
jit_env.CCCL_INCLUDE_DIRS = [
    source / "3rdparty/cccl/cub",
    source / "3rdparty/cccl/libcudacxx/include",
    source / "3rdparty/cccl/thrust",
]
jit_env.FLASHINFER_WORKSPACE_DIR = build
jit_env.FLASHINFER_JIT_DIR = build / "cached_ops"
jit_env.FLASHINFER_GEN_SRC_DIR = build / "generated"
jit_env.FLASHINFER_AOT_DIR = build / "empty-aot"
jit_env.FLASHINFER_JIT_DIR.mkdir(parents=True, exist_ok=True)
jit_env.FLASHINFER_GEN_SRC_DIR.mkdir(parents=True, exist_ok=True)

spec = gen_topk_module()
build_jit_specs([spec], verbose=True, skip_prebuilt=False)
if not spec.jit_library_path.is_file():
    raise RuntimeError(f"TopK build did not produce {spec.jit_library_path}")
shutil.copy2(spec.jit_library_path, output / "flashinfer-topk-sm121.so")
PY

sha256sum "$output/flashinfer-topk-sm121.so" \
  > "$output/flashinfer-topk-sm121.sha256"
