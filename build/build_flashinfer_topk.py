#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build FlashInfer's AOT TopK module for the pinned GB10 SM121 target."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from flashinfer.jit import build_jit_specs
from flashinfer.jit import env as jit_env
from flashinfer.jit.topk import gen_topk_module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
