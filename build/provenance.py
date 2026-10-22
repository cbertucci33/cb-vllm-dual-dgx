#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Emit source and toolchain provenance for the unified runner image."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    if len(sys.argv) != 6:
        raise SystemExit(
            "usage: provenance.py OUTPUT SOURCE_COMMIT BUILD_BASE FINAL_BASE CUDA_DEVEL"
        )
    output = Path(sys.argv[1]).resolve()
    expected = sys.argv[2]
    build_base_image, final_base_image, cuda_devel_image = sys.argv[3:]
    versions: dict[str, str] = {}
    for raw in (repo / "build/versions.env").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            versions[key] = value
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != expected:
        raise SystemExit(
            f"source HEAD is {commit}; requested source commit is {expected}"
        )
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    if status:
        raise SystemExit("source tree must be clean, including untracked files")
    diff = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--binary", "HEAD"],
    )
    source_inputs = [
        repo / ".dockerignore",
        repo / ".gitignore",
        repo / "CMakeLists.txt",
        repo / "docker/Dockerfile",
        repo / "requirements/cuda.txt",
        repo / "requirements/common.txt",
        repo / "build/versions.env",
        repo / "build/media-requirements.txt",
        repo / "build/assets/glm53/chat_template.jinja",
        repo / "build/fetch_sources.sh",
        repo / "build/verify_runtime.py",
        repo / "build/build_flashinfer_topk.py",
        repo / "build/reconcile_torch_nccl_metadata.py",
        repo / "build/build_instanttensor.py",
        repo / "build/patches/instanttensor-0.2.2-cleanup.patch",
        repo / "build/patches/instanttensor-0.2.2-startup-safety.patch",
        repo / "build/patches/exllamav3-startup-safety.patch",
        repo / "build/patches/exllamav3-compiler-memory.patch",
        repo / "build/patches/exllamav3-aarch64.patch",
        repo / "build/patches/sparkinfer-cutlass-dsl.patch",
        repo / "build/patches/sparkinfer-trellis-swiglu-limit.patch",
        repo / "build/patches/sparkinfer-trellis-barrier-order.patch",
        repo / "build/patches/sparkinfer-small-prefix-histogram.patch",
        repo / "build/patches/flashinfer-sm120-memory-order.patch",
        repo / "build/patches/flashinfer-glm53-compact-cache.patch",
        repo / "build/patches/flashinfer-sparse-invalid-kv.patch",
        repo / "build/tests/test_flashinfer_sparse_invalid_kv.py",
        repo / "build/tests/flashinfer_compact_cache_contract.cu",
        repo / "build/tests/test_flashinfer_compact_cache.py",
        repo / "build/patches/transformers-glm5next-temporal-budget.patch",
        repo / "build/patches/b12x-cutlass-dsl.patch",
        repo / "build/profiles/nvidia.gb10.48sm.json.gz",
        repo / "build/profiles/nvidia.gb10.48sm.evidence.json.gz",
        repo / "build/provenance.py",
        repo / "build/Dockerfile.runner",
        repo / "build/build_image.sh",
    ]
    result = {
        "schema_version": 1,
        "source": {"vllm_version": versions["VLLM_VERSION"], "vllm_commit": commit},
        "images": {
            "official_source_build": (
                f"vllm/vllm-openai:v{versions['VLLM_VERSION']} (built from source)"
            ),
            "build_base": build_base_image,
            "final_base": final_base_image,
            "cuda_devel": cuda_devel_image,
        },
        "platform": {
            "os": "linux",
            "architecture": "arm64",
            "cuda_arch": versions["TORCH_CUDA_ARCH_LIST"],
            "flashinfer_cuda_arch": versions["FLASHINFER_CUDA_ARCH_LIST"],
            "vllm_dgx_sm121_only": versions["VLLM_DGX_SM121_ONLY"] == "ON",
        },
        "versions": {
            key: value
            for key, value in versions.items()
            if key.endswith("_VERSION")
            or key
            in {
                "CUDA_VERSION",
                "PYTHON_VERSION",
                "TORCH_VERSION",
                "NCCL_VERSION",
                "TRITON_VERSION",
            }
        },
        "native_revisions": {
            key.removesuffix("_COMMIT").lower(): value
            for key, value in versions.items()
            if key.endswith("_COMMIT")
        },
        "input_sha256": {
            str(path.relative_to(repo)): digest(path) for path in source_inputs
        },
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "build_environment": {
            "max_jobs": versions["MAX_JOBS"],
            "vllm_core_max_jobs": versions["VLLM_CORE_MAX_JOBS"],
            "nvcc_threads": versions["NVCC_THREADS"],
        },
        "model_artifacts": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
