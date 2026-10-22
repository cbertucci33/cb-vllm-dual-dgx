# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native compiler-ordering and packaged-header ownership contracts."""

import difflib
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import regex as re

_REPO = Path(__file__).resolve().parents[3]
_ARCH = Path("include/flashinfer/attention/sparse_mla_sm120/arch")


@pytest.mark.parametrize(
    "defect", ("none", "unpatched_source", "stale_barrier", "stale_copy", "stale_topk")
)
def test_flashinfer_header_contract(tmp_path, defect):
    spec = importlib.util.spec_from_file_location(
        "runner_build_verifier", _REPO / "build/verify_runtime.py"
    )
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)

    source = tmp_path / "source"
    package = tmp_path / "package"
    patches: list[str] = []
    for name in ("barrier.cuh", "cp_async.cuh", "topk.cuh"):
        old = f"// {name}\nconstexpr int revision = 0;\n"
        new = f"// {name}\nconstexpr int revision = 1;\n"
        relative = (Path("include/flashinfer") if name == "topk.cuh" else _ARCH) / name
        src = source / relative
        installed = package / "data" / relative
        src.parent.mkdir(parents=True, exist_ok=True)
        installed.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(old if defect == "unpatched_source" else new)
        stale = (
            (defect == "stale_barrier" and name == "barrier.cuh")
            or (defect == "stale_copy" and name == "cp_async.cuh")
            or (defect == "stale_topk" and name == "topk.cuh")
        )
        installed.write_text(old if stale else new)
        patches.extend(
            difflib.unified_diff(
                old.splitlines(True),
                new.splitlines(True),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    patch = tmp_path / "memory-order.patch"
    patch.write_text("".join(patches))

    if defect == "none":
        verifier.verify_flashinfer_memory_ordering_headers(package, source, patch)
    elif defect == "unpatched_source":
        with pytest.raises(subprocess.CalledProcessError):
            verifier.verify_flashinfer_memory_ordering_headers(package, source, patch)
    else:
        with pytest.raises(RuntimeError, match="packaged header differs"):
            verifier.verify_flashinfer_memory_ordering_headers(package, source, patch)


@pytest.mark.parametrize("operation", ("wait", "publish"))
def test_flashinfer_memory_ordering_ptx(tmp_path, operation):
    """Compile the real header; no CUDA context or kernel launch is needed.

    An acquire must invalidate an earlier shared load. A release must preserve
    the store that precedes publication even if the same slot is stored again.
    These witnesses inspect optimized compiler output, not source spellings.
    """
    nvcc = os.environ.get("NVCC") or shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("CUDA compiler is required for the native PTX contract")
    include_override = os.environ.get("FLASHINFER_INCLUDE_DIR")
    if include_override:
        include = Path(include_override)
    else:
        try:
            distribution = importlib.metadata.distribution("flashinfer-python")
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("FlashInfer native headers are not installed")
        include = Path(distribution.locate_file("flashinfer/data/include"))
    header = include / _ARCH.relative_to("include") / "barrier.cuh"
    assert header.is_file(), header
    if operation == "wait":
        body = """
    payload[0] = result[0];
    asm volatile("" ::: "memory");
    float before = payload[0];
    mbarrier_wait_parity(&ready, phase);
    float after = payload[0];
    result[0] = after - before;
"""
        mnemonic = "ld"
        barrier = "mbarrier.try_wait.parity"
    else:
        body = """
    payload[0] = 1.0f;
    mbarrier_arrive(&ready);
    payload[0] = 2.0f;
    asm volatile("" ::: "memory");
    result[0] = payload[0];
"""
        mnemonic = "st"
        barrier = "mbarrier.arrive.shared"
    code = (
        "#include <flashinfer/attention/sparse_mla_sm120/arch/barrier.cuh>\n"
        'extern "C" __global__ void ordering_witness(float* result, unsigned phase) {\n'
        "    __shared__ float payload[1];\n"
        "    __shared__ unsigned long long ready;\n"
        + body.replace("&ready", "reinterpret_cast<uint64_t*>(&ready)")
        + "}\n"
    )
    assert nvcc is not None
    source = tmp_path / "ordering.cu"
    output = tmp_path / "ordering.ptx"
    source.write_text(code)
    subprocess.run(
        [
            nvcc,
            "--std=c++17",
            "--ptx",
            "--gpu-architecture=compute_121a",
            "-O3",
            "--use_fast_math",
            f"-I{include}",
            str(source),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    ptx = output.read_text()
    barrier_at = ptx.index(barrier)
    accesses = [
        match.start()
        for match in re.finditer(rf"\b{mnemonic}\.shared(?:\.[A-Za-z0-9_]+)*\s", ptx)
    ]
    assert any(pos < barrier_at for pos in accesses), ptx
    assert any(pos > barrier_at for pos in accesses), ptx


@pytest.mark.parametrize("operation", ("acquire", "reduce_release", "store_release"))
def test_flashinfer_topk_memory_ordering_ptx(tmp_path, operation):
    """Inspect the real multi-CTA synchronization helpers; do not launch kernels."""
    nvcc = os.environ.get("NVCC") or shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("CUDA compiler is required for the native PTX contract")
    include_override = os.environ.get("FLASHINFER_INCLUDE_DIR")
    if include_override:
        include = Path(include_override)
    else:
        try:
            distribution = importlib.metadata.distribution("flashinfer-python")
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("FlashInfer native headers are not installed")
        include = Path(distribution.locate_file("flashinfer/data/include"))
    cccl = Path(
        os.environ.get(
            "FLASHINFER_CCCL_DIR", "/opt/vllm-runner/sources/flashinfer/3rdparty/cccl"
        )
    )
    assert (include / "flashinfer/topk.cuh").is_file()
    assert (cccl / "cub/cub/cub.cuh").is_file()
    if operation == "acquire":
        body = """
    int before = payload[0];
    int value = flashinfer::sampling::ld_acquire(ready);
    int after = payload[0];
    result[0] = after - before + value;
"""
        mnemonic, boundary = "ld", "ld.global.acquire.gpu"
    else:
        helper = "red_release" if operation == "reduce_release" else "st_release"
        body = (
            "    payload[0] = 1;\n"
            f"    flashinfer::sampling::{helper}(ready, 1);\n"
            "    payload[0] = 2;\n"
            '    asm volatile("" ::: "memory");\n'
            "    result[0] = payload[0];\n"
        )
        mnemonic, boundary = "st", "fence.acq_rel.gpu"
    assert nvcc is not None
    source = tmp_path / "topk_ordering.cu"
    output = tmp_path / "topk_ordering.ptx"
    source.write_text(
        "#include <flashinfer/topk.cuh>\n"
        'extern "C" __global__ void topk_ordering_witness('
        "int* __restrict__ payload, int* __restrict__ ready, "
        "int* __restrict__ result) {\n" + body + "}\n"
    )
    subprocess.run(
        [
            nvcc,
            "--std=c++17",
            "--ptx",
            "--gpu-architecture=compute_121a",
            "-O3",
            "--use_fast_math",
            f"-I{include}",
            f"-I{cccl / 'cub'}",
            f"-I{cccl / 'thrust'}",
            f"-I{cccl / 'libcudacxx/include'}",
            str(source),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    ptx = output.read_text()
    at = ptx.index(boundary)
    accesses = [
        m.start()
        for m in re.finditer(rf"\b{mnemonic}\.global(?:\.[A-Za-z0-9_]+)*\s", ptx)
    ]
    assert any(pos < at for pos in accesses), ptx
    assert any(pos > at for pos in accesses), ptx
