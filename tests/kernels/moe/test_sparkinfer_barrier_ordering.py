# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU DSL-emission contracts; these do not execute a CUDA memory model."""

import ast
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

_ROOT = Path(
    os.environ.get("SPARKINFER_REVIEW_ROOT", "/opt/vllm-runner/sources/sparkinfer")
)
_INTRINSICS = _ROOT / "sparkinfer/_lib/intrinsics.py"
_KERNEL = _ROOT / "sparkinfer/moe/_shared/kernels/w4a16/kernel.py"


def _emit(name, argc):
    node = next(
        n
        for n in ast.parse(_INTRINSICS.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    emitted = []

    class Scalar:
        def __init__(self, value):
            self.value = value

        def ir_value(self, **kwargs):
            return self.value

    def inline_asm(*args, **kwargs):
        emitted.append((args, kwargs))
        return 0

    ns = {
        "Int32": Scalar,
        "Int64": Scalar,
        "Uint32": Scalar,
        "Float32": Scalar,
        "T": SimpleNamespace(i32=lambda: "i32", f32=lambda: "f32"),
        "llvm": SimpleNamespace(
            inline_asm=inline_asm,
            StructType=SimpleNamespace(get_literal=lambda types: tuple(types)),
            extractvalue=lambda typ, value, index, **kw: 0,
            AsmDialect=SimpleNamespace(AD_ATT="att"),
        ),
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(_INTRINSICS), "exec"), ns)
    emit = cast(Callable[..., object], ns[name])
    emit(*range(argc))
    assert len(emitted) == 1
    return emitted[0]


@pytest.mark.parametrize(
    "name,argc,mnemonic",
    [
        ("atomic_add_global_acq_rel_i32", 2, "atom.acq_rel.gpu.global.add.s32"),
        ("ld_global_acquire_i32", 1, "ld.global.acquire.gpu.s32"),
        ("red_add_global_release_i32", 2, "fence.acq_rel.gpu"),
        ("st_global_i32", 2, "st.global.s32"),
        ("threadfence", 0, "membar.gl"),
    ],
)
def test_barrier_helpers_emit_memory_contract(name, argc, mnemonic):
    args, kwargs = _emit(name, argc)
    assert mnemonic in args[2]
    assert "~{memory}" in args[3]
    assert kwargs["has_side_effects"] is True


@pytest.mark.parametrize(
    "name,argc,mnemonic",
    [
        ("cp_async4_shared_global", 2, "cp.async.cg.shared.global"),
        ("cp_async4_shared_global_pred", 3, "@p cp.async.cg.shared.global"),
        ("ld_global_v4_f32", 1, "ld.global.v4.f32"),
        ("ld_shared_f32", 1, "ld.shared.f32"),
        ("ld_shared_i32_relaxed", 1, "ld.shared.s32"),
        ("ld_shared_u32", 1, "ld.shared.u32"),
        ("ld_shared_v2_u32", 1, "ld.shared.v2.u32"),
        ("ld_shared_v4_f32", 1, "ld.shared.v4.f32"),
        ("ld_shared_v4_u32", 1, "ld.shared.v4.u32"),
        ("ldmatrix_m8n8x4_b16", 1, "ldmatrix.sync.aligned.m8n8.x4.shared.b16"),
        ("ldmatrix_m8n8x2_b16", 1, "ldmatrix.sync.aligned.m8n8.x2.shared.b16"),
        ("st_global_v4_u32", 5, "st.global.v4.u32"),
        ("st_global_v4_f32", 5, "st.global.v4.f32"),
        ("st_shared_f16_from_f32", 2, "st.shared.b16"),
        ("st_shared_f32", 2, "st.shared.f32"),
        ("st_shared_i32", 2, "st.shared.s32"),
        ("st_shared_u32", 2, "st.shared.u32"),
        ("st_shared_v4_f32", 5, "st.shared.v4.f32"),
        ("st_shared_v4_u32", 5, "st.shared.v4.u32"),
    ],
)
def test_trellis_scratch_helpers_emit_memory_contract(name, argc, mnemonic):
    args, kwargs = _emit(name, argc)
    assert mnemonic in args[2]
    assert "~{memory}" in args[3]
    assert kwargs["has_side_effects"] is True


def test_grid_arrival_publishes_and_acquires_preceding_blocks():
    kernel = ast.parse(_KERNEL.read_text())
    cls = next(
        n
        for n in kernel.body
        if isinstance(n, ast.ClassDef) and n.name == "W4A16FusedMoeKernel"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_grid_barrier"
    )
    arrivals = [
        n.func.id
        for n in ast.walk(method)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id.startswith("atomic_add_global")
    ]
    assert arrivals == ["atomic_add_global_acq_rel_i32"]
    args, _ = _emit(arrivals[0], 2)
    assert "atom.acq_rel.gpu.global.add.s32" in args[2]


def test_relaxed_ticket_allocator_is_not_changed():
    args, _ = _emit("atomic_add_global_i32", 2)
    assert args[2] == "atom.global.add.s32 $0, [$1], $2;"


@pytest.mark.parametrize("defect", ("none", "unpatched", "intrinsics", "kernel"))
def test_sparkinfer_barrier_packaging(tmp_path, defect):
    import difflib
    import importlib.util
    import subprocess

    repo = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "runner_barrier_verifier", repo / "build/verify_runtime.py"
    )
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    source = tmp_path / "source"
    package = tmp_path / "package"
    patch_lines: list[str] = []
    for relative, label in (
        (Path("_lib/intrinsics.py"), "intrinsics"),
        (Path("moe/_shared/kernels/w4a16/kernel.py"), "kernel"),
    ):
        old, new = "version = 0\n", "version = 1\n"
        src = source / "sparkinfer" / relative
        installed = package / relative
        src.parent.mkdir(parents=True, exist_ok=True)
        installed.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(old if defect == "unpatched" else new)
        installed.write_text(old if defect == label else new)
        patch_lines.extend(
            difflib.unified_diff(
                old.splitlines(True),
                new.splitlines(True),
                fromfile=f"a/sparkinfer/{relative}",
                tofile=f"b/sparkinfer/{relative}",
            )
        )
    patch = tmp_path / "barrier.patch"
    patch.write_text("".join(patch_lines))
    if defect == "none":
        verifier.verify_sparkinfer_barrier_sources(package, source, patch)
    elif defect == "unpatched":
        with pytest.raises(subprocess.CalledProcessError):
            verifier.verify_sparkinfer_barrier_sources(package, source, patch)
    else:
        with pytest.raises(RuntimeError, match="packaged source differs"):
            verifier.verify_sparkinfer_barrier_sources(package, source, patch)


@pytest.mark.parametrize("blocks", (1, 2, "sm_count"))
@pytest.mark.parametrize("graph", (False, True))
def test_trellis_grid_barrier_native(blocks, graph):
    """Exercise the actual barrier with all-thread stores and repeated epochs."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("GB10 CUDA execution is required")
    assert torch.cuda.get_device_capability() == (12, 1)
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from sparkinfer.moe._shared.kernels.w4a16.kernel import W4A16FusedMoeKernel

    blocks = (
        torch.cuda.get_device_properties(0).multi_processor_count
        if blocks == "sm_count"
        else blocks
    )

    class BarrierProbe(W4A16FusedMoeKernel):
        def __init__(self, blocks):
            self.blocks = blocks
            self.barrier_count_off = 0
            self.barrier_sense_off = 1

        @property
        def __cache_key__(self):
            return ("trellis_barrier_probe", self.blocks)

        @cute.jit
        def __call__(
            self,
            payload: cute.Tensor,
            errors: cute.Tensor,
            locks: cute.Tensor,
            stream: cuda.CUstream,
        ):
            self.probe(payload, errors, locks).launch(
                grid=(self.blocks, 1, 1),
                block=(128, 1, 1),
                cooperative=True,
                stream=stream,
            )

        @cute.kernel
        def probe(self, payload: cute.Tensor, errors: cute.Tensor, locks: cute.Tensor):
            tx, _, _ = cute.arch.thread_idx()
            bx, _, _ = cute.arch.block_idx()
            tid, cta = cutlass.Int32(tx), cutlass.Int32(bx)
            index = cta * cutlass.Int32(128) + tid
            failed = cutlass.Int32(0)
            for epoch in range(3):
                payload[index] = cutlass.Int32(epoch + 1) * 10000 + index
                self._grid_barrier(locks, tid, cutlass.Int32(self.blocks))
                for peer in range(self.blocks):
                    peer_index = cutlass.Int32(peer * 128) + tid
                    expected = cutlass.Int32(epoch + 1) * 10000 + peer_index
                    if payload[peer_index] != expected:
                        failed += cutlass.Int32(1)
                # Readers must finish before any block overwrites the next epoch.
                self._grid_barrier(locks, tid, cutlass.Int32(self.blocks))
            errors[index] = failed

    payload = torch.empty(blocks * 128, dtype=torch.int32, device="cuda")
    errors = torch.empty_like(payload)
    locks = torch.zeros(2, dtype=torch.int32, device="cuda")
    tensors = [from_dlpack(x) for x in (payload, errors, locks)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(BarrierProbe(blocks), *tensors, stream)
    compiled(*tensors, stream)
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            capture_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            compiled(*tensors, capture_stream)
        captured.replay()
        captured.replay()
    else:
        compiled(*tensors, stream)
    torch.accelerator.synchronize()
    assert torch.count_nonzero(errors).item() == 0
    assert locks[0].item() == 0
