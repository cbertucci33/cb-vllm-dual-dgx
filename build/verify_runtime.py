#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check the unified image's installed dependency/API contract without a model."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import inspect
import json
import mmap
import os
import site
import subprocess
import sys
from pathlib import Path

import regex as re
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

GLM_CHAT_TEMPLATE = Path("/opt/glm53/chat_template.jinja")
GLM_CHAT_TEMPLATE_SHA256 = (
    "4e5c64b7f8577e1f251a7e6ed64ebb88db242e63ab808f34b3c652846be78ae4"
)


def verify_runtime_assets() -> None:
    if not GLM_CHAT_TEMPLATE.is_file() or GLM_CHAT_TEMPLATE.is_symlink():
        raise RuntimeError(
            f"required GLM chat template is missing: {GLM_CHAT_TEMPLATE}"
        )
    actual = hashlib.sha256(GLM_CHAT_TEMPLATE.read_bytes()).hexdigest()
    if actual != GLM_CHAT_TEMPLATE_SHA256:
        raise RuntimeError(
            f"GLM chat template hash is {actual}; expected {GLM_CHAT_TEMPLATE_SHA256}"
        )


def verify_native_patch_inputs(provenance: dict, runtime_root: Path) -> None:
    inputs = provenance.get("input_sha256", {})
    for relative in (
        "build/build_instanttensor.py",
        "build/patches/instanttensor-0.2.2-cleanup.patch",
        "build/patches/instanttensor-0.2.2-startup-safety.patch",
        "build/patches/exllamav3-startup-safety.patch",
        "build/patches/exllamav3-compiler-memory.patch",
        "build/patches/sparkinfer-trellis-swiglu-limit.patch",
        "build/patches/sparkinfer-trellis-barrier-order.patch",
        "build/patches/sparkinfer-small-prefix-histogram.patch",
        "build/patches/flashinfer-sm120-memory-order.patch",
        "build/patches/transformers-glm5next-temporal-budget.patch",
    ):
        expected = inputs.get(relative)
        if expected is None:
            raise RuntimeError(f"native build input lacks provenance: {relative}")
        installed = runtime_root / relative.removeprefix("build/")
        actual = hashlib.sha256(installed.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"native build input provenance mismatch: {relative}")


def verify_glm5next_temporal_sources(package_root: Path, patch: Path) -> None:
    """Require the processor and its source to use the same padded-frame budget."""
    subprocess.run(
        ["git", "-C", str(package_root), "apply", "--reverse", "--check", str(patch)],
        check=True,
        capture_output=True,
        text=True,
    )


def verify_instanttensor_contract() -> None:
    require_distribution("instanttensor", "0.2.2")
    module = importlib.import_module("instanttensor")
    parameters = inspect.signature(module.safe_open).parameters
    for name in ("allocation_admission", "allocation_budget", "startup_timeout"):
        if name not in parameters:
            raise RuntimeError(f"InstantTensor safe_open lacks {name}")
    native = importlib.import_module("instanttensor._C")
    if "startup_timeout" not in (native.open.__doc__ or ""):
        raise RuntimeError("InstantTensor native open lacks startup timeout ABI")


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


def require_distribution(name: str, expected: str) -> None:
    try:
        actual = metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required distribution {name} is not installed") from exc
    if actual != expected:
        raise RuntimeError(f"{name} version is {actual}; expected {expected}")


def _major_minor(version: str, label: str) -> tuple[int, int]:
    try:
        release = Version(version).release
    except InvalidVersion as exc:
        raise RuntimeError(f"{label} version {version!r} is invalid") from exc
    if len(release) < 2:
        raise RuntimeError(f"{label} version {version!r} lacks major/minor fields")
    return release[:2]


def require_torch_distribution(expected: str, expected_cuda: str) -> None:
    try:
        actual = metadata.version("torch")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("required distribution torch is not installed") from exc

    try:
        actual_public = Version(Version(actual).public)
        expected_public = Version(Version(expected).public)
    except InvalidVersion as exc:
        raise RuntimeError(
            f"torch version is {actual!r}; expected valid version {expected!r}"
        ) from exc
    if actual_public != expected_public:
        raise RuntimeError(
            f"torch version is {actual}; expected {expected} "
            "(CUDA local-version suffix is allowed)"
        )

    torch = importlib.import_module("torch")
    actual_cuda = getattr(getattr(torch, "version", None), "cuda", None)
    if not isinstance(actual_cuda, str):
        raise RuntimeError("torch does not report a CUDA build version")
    if _major_minor(actual_cuda, "torch CUDA") != _major_minor(
        expected_cuda, "configured CUDA"
    ):
        raise RuntimeError(
            f"torch CUDA version is {actual_cuda}; expected the "
            f"{'.'.join(expected_cuda.split('.')[:2])} line"
        )


def require_flashinfer_distribution(expected: str, source_commit: str) -> None:
    try:
        dist = metadata.distribution("flashinfer-python")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "required distribution flashinfer-python is not installed"
        ) from exc
    if dist.version != expected:
        raise RuntimeError(
            f"flashinfer-python version is {dist.version}; expected {expected}"
        )
    verify_local_source(
        dist, dist.read_text("direct_url.json"), "flashinfer", source_commit
    )


def verify_local_source(
    dist: metadata.Distribution,
    direct_url: str | None,
    source_name: str,
    expected_commit: str,
) -> None:
    data = json.loads(direct_url or "{}")
    source_path = Path(f"/opt/vllm-runner/sources/{source_name}").resolve()
    url = data.get("url", "")
    if (
        not url.startswith("file:")
        or Path(url.removeprefix("file://")).resolve() != source_path
    ):
        raise RuntimeError(
            f"{dist.metadata['Name']} direct_url does not identify {source_path}"
        )
    head = subprocess.check_output(
        ["git", "-C", str(source_path), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != expected_commit:
        raise RuntimeError(
            f"{source_name} checkout is {head}; expected {expected_commit}"
        )


def require_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(f"required module {name} is not importable")


def verify_distribution_identity() -> None:
    # Only distributions owned and pinned by this image are in scope. Base-image
    # OS packages can legitimately expose both Debian egg-info and locally
    # installed dist-info metadata under different site roots.
    protected_names = {
        canonicalize_name(name)
        for name in (
            "vllm",
            "torch",
            "nvidia-cutlass-dsl",
            "humming-kernels",
            "nvidia-nccl-cu13",
            "nvidia-cusparselt-cu13",
            "triton",
            "cmake",
            "ninja",
            "tilelang",
            "nvidia-cuda-crt",
            "nvidia-cuda-nvcc",
            "nvidia-nvvm",
            "flashinfer-python",
            "sparkinfer",
            "exllamav3-ext-vllm",
            "b12x",
            "numpy",
            "openai",
            "pydantic",
            "fastapi",
            "starlette",
            "mcp",
            "mcp-types",
        )
    }
    records: dict[str, list[tuple[str, Path]]] = {}
    for root_text in site.getsitepackages():
        root = Path(root_text)
        for dist in metadata.distributions(path=[str(root)]):
            name = dist.metadata.get("Name")
            if not name:
                raise RuntimeError(f"distribution at {dist._path} has no Name metadata")
            canonical_name = canonicalize_name(name)
            if canonical_name in protected_names:
                records.setdefault(canonical_name, []).append(
                    (dist.version, Path(dist._path))
                )
    duplicates = {name: values for name, values in records.items() if len(values) != 1}
    if duplicates:
        raise RuntimeError(f"duplicate distribution metadata: {duplicates}")


def verify_dependency_closure() -> None:
    for dist in metadata.distributions():
        distribution = canonicalize_name(dist.metadata["Name"])
        for raw in dist.requires or ():
            requirement = Requirement(raw)
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            try:
                actual = metadata.version(requirement.name)
            except metadata.PackageNotFoundError as exc:
                raise RuntimeError(
                    f"{distribution} requires missing distribution {requirement.name}"
                ) from exc
            if requirement.specifier and actual not in requirement.specifier:
                raise RuntimeError(
                    f"{distribution} requires {requirement}, but {requirement.name} "
                    f"is {actual}"
                )


def verify_flashinfer_cache_root(cache_root: Path) -> None:
    """Allow import-created directories/logs, but reject executable JIT payloads."""
    if not cache_root.exists():
        return
    unexpected: list[Path] = []
    for path in cache_root.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            continue
        if (
            path.is_file()
            and not path.is_symlink()
            and path.name == "flashinfer_jit.log"
        ):
            continue
        unexpected.append(path)
    if unexpected:
        raise RuntimeError(
            "FlashInfer build/JIT cache contains stale payloads: "
            f"{sorted(str(path.relative_to(cache_root)) for path in unexpected)}"
        )


def verify_exllamav3_compiler_memory_sources(source_root: Path, patch: Path) -> None:
    """Require the pinned native header correction in the exact built source."""
    subprocess.run(
        ["git", "-C", str(source_root), "apply", "--reverse", "--check", str(patch)],
        check=True,
        capture_output=True,
        text=True,
    )
    relative = Path("exllamav3/exllamav3_ext/ptx.cuh")
    if not (source_root / relative).is_file():
        raise RuntimeError(f"ExLlamaV3 patched header missing: {relative}")


def verify_sparkinfer_route_sources(
    package_root: Path, source_root: Path, patch: Path
) -> None:
    """Require the pinned histogram patch in both source and installed package."""
    subprocess.run(
        ["git", "-C", str(source_root), "apply", "--reverse", "--check", str(patch)],
        check=True,
        capture_output=True,
        text=True,
    )
    relative = Path("moe/_shared/kernels/w4a16/route_pack.py")
    if (package_root / relative).read_bytes() != (
        source_root / "sparkinfer" / relative
    ).read_bytes():
        raise RuntimeError(
            f"Sparkinfer packaged source differs from patched source: {relative}"
        )


def verify_sparkinfer_barrier_sources(
    package_root: Path, source_root: Path, patch: Path
) -> None:
    """Reject a missing barrier patch or stale installed DSL source."""
    subprocess.run(
        ["git", "-C", str(source_root), "apply", "--reverse", "--check", str(patch)],
        check=True,
        capture_output=True,
        text=True,
    )
    for relative in (
        Path("_lib/intrinsics.py"),
        Path("moe/_shared/kernels/w4a16/kernel.py"),
    ):
        if (package_root / relative).read_bytes() != (
            source_root / "sparkinfer" / relative
        ).read_bytes():
            raise RuntimeError(
                f"Sparkinfer packaged source differs from patched source: {relative}"
            )


def verify_flashinfer_memory_ordering_headers(
    flashinfer_root: Path,
    source_root: Path,
    patch: Path,
) -> None:
    """Reject an unpatched source or stale packaged native headers."""
    subprocess.run(
        ["git", "-C", str(source_root), "apply", "--reverse", "--check", str(patch)],
        check=True,
        capture_output=True,
        text=True,
    )
    for relative in (
        Path("include/flashinfer/attention/sparse_mla_sm120/arch/barrier.cuh"),
        Path("include/flashinfer/attention/sparse_mla_sm120/arch/cp_async.cuh"),
        Path("include/flashinfer/topk.cuh"),
    ):
        source = source_root / relative
        installed = flashinfer_root / "data" / relative
        if installed.read_bytes() != source.read_bytes():
            raise RuntimeError(
                f"FlashInfer packaged header differs from patched source: {relative}"
            )


def verify_flashinfer_sparse_mla_wrapper(source: str, exports: list[str]) -> None:
    """Require the exported decode route to preserve scoped sparse autotuning."""
    wrapper_name = "flashinfer_trtllm_batch_decode_with_kv_cache_mla"
    expected = ast.parse(
        "_autotune_sparse_mla_call(_lazy_import_wrapper("
        "'flashinfer.decode', 'trtllm_batch_decode_with_kv_cache_mla', "
        "fallback_fn=_missing_sparse_mla))",
        mode="eval",
    ).body
    assignments = [
        node.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == wrapper_name
            for target in node.targets
        )
    ]
    if (
        len(assignments) != 1
        or ast.dump(assignments[0]) != ast.dump(expected)
        or wrapper_name not in exports
    ):
        raise RuntimeError(
            "vllm.utils.flashinfer wrapper must retain scoped autotuning around "
            "the public FlashInfer decode symbol and sparse-MLA fallback"
        )


def verify_flashinfer() -> None:
    required = {
        "query",
        "kv_cache",
        "workspace_buffer",
        "qk_nope_head_dim",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "block_tables",
        "seq_lens",
        "max_seq_len",
        "out",
        "bmm1_scale",
        "bmm2_scale",
        "sparse_mla_top_k",
        "kv_scale_format",
    }
    decode = importlib.import_module("flashinfer.decode")
    public_route = getattr(decode, "trtllm_batch_decode_with_kv_cache_mla", None)
    if not callable(public_route):
        raise RuntimeError("FlashInfer public decode route is unavailable")
    public_signature = inspect.signature(public_route)
    missing = required - set(public_signature.parameters)
    if missing:
        raise RuntimeError(f"public FlashInfer decode API missing {sorted(missing)}")

    sparse = importlib.import_module("flashinfer.mla._sparse_mla_sm120")
    if not hasattr(sparse, "_MODEL_TYPE_GLM53_NOPE"):
        raise RuntimeError("FlashInfer lacks the GLM53_NOPE sparse-MLA route")
    for distribution in ("flashinfer-cubin", "flashinfer-jit-cache"):
        try:
            metadata.distribution(distribution)
        except metadata.PackageNotFoundError:
            pass
        else:
            raise RuntimeError(f"stale {distribution} distribution remains installed")
    for module in ("flashinfer_cubin", "flashinfer_jit_cache"):
        if importlib.util.find_spec(module) is not None:
            raise RuntimeError(f"stale {module} package payload remains importable")
    flashinfer_root = Path(importlib.util.find_spec("flashinfer").origin).parent
    verify_flashinfer_memory_ordering_headers(
        flashinfer_root,
        Path("/opt/vllm-runner/sources/flashinfer"),
        Path("/opt/vllm-runner/patches/flashinfer-sm120-memory-order.patch"),
    )
    compact_source = Path("/opt/vllm-runner/sources/flashinfer")
    subprocess.run(
        [
            "git",
            "-C",
            str(compact_source),
            "apply",
            "--reverse",
            "--check",
            "/opt/vllm-runner/patches/flashinfer-glm53-compact-cache.patch",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for relative in (
        "flashinfer/mla/_core.py",
        "flashinfer/mla/_sparse_mla_sm120.py",
        "flashinfer/mla/_sparse_mla_sm120_cpb.py",
        "flashinfer/mla/_sparse_mla_sm120_plan.py",
        "include/flashinfer/attention/sparse_mla_sm120/model/model_type.h",
        "include/flashinfer/attention/sparse_mla_sm120/model/kv_cache_traits.cuh",
    ):
        installed = (
            flashinfer_root / relative.removeprefix("flashinfer/")
            if relative.startswith("flashinfer/")
            else flashinfer_root / "data" / relative
        )
        if installed.read_bytes() != (compact_source / relative).read_bytes():
            raise RuntimeError(
                f"FlashInfer compact cache source/package mismatch: {relative}"
            )
    subprocess.run(
        [
            "git",
            "-C",
            str(compact_source),
            "apply",
            "--reverse",
            "--check",
            "/opt/vllm-runner/patches/flashinfer-sparse-invalid-kv.patch",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    invalid_kv_header = (
        "include/flashinfer/attention/sparse_mla_sm120/decode_dsv3_2_kernel.cuh"
    )
    if (flashinfer_root / "data" / invalid_kv_header).read_bytes() != (
        compact_source / invalid_kv_header
    ).read_bytes():
        raise RuntimeError(
            "FlashInfer invalid sparse KV repair is missing from packaged header"
        )
    zero_row_header = (
        "include/flashinfer/attention/sparse_mla_sm120/common/zero_row.cuh"
    )
    if (flashinfer_root / "data" / zero_row_header).read_bytes() != (
        compact_source / zero_row_header
    ).read_bytes():
        raise RuntimeError(
            "FlashInfer immutable zero row is missing from packaged header"
        )
    for prefill_header in (
        "include/flashinfer/attention/sparse_mla_sm120/common/kv_cache_io.cuh",
        "include/flashinfer/attention/sparse_mla_sm120/prefill_common.cuh",
        "include/flashinfer/attention/sparse_mla_sm120/prefill_swapab_kernel.cuh",
    ):
        if (flashinfer_root / "data" / prefill_header).read_bytes() != (
            compact_source / prefill_header
        ).read_bytes():
            raise RuntimeError(
                f"FlashInfer invalid-prefill-KV repair missing from {prefill_header}"
            )
    topk = flashinfer_root / "data/aot/topk/topk.so"
    if not topk.is_file():
        raise RuntimeError("newly installed FlashInfer TopK AOT payload is missing")
    payload_suffixes = {".so", ".cubin", ".fatbin", ".ptx"}
    payloads = {
        path.resolve()
        for path in flashinfer_root.rglob("*")
        if path.is_file() and path.suffix in payload_suffixes
    }
    if payloads != {topk.resolve()}:
        unexpected = sorted(
            str(path.relative_to(flashinfer_root))
            for path in payloads - {topk.resolve()}
        )
        raise RuntimeError(
            f"FlashInfer contains stale or unexpected AOT/JIT payloads: {unexpected}"
        )
    verify_flashinfer_cache_root(Path("/cache/jit/flashinfer"))
    root_cache = Path("/root/.cache")
    stale_cache = list(root_cache.glob("flashinfer*")) if root_cache.exists() else []
    if stale_cache:
        raise RuntimeError(f"stale FlashInfer cache remains: {stale_cache}")

    flashinfer = importlib.import_module("flashinfer")
    prefill_wrapper = getattr(flashinfer, "BatchPrefillWithPagedKVCacheWrapper", None)
    decode_wrapper = getattr(flashinfer, "BatchDecodeWithPagedKVCacheWrapper", None)
    if not callable(prefill_wrapper) or not callable(decode_wrapper):
        raise RuntimeError("FlashInfer paged draft-attention wrappers are unavailable")
    require_keyword_contract(
        prefill_wrapper, "FlashInfer paged prefill wrapper", 2, {"backend"}
    )
    require_keyword_contract(
        decode_wrapper,
        "FlashInfer paged decode wrapper",
        2,
        {
            "use_cuda_graph",
            "paged_kv_indptr_buffer",
            "paged_kv_indices_buffer",
            "paged_kv_last_page_len_buffer",
            "use_tensor_cores",
            "backend",
        },
    )

    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    backend_cls = AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120.get_class()
    if backend_cls.__name__ != "FlashInferMLASparseSM120Backend":
        raise RuntimeError(
            "registered SM120 sparse-MLA backend resolves to an unexpected class"
        )
    impl_cls = backend_cls.get_impl_cls()
    impl = importlib.import_module(
        "vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120"
    ).FlashInferMLASparseSM120Impl
    if impl_cls is not impl:
        raise RuntimeError(
            "registered backend does not select FlashInferMLASparseSM120Impl"
        )

    utils = importlib.import_module("vllm.utils.flashinfer")
    wrapper_name = "flashinfer_trtllm_batch_decode_with_kv_cache_mla"
    wrapper = getattr(utils, wrapper_name, None)
    if not callable(wrapper) or not utils.has_flashinfer_sparse_mla_sm120():
        raise RuntimeError(
            "vllm.utils.flashinfer sparse-MLA route does not resolve to FlashInfer"
        )
    utils_path = Path(inspect.getsourcefile(utils) or "")
    verify_flashinfer_sparse_mla_wrapper(
        utils_path.read_text(encoding="utf-8"), utils.__all__
    )

    impl_path = Path(inspect.getsourcefile(impl) or "")
    impl_tree = ast.parse(impl_path.read_text(encoding="utf-8"))
    calls: list[ast.Call] = []
    class_node = next(
        (
            node
            for node in impl_tree.body
            if isinstance(node, ast.ClassDef) and node.name == impl.__name__
        ),
        None,
    )
    method = (
        next(
            (
                node
                for node in class_node.body
                if isinstance(node, ast.FunctionDef) and node.name == "_run_mqa_kernel"
            ),
            None,
        )
        if class_node
        else None
    )
    if method is None:
        raise RuntimeError("FlashInferMLASparseSM120Impl lacks _run_mqa_kernel")
    wrapper_imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "vllm.utils.flashinfer"
        and wrapper_name in {alias.name for alias in node.names}
        for node in ast.walk(method)
    )
    if not wrapper_imported:
        raise RuntimeError("SM120 call chain does not import the utility wrapper")
    for node in ast.walk(method):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == wrapper_name
        ):
            calls.append(node)
    if len(calls) != 1:
        raise RuntimeError(
            "SM120 implementation must call the vLLM FlashInfer wrapper exactly once"
        )
    call = calls[0]
    call_keywords = {
        keyword.arg: keyword.value for keyword in call.keywords if keyword.arg
    }
    try:
        public_signature.bind(**{name: object() for name in call_keywords})
    except TypeError as exc:
        raise RuntimeError(
            f"SM120 call does not bind to public FlashInfer API: {exc}"
        ) from exc
    if not required.issubset(call_keywords):
        omitted = sorted(required - set(call_keywords))
        raise RuntimeError(f"SM120 call omits required FlashInfer arguments: {omitted}")
    kv_expr = call_keywords["kv_cache"]
    if not (
        isinstance(kv_expr, ast.Call)
        and isinstance(kv_expr.func, ast.Attribute)
        and kv_expr.func.attr == "unsqueeze"
        and len(kv_expr.args) == 1
        and isinstance(kv_expr.func.value, ast.Call)
        and isinstance(kv_expr.func.value.func, ast.Attribute)
        and kv_expr.func.value.func.attr == "view"
    ):
        raise RuntimeError(
            "SM120 cache call does not view packed bytes before adding the "
            "q-length axis"
        )


def verify_flashinfer_compact_cache_route(rope_dim: int, row_bytes: int) -> None:
    """Execute the producer's layout through actual CPU-side FlashInfer guards.

    Only the GPU attention callee is replaced: the public v32 wrapper's model
    detection, shape validator, and inner runner's layout resolver execute.
    This proves the static layout contract, not GPU attention numerics.
    """
    from unittest.mock import patch

    import torch
    from flashinfer.mla import _core, _sparse_mla_sm120

    query = torch.empty((1, 1, 16, 512 + rope_dim), dtype=torch.bfloat16)
    for shape in ((2, 64, row_bytes), (2, 1, 64, row_bytes)):
        cache = torch.empty(shape, dtype=torch.uint8)
        reached = []

        def check_consumer(cache=cache, reached=reached, **kwargs):
            actual_cache = kwargs["kv_cache"]
            if actual_cache is not cache:
                raise RuntimeError("FlashInfer unexpectedly converted the packed cache")
            model_type = _sparse_mla_sm120._resolve_model_type(
                query.shape[-1], kwargs["kv_scale_format"]
            )
            payload = _sparse_mla_sm120._bytes_per_token_for_model_type(model_type)
            if payload != row_bytes:
                raise RuntimeError(
                    f"FlashInfer payload {payload} != producer {row_bytes}"
                )
            if (
                _sparse_mla_sm120._packed_kv_page_block_size(
                    cache, model_type=model_type, name="kv_cache"
                )
                != 64
            ):
                raise RuntimeError("FlashInfer interpreted the wrong cache page size")
            reached.append(True)
            return kwargs["out"]

        with patch.object(
            _core, "_trtllm_batch_decode_sparse_mla_sm120", check_consumer
        ):
            _core._trtllm_batch_decode_sparse_mla_v32_sm120(
                query=query,
                kv_cache=cache,
                workspace_buffer=torch.empty(0, dtype=torch.uint8),
                qk_nope_head_dim=512,
                kv_lora_rank=512,
                qk_rope_head_dim=rope_dim,
                block_tables=torch.zeros((1, 1, 64), dtype=torch.int32),
                seq_lens=None,
                sparse_mla_top_k=64,
                out=None,
                bmm1_scale=1.0,
                bmm2_scale=1.0,
                sinks=None,
                skip_softmax_threshold_scale_factor=None,
                uses_shared_paged_kv_idx=True,
                lse=None,
                return_lse=False,
                kv_scale_format="arbitrary_fp32",
            )
        if reached != [True]:
            raise RuntimeError("FlashInfer compact-cache consumer was not reached")


def verify_fp8_ds_mla_cache_boundary(build_only: bool) -> None:
    # The cache dtype canonicalization and actual cache-spec constructor are
    # executable without loading a model or allocating a CUDA tensor.
    from types import SimpleNamespace

    import torch

    from vllm.model_executor.layers.attention.mla_attention import (
        MLAAttention,
        _canonicalize_sparse_mla_kv_cache_dtype,
    )
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    backend_cls = AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120.get_class()
    for source_dtype in ("auto", "fp8", "fp8_e4m3"):
        if (
            _canonicalize_sparse_mla_kv_cache_dtype(backend_cls, source_dtype)
            != "fp8_ds_mla"
        ):
            raise RuntimeError(
                f"SM120 backend does not canonicalize {source_dtype} to fp8_ds_mla"
            )
    for rope_dim, row_bytes in ((0, 528), (64, 656)):
        layer = MLAAttention.__new__(MLAAttention)
        layer.attn_backend = backend_cls
        layer.kv_cache_dtype = "fp8_ds_mla"
        layer.head_size = 512 + rope_dim
        layer.sliding_window = None
        layer.indexer = None
        layer.non_causal_multi_token_decode = False
        impl_cls = backend_cls.get_impl_cls()
        impl = impl_cls.__new__(impl_cls)
        impl.kv_lora_rank = 512
        impl.qk_rope_head_dim = rope_dim
        layer.impl = impl
        config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=64),
            model_config=None,
        )
        spec = layer.get_kv_cache_spec(config)
        if spec.cache_dtype_str != "fp8_ds_mla" or spec.dtype is not torch.uint8:
            raise RuntimeError(
                "SM120 MLA cache spec has the wrong dtype or byte format"
            )
        if (
            spec.block_size != 64
            or spec.state_content_bytes != row_bytes
            or spec.state_content_size_bytes != row_bytes
            or spec.page_size_bytes != 64 * row_bytes
        ):
            raise RuntimeError(
                f"SM120 MLA cache spec must publish {row_bytes}-byte rows "
                f"for qk_rope_head_dim={rope_dim}"
            )

        verify_flashinfer_compact_cache_route(rope_dim, spec.state_content_bytes)

    # Verify the Python wrapper and both source consumers in every environment.
    custom_ops = importlib.import_module("vllm._custom_ops")
    python_op = custom_ops.cp_gather_and_upconvert_fp8_kv_cache
    expected_names = (
        "src_cache",
        "dst",
        "block_table",
        "workspace_starts",
        "batch_size",
        "seq_starts",
        "host_cache",
        "host_row_ids",
        "device_row_ids",
    )
    if tuple(inspect.signature(python_op).parameters) != expected_names:
        raise RuntimeError(
            "Python cp_gather_and_upconvert_fp8_kv_cache signature changed"
        )

    if build_only:
        # BuildKit has no NVIDIA driver, so the CUDA platform cannot load the
        # stable extension.  Verify the exact compiled artifact and embedded
        # schema here; the post-build --gpus all invocation verifies dispatcher
        # registration below.
        vllm_root = Path(importlib.util.find_spec("vllm").origin).parent
        extensions = list(vllm_root.glob("_C_stable_libtorch*.so"))
        if len(extensions) != 1:
            raise RuntimeError(
                f"expected one stable vLLM core extension, found {len(extensions)}"
            )
        expected_native_schema = (
            b"cp_gather_and_upconvert_fp8_kv_cache(Tensor src_cache, Tensor! dst, "
            b"Tensor block_table, Tensor workspace_starts, int batch_size, Tensor? "
            b"seq_starts, Tensor? host_cache=None, Tensor? host_row_ids=None, Tensor? "
            b"device_row_ids=None) -> ()"
        )
        with (
            extensions[0].open("rb") as extension_file,
            mmap.mmap(extension_file.fileno(), 0, access=mmap.ACCESS_READ) as binary,
        ):
            if binary.find(expected_native_schema) < 0:
                raise RuntimeError(
                    "stable vLLM core extension lacks the exact native FP8 "
                    "cache gather schema"
                )
    else:
        # The final image verifier runs with --gpus all.  At that boundary the
        # CUDA platform loads _C_stable_libtorch and the dispatcher contract can
        # be checked directly.  Tensor-layout and GPU execution remain part of
        # model acceptance.
        dispatcher_op = (
            torch.ops._C_cache_ops.cp_gather_and_upconvert_fp8_kv_cache.default
        )
        schema = dispatcher_op._schema
        if tuple(argument.name for argument in schema.arguments) != expected_names:
            raise RuntimeError(
                "registered native cp_gather_and_upconvert_fp8_kv_cache schema changed"
            )
        if len(schema.returns) != 0:
            raise RuntimeError(
                "registered native FP8 cache gather op must return no values"
            )
        dst_alias = schema.arguments[1].alias_info
        if dst_alias is None or not dst_alias.is_write:
            raise RuntimeError(
                "registered native FP8 cache gather schema does not mark dst writable"
            )

    mla_module = importlib.import_module(
        "vllm.model_executor.layers.attention.mla_attention"
    )
    source_callers = (
        (
            Path(mla_module.__file__),
            0,
            {
                "src_cache",
                "dst",
                "block_table",
                "workspace_starts",
                "batch_size",
                "seq_starts",
            },
        ),
        (
            Path(
                importlib.import_module(
                    "vllm.v1.attention.backends.mla.index_group"
                ).__file__
            ),
            5,
            set(),
        ),
    )
    for path, minimum_positional, required_keywords in source_callers:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "cp_gather_and_upconvert_fp8_kv_cache"
        ]
        if not calls:
            raise RuntimeError(
                f"FP8 MLA cache consumer missing native gather call: {path.name}"
            )
        if not any(
            len(node.args) >= minimum_positional
            and required_keywords.issubset(
                {keyword.arg for keyword in node.keywords if keyword.arg}
            )
            for node in calls
        ):
            raise RuntimeError(
                "FP8 MLA cache consumer no longer passes its packed-cache/config "
                f"boundary: {path.name}"
            )


def verify_flashkda(build_only: bool) -> None:
    import torch

    root = Path(importlib.util.find_spec("vllm").origin).parent
    candidates = list(root.glob("_flashkda_C*.so"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one vllm FlashKDA extension, found {len(candidates)}"
        )
    inherited = json.loads(
        Path("/opt/vllm-runner/inherited-native.json").read_text(encoding="utf-8")
    )
    if inherited.get("origin") != "official_source_build":
        raise RuntimeError("FlashKDA origin is not the official vLLM source build")
    base_image_id = inherited.get("base_image_id")
    if not isinstance(base_image_id, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", base_image_id
    ):
        raise RuntimeError("inherited vLLM base image identity is missing or invalid")
    if Path(str(inherited.get("path"))) != candidates[0]:
        raise RuntimeError(
            "inherited FlashKDA path differs from the installed extension"
        )
    actual_sha256 = hashlib.sha256(candidates[0].read_bytes()).hexdigest()
    if inherited.get("sha256") != actual_sha256:
        raise RuntimeError(
            "inherited FlashKDA extension changed during overlay assembly"
        )
    require_sass_target(candidates[0], "sm_121")
    if not build_only:
        torch.ops.load_library(str(candidates[0]))
        namespace = torch.ops._flashkda_C
        for name in ("fwd", "get_workspace_size"):
            if not hasattr(namespace, name):
                raise RuntimeError(f"FlashKDA extension lacks operator {name}")
        expected_schema = (
            "q",
            "k",
            "v",
            "g",
            "beta",
            "scale",
            "out",
            "workspace",
            "A_log",
            "dt_bias",
            "lower_bound",
            "initial_state",
            "final_state",
            "cu_seqlens",
            "checkpoint_state",
            "checkpoint_offsets",
        )
        actual_schema = tuple(
            arg.name for arg in namespace.fwd.default._schema.arguments
        )
        if actual_schema != expected_schema:
            raise RuntimeError(
                f"FlashKDA fwd schema mismatch: expected {expected_schema}, "
                f"got {actual_schema}"
            )
        workspace_schema = namespace.get_workspace_size.default._schema
        workspace_arguments = tuple(arg.name for arg in workspace_schema.arguments)
        if workspace_arguments != ("T_total", "H", "N"):
            raise RuntimeError(
                "FlashKDA get_workspace_size schema mismatch: expected "
                f"('T_total', 'H', 'N'), got {workspace_arguments}"
            )
        if (
            len(workspace_schema.returns) != 1
            or str(workspace_schema.returns[0].type) != "int"
        ):
            raise RuntimeError("FlashKDA get_workspace_size must return one int")
    exllama_spec = importlib.util.find_spec("exllamav3_ext")
    if exllama_spec is None or exllama_spec.origin is None:
        raise RuntimeError("ExLlamaV3 native extension file is missing")
    require_sass_target(Path(exllama_spec.origin), "sm_121")


def require_keyword_contract(
    function: object, name: str, positional: int, keywords: set[str]
) -> None:
    try:
        signature = inspect.signature(function)
        args = [object() for _ in range(positional)]
        kwargs = {key: object() for key in keywords}
        signature.bind(*args, **kwargs)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{name} does not accept the vLLM call contract: {exc}"
        ) from exc


def verify_mhc_tilelang(versions: dict[str, str]) -> None:
    require_distribution("tilelang", versions["TILELANG_VERSION"])
    kernels = importlib.import_module(
        "vllm.model_executor.kernels.mhc.tilelang_kernels"
    )
    required = {
        "mhc_pre_big_fuse_tilelang",
        "mhc_pre_big_fuse_with_norm_tilelang",
        "mhc_pre_big_fuse_broadcast_with_norm_tilelang",
        "mhc_fused_tilelang",
        "mhc_post_tilelang",
        "hc_prenorm_gemm_tilelang",
        "hc_prenorm_gemm_block_m_tilelang",
    }
    missing = sorted(
        name for name in required if not callable(getattr(kernels, name, None))
    )
    if missing:
        raise RuntimeError(f"selected TileLang mHC kernels are unavailable: {missing}")


def verify_bundled_fa2(versions: dict[str, str], build_only: bool) -> None:
    cmake = Path("/opt/vllm-runner/source-manifests/vllm_flash_attn.cmake")
    if not cmake.is_file() or versions["VLLM_FLASH_ATTN_COMMIT"] not in cmake.read_text(
        encoding="utf-8"
    ):
        raise RuntimeError(
            "bundled FlashAttention source revision is not the configured pin"
        )
    vllm_spec = importlib.util.find_spec("vllm")
    if vllm_spec is None or vllm_spec.origin is None:
        raise RuntimeError("vLLM package is missing")
    extensions = list(
        Path(vllm_spec.origin).parent.glob("vllm_flash_attn/_vllm_fa2_C*.so")
    )
    if len(extensions) != 1:
        raise RuntimeError(
            f"expected one bundled FlashAttention 2 extension, found {extensions}"
        )
    if build_only:
        return
    interface = importlib.import_module("vllm.vllm_flash_attn.flash_attn_interface")
    if not interface.FA2_AVAILABLE:
        raise RuntimeError(
            "bundled FlashAttention 2 is unavailable: "
            f"{interface.FA2_UNAVAILABLE_REASON}"
        )
    torch = importlib.import_module("torch")
    op = getattr(torch.ops._vllm_fa2_C, "varlen_fwd", None)
    if op is None or not callable(op):
        raise RuntimeError("FlashAttention 2 lacks torch.ops._vllm_fa2_C.varlen_fwd")


def verify_deepgemm_indexer(versions: dict[str, str]) -> None:
    cmake = Path("/opt/vllm-runner/source-manifests/deepgemm.cmake")
    if not cmake.is_file() or versions["DEEPGEMM_COMMIT"] not in cmake.read_text(
        encoding="utf-8"
    ):
        raise RuntimeError(
            "vendored DeepGEMM source revision is not the configured pin"
        )
    deep_gemm = importlib.import_module("vllm.third_party.deep_gemm")
    for name in ("fp8_fp4_mqa_logits", "fp8_fp4_paged_mqa_logits"):
        if not callable(getattr(deep_gemm, name, None)):
            raise RuntimeError(f"vendored DeepGEMM lacks callable {name}")
    root = Path(deep_gemm.__file__).parent
    extension = root / "_C_extension.abi3.so"
    if not extension.is_file():
        raise RuntimeError(
            f"vendored DeepGEMM extension for this interpreter is missing: {extension}"
        )
    loaded_extension = importlib.import_module(
        "vllm.third_party.deep_gemm._C_extension"
    )
    if Path(loaded_extension.__file__) != extension:
        raise RuntimeError("vendored DeepGEMM loaded from an unexpected path")
    include_root = root / "include"
    if not include_root.is_dir() or not any(include_root.rglob("*.cuh")):
        raise RuntimeError("vendored DeepGEMM JIT headers are missing")
    cache_root = Path(os.environ.get("DG_JIT_CACHE_DIR", "/root/.cache/vllm/deep_gemm"))
    if cache_root.exists():
        payloads = [
            path
            for path in cache_root.rglob("*")
            if path.is_file() and path.suffix in {".so", ".cubin", ".fatbin", ".ptx"}
        ]
        if payloads:
            raise RuntimeError(f"stale DeepGEMM JIT payloads remain: {payloads}")


def verify_b12x_loader_contract() -> None:
    from types import SimpleNamespace

    from b12x.integration.vllm.loader import B12xModelLoader

    class Session:
        index_path: Path | None = None

        def weights(self, shards: object, **kwargs: object):
            self.index_path = kwargs.get("index_path")
            return iter(())

    loader = B12xModelLoader.__new__(B12xModelLoader)
    session = Session()
    loader._session = session
    loader.counter_before_loading_weights = 0.0
    loader.load_config = SimpleNamespace(use_tqdm_on_load=False)
    loader.local_expert_ids = None
    loader._prepare_weights = lambda *args: (
        "/weights",
        ["model-00001-of-00001.safetensors"],
        True,
        "custom.safetensors.index.json",
    )
    source = SimpleNamespace(
        model_or_path="/weights",
        subfolder=None,
        revision=None,
        allow_patterns_overrides=None,
        prefix="",
    )
    list(loader._get_weights_iterator(source))
    expected = Path("/weights/custom.safetensors.index.json")
    if session.index_path != expected:
        raise RuntimeError(
            "B12X loader does not consume the four-value _prepare_weights "
            f"contract: expected index path {expected}, got {session.index_path}"
        )


def require_sass_target(path: Path, expected: str) -> None:
    result = subprocess.run(
        ["/usr/local/cuda/bin/cuobjdump", "--list-elf", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    targets = set(re.findall(r"sm_(\d+)", result.stdout))
    target_names = {f"sm_{target}" for target in targets}
    if target_names != {expected}:
        raise RuntimeError(
            f"{path.name} must contain only exact required SASS ELF target {expected}; "
            f"found {sorted(target_names)}"
        )


def verify_dflash_attention_config_boundary() -> None:
    """Prove draft metadata builders cannot inherit the target KV-cache dtype."""
    from types import SimpleNamespace

    import torch

    from vllm.config import DeviceConfig, VllmConfig
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend
    from vllm.v1.worker.gpu import attn_utils
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    target_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    target_config.cache_config.cache_dtype = "fp8_ds_mla"
    target_config.cache_config.kv_cache_layout = "LBHNC"
    speculator = object.__new__(DFlashSpeculator)
    speculator.vllm_config = target_config
    speculator.speculative_config = SimpleNamespace(
        attention_backend="FLASHINFER",
        kv_cache_dtype="fp8_e4m3",
    )
    speculator.requires_non_causal = True

    draft_config = speculator.attn_vllm_config
    if draft_config.cache_config is target_config.cache_config:
        raise RuntimeError("DFlash draft and target share the cache config object")
    if draft_config.cache_config.kv_cache_layout != "LBHNC":
        raise RuntimeError("DFlash draft lost the resolved KV cache layout")
    if draft_config.cache_config.cache_dtype != "fp8_e4m3":
        raise RuntimeError("DFlash draft did not retain its FP8 E4M3 cache dtype")
    if draft_config.attention_config.backend.name != "FLASHINFER":
        raise RuntimeError("DFlash draft did not retain its FlashInfer backend")
    if (
        FlashInferBackend.get_dtype_for_flashinfer(
            draft_config.cache_config.cache_dtype
        )
        is not torch.float8_e4m3fn
    ):
        raise RuntimeError("DFlash draft dtype does not bind to FlashInfer E4M3")

    parameters = inspect.signature(attn_utils.init_attn_backend).parameters
    if "draft_vllm_config" not in parameters:
        raise RuntimeError("attention initialization cannot receive draft config")
    init_source = inspect.getsource(attn_utils.init_attn_backend)
    if (
        "set(group.layer_names).issubset(draft_layer_names)" not in init_source
        or "vllm_config=group_vllm_config" not in init_source
    ):
        raise RuntimeError("draft-only attention groups do not select draft config")
    runner_source = inspect.getsource(GPUModelRunner.initialize_kv_cache)
    if "draft_vllm_config=" not in runner_source:
        raise RuntimeError("GPU model runner does not pass draft attention config")


def verify_core_and_topk(build_only: bool) -> None:
    vllm_root = Path(importlib.util.find_spec("vllm").origin).parent
    core = list(vllm_root.glob("_C_stable_libtorch*.so"))
    if len(core) != 1:
        raise RuntimeError(
            f"expected one stable vLLM core extension, found {len(core)}"
        )
    require_sass_target(core[0], "sm_121")
    flashinfer = Path(importlib.util.find_spec("flashinfer").origin).parent
    topk = flashinfer / "data/aot/topk/topk.so"
    if not topk.is_file():
        raise RuntimeError("FlashInfer SM121 TopK AOT module is missing")
    require_sass_target(topk, "sm_121")
    required = {
        "radix_topk",
        "radix_topk_page_table_transform",
        "radix_topk_ragged_transform",
        "cub_topk_page_table_transform",
        "cub_topk_page_table_transform_workspace_size",
        "cub_topk_ragged_transform",
        "cub_topk_ragged_transform_workspace_size",
        "cub_topk",
        "cub_topk_workspace_size",
        "can_implement_filtered_topk",
    }
    if not build_only:
        from flashinfer.topk import get_topk_module

        module = get_topk_module()
        missing = sorted(name for name in required if not hasattr(module, name))
        if missing:
            raise RuntimeError(f"FlashInfer TopK exports missing: {missing}")


def verify_rust_frontend() -> None:
    rust_binary = Path(importlib.util.find_spec("vllm").origin).parent / "vllm-rs"
    rust_cli = Path("/usr/local/bin/vllm-rs")
    if not rust_binary.is_file() or not os.access(rust_binary, os.X_OK):
        raise RuntimeError("vLLM Rust frontend binary is missing")
    if not rust_cli.is_symlink() or rust_cli.resolve() != rust_binary:
        raise RuntimeError("vLLM Rust frontend CLI does not resolve to packaged binary")
    rust_version = subprocess.check_output(
        [str(rust_binary), "--version"], text=True
    ).strip()
    source_commit = os.environ.get("VLLM_SOURCE_COMMIT", "")
    rust_package_version = rust_version.removeprefix("vllm-rs ")
    if (
        not rust_version.startswith("vllm-rs 0.32.")
        or not metadata.version("vllm").startswith(rust_package_version)
        or len(source_commit) != 40
        or source_commit[:9] not in rust_package_version
    ):
        raise RuntimeError(
            f"vLLM Rust frontend version {rust_version} does not match source"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--versions", type=Path, default=Path(__file__).with_name("versions.env")
    )
    parser.add_argument(
        "--provenance", type=Path, default=Path("/opt/vllm-runner/provenance.json")
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="run import/API checks; never load a model",
    )
    args = parser.parse_args()
    verify_runtime_assets()
    versions = parse_env(args.versions)
    provenance: dict[str, object] = {}
    if args.provenance.exists():
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        verify_native_patch_inputs(provenance, args.provenance.parent)
        recorded = provenance.get("native_revisions", {})
        for key, value in versions.items():
            if key.endswith("_COMMIT"):
                name = key.removesuffix("_COMMIT").lower()
                if not isinstance(recorded, dict) or recorded.get(name) != value:
                    raise RuntimeError(f"provenance mismatch for {key}")

    if tuple(sys.version_info[:2]) != tuple(
        map(int, versions["PYTHON_VERSION"].split("."))
    ):
        raise RuntimeError(
            f"Python runtime is {sys.version.split()[0]}, expected "
            f"{versions['PYTHON_VERSION']}"
        )

    installed_vllm_version = metadata.version("vllm")
    if not installed_vllm_version.startswith("0.32."):
        raise RuntimeError(
            f"installed vLLM version {installed_vllm_version} is not the v0.32 line"
        )
    require_torch_distribution(versions["TORCH_VERSION"], versions["CUDA_VERSION"])
    for name, key in (
        ("nvidia-cutlass-dsl", "NVIDIA_CUTLASS_DSL_VERSION"),
        ("humming-kernels", "HUMMING_KERNELS_VERSION"),
        ("nvidia-nccl-cu13", "NCCL_VERSION"),
        ("nvidia-cusparselt-cu13", "CUSPARSELT_VERSION"),
        ("triton", "TRITON_VERSION"),
        ("cmake", "CMAKE_VERSION"),
        ("ninja", "NINJA_VERSION"),
        ("tilelang", "TILELANG_VERSION"),
        ("nvidia-cuda-crt", "CUDA_PYTHON_COMPONENT_VERSION"),
        ("nvidia-cuda-nvcc", "CUDA_PYTHON_COMPONENT_VERSION"),
        ("nvidia-nvvm", "CUDA_PYTHON_COMPONENT_VERSION"),
        ("numpy", "NUMPY_VERSION"),
        ("openai", "OPENAI_VERSION"),
        ("pydantic", "PYDANTIC_VERSION"),
        ("fastapi", "FASTAPI_VERSION"),
        ("starlette", "STARLETTE_VERSION"),
        ("mcp", "MCP_VERSION"),
        ("mcp-types", "MCP_TYPES_VERSION"),
    ):
        require_distribution(name, versions[key])
    require_flashinfer_distribution(
        versions["FLASHINFER_VERSION"], versions["FLASHINFER_COMMIT"]
    )

    require_distribution("transformers", "5.17.0")
    verify_glm5next_temporal_sources(
        Path(importlib.util.find_spec("transformers").origin).parent,
        Path("/opt/vllm-runner/patches/transformers-glm5next-temporal-budget.patch"),
    )
    verify_instanttensor_contract()
    for module in ("exllamav3_ext", "sparkinfer.moe", "b12x"):
        require_module(module)
    if not args.build_only:
        exl3 = importlib.import_module("exllamav3_ext")
        if not callable(getattr(exl3, "exl3_gemm", None)):
            raise RuntimeError("ExLlamaV3 extension lacks callable exl3_gemm")
    for dist_name, source_name, key in (
        ("sparkinfer", "sparkinfer", "SPARKINFER_COMMIT"),
        ("exllamav3-ext-vllm", "exllamav3", "EXLLAMAV3_COMMIT"),
        ("b12x", "b12x", "B12X_COMMIT"),
    ):
        dist = metadata.distribution(dist_name)
        verify_local_source(
            dist, dist.read_text("direct_url.json"), source_name, versions[key]
        )
    verify_distribution_identity()
    verify_dependency_closure()
    subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=True,
        text=True,
    )
    if metadata.version("b12x") != versions["B12X_VERSION"]:
        raise RuntimeError(
            f"B12X version mismatch: expected {versions['B12X_VERSION']}"
        )
    verify_exllamav3_compiler_memory_sources(
        Path("/opt/vllm-runner/sources/exllamav3"),
        Path("/opt/vllm-runner/patches/exllamav3-compiler-memory.patch"),
    )
    verify_sparkinfer_barrier_sources(
        Path(importlib.util.find_spec("sparkinfer").origin).parent,
        Path("/opt/vllm-runner/sources/sparkinfer"),
        Path("/opt/vllm-runner/patches/sparkinfer-trellis-barrier-order.patch"),
    )
    verify_sparkinfer_route_sources(
        Path(importlib.util.find_spec("sparkinfer").origin).parent,
        Path("/opt/vllm-runner/sources/sparkinfer"),
        Path("/opt/vllm-runner/patches/sparkinfer-small-prefix-histogram.patch"),
    )
    trellis = importlib.import_module("sparkinfer.moe").trellis_moe
    require_keyword_contract(
        trellis.prepare_weights,
        "Sparkinfer prepare_weights",
        2,
        {
            "gate_suh",
            "up_suh",
            "intermediate_rotations",
            "down_svh",
            "codebook",
            "mcg",
            "tile_config",
        },
    )
    require_keyword_contract(
        trellis.Caps,
        "Sparkinfer Caps",
        0,
        {
            "max_tokens",
            "num_topk",
            "num_experts",
            "hidden_size",
            "intermediate_size",
            "route_num_experts",
            "block_size_m",
            "trellis_bits",
            "tile_config",
            "input_dtype",
            "swiglu_limit",
            "device",
        },
    )
    require_keyword_contract(trellis.plan, "Sparkinfer plan", 1, set())
    require_keyword_contract(
        trellis.bind,
        "Sparkinfer bind",
        1,
        {"scratch", "a", "weights", "topk_weights", "topk_ids"},
    )
    require_keyword_contract(trellis.run, "Sparkinfer run", 0, {"binding"})
    verify_b12x_loader_contract()
    for module in (
        "b12x.attention.paged",
        "b12x.gemm.blockscaled",
        "b12x._lib.intrinsics",
        "b12x.gemm.mxfp8_linear",
        "b12x.gemm.tensor_fp8_linear",
        "b12x.moe.fused_moe",
    ):
        require_module(module)

    from b12x.gemm import mxfp8_linear

    for attr in ("is_supported", "pack_weight", "mm"):
        if not callable(getattr(mxfp8_linear, attr, None)):
            raise RuntimeError(f"B12X MXFP8 capability lacks callable {attr}")
    api = Path(inspect.getsourcefile(mxfp8_linear) or "").parent / "api.py"
    if "kernel_supported, _ = _kernel_is_supported()" not in api.read_text(
        encoding="utf-8"
    ):
        raise RuntimeError(
            "B12X MXFP8 capability predicate is not the accepted GB10 implementation"
        )
    pack_parameters = tuple(
        inspect.signature(mxfp8_linear.pack_weight).parameters.values()
    )
    pack_contract = (
        ("weight", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("scale", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("recipe", inspect.Parameter.KEYWORD_ONLY),
        ("global_scale", inspect.Parameter.KEYWORD_ONLY),
        ("global_scale_kind", inspect.Parameter.KEYWORD_ONLY),
    )
    if tuple((p.name, p.kind) for p in pack_parameters) != pack_contract:
        raise RuntimeError(
            "B12X pack_weight signature differs from accepted MXFP8 call contract"
        )
    mm_parameters = tuple(inspect.signature(mxfp8_linear.mm).parameters.values())
    if (
        tuple((p.name, p.kind) for p in mm_parameters[:3])
        != (
            ("lhs", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ("rhs", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ("out", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        )
        or len(mm_parameters) != 4
        or mm_parameters[3].kind != inspect.Parameter.VAR_KEYWORD
    ):
        raise RuntimeError(
            "B12X mm signature differs from accepted MXFP8 call contract"
        )
    profile = (
        Path(mxfp8_linear.__file__).parents[2]
        / "policy/_profiles/data/nvidia.gb10.48sm.json.gz"
    )
    expected_profile = Path(__file__).with_name("profiles") / "nvidia.gb10.48sm.json.gz"
    if (
        not profile.resolve().is_file()
        or profile.resolve().read_bytes() != expected_profile.read_bytes()
    ):
        raise RuntimeError("B12X accepted GB10 MXFP8 profile is missing or altered")

    verify_flashinfer()
    verify_mhc_tilelang(versions)
    verify_bundled_fa2(versions, args.build_only)
    verify_deepgemm_indexer(versions)
    verify_fp8_ds_mla_cache_boundary(args.build_only)
    verify_dflash_attention_config_boundary()
    verify_core_and_topk(args.build_only)
    verify_flashkda(args.build_only)
    verify_rust_frontend()
    if args.provenance.exists():
        source_commit = provenance.get("source", {}).get("vllm_commit")
        if (
            not isinstance(source_commit, str)
            or len(source_commit) != 40
            or os.environ.get("VLLM_SOURCE_COMMIT") != source_commit
        ):
            raise RuntimeError(
                "installed vLLM source revision differs from exact image provenance"
            )
    elif not args.build_only:
        raise FileNotFoundError(args.provenance)
    if args.build_only:
        print(
            "dependency versions, Python call chain, and native artifact schema "
            "verified, including compact cache producer/consumer guards; "
            "GPU dispatcher and cache numerics remain "
            "dynamic; no model loaded"
        )
    else:
        print(
            "dependency versions, Python call chain, and registered dispatcher "
            "schemas verified; "
            "native GPU cache-layout ABI remains dynamic; no model loaded"
        )


if __name__ == "__main__":
    main()
