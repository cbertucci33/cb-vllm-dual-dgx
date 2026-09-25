from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import subprocess
from pathlib import Path

required_trellis_symbols = (
    "prepare_weights",
    "Caps",
    "plan",
    "bind",
    "run",
)

required_sparse_mla_parameters = {
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

B12X_VERSION = "1.3.0"
B12X_PACK_WEIGHT_SIGNATURE = (
    ("weight", "POSITIONAL_OR_KEYWORD"),
    ("scale", "POSITIONAL_OR_KEYWORD"),
    ("recipe", "KEYWORD_ONLY"),
    ("global_scale", "KEYWORD_ONLY"),
    ("global_scale_kind", "KEYWORD_ONLY"),
)
B12X_MM_SIGNATURE = (
    ("lhs", "POSITIONAL_OR_KEYWORD"),
    ("rhs", "POSITIONAL_OR_KEYWORD"),
    ("out", "POSITIONAL_OR_KEYWORD"),
    ("kwargs", "VAR_KEYWORD"),
)
REQUIRED_RUNTIME_CONTRACT = {
    "load_format": "instanttensor",
    "attention_backend": "FLASHINFER_MLA_SPARSE_SM120",
    "target_cache_dtype": "fp8_ds_mla",
    "draft_cache_dtype": "fp8_e4m3",
    "linear_backend": "b12x",
}

REQUIRED_TOPK_EXPORTS = (
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
)

FLASHKDA_FWD_ARGUMENTS = (
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


def package_root(distribution: str) -> Path:
    return Path(importlib.metadata.distribution(distribution).locate_file(""))


def require_one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern}, found {len(matches)}")
    return matches[0]


def require_cuda_architecture(path: Path, architecture: str) -> None:
    result = subprocess.run(
        ["/usr/local/cuda/bin/cuobjdump", "--list-elf", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    if architecture not in result.stdout:
        raise RuntimeError(f"{path.name} lacks {architecture} CUDA code")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def function_parameters(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (
            isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            )
            and node.name == function_name
        ):
            return {
                argument.arg for argument in (*node.args.posonlyargs, *node.args.args)
            } | {argument.arg for argument in node.args.kwonlyargs}
    raise RuntimeError(f"{function_name} is missing from {path}")


def source_signature(
    path: Path,
    function_name: str,
) -> tuple[tuple[str, str], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (
            isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            )
            and node.name == function_name
        ):
            positional_only = tuple(
                (argument.arg, "POSITIONAL_ONLY") for argument in node.args.posonlyargs
            )
            positional_or_keyword = tuple(
                (argument.arg, "POSITIONAL_OR_KEYWORD") for argument in node.args.args
            )
            keyword_only = tuple(
                (argument.arg, "KEYWORD_ONLY") for argument in node.args.kwonlyargs
            )
            var_positional = (
                ((node.args.vararg.arg, "VAR_POSITIONAL"),)
                if node.args.vararg is not None
                else ()
            )
            var_keyword = (
                ((node.args.kwarg.arg, "VAR_KEYWORD"),)
                if node.args.kwarg is not None
                else ()
            )
            return (
                *positional_only,
                *positional_or_keyword,
                *var_positional,
                *keyword_only,
                *var_keyword,
            )
    raise RuntimeError(f"{function_name} is missing from {path}")


def require_signature(
    function: object,
    expected: tuple[tuple[str, str], ...],
    name: str,
) -> None:
    actual = tuple(
        (parameter.name, parameter.kind.name)
        for parameter in inspect.signature(function).parameters.values()
    )
    if actual != expected:
        raise RuntimeError(
            f"{name} signature mismatch: expected {expected}, got {actual}"
        )


def verify_build() -> dict[str, str]:
    site_root = package_root("vllm")
    vllm_root = site_root / "vllm"
    core_extension = require_one(vllm_root, "_C_stable_libtorch*.so")
    require_cuda_architecture(core_extension, "sm_120")
    flashkda_extension = require_one(vllm_root, "_flashkda_C*.so")
    require_cuda_architecture(flashkda_extension, "sm_120")
    require_one(vllm_root, "_rust_tool_parser*.so")
    if not (vllm_root / "vllm-rs").is_file():
        raise RuntimeError("vLLM Rust frontend binary is missing")

    flashinfer_source = site_root / "flashinfer/mla/_sparse_mla_sm120/__init__.py"
    if "_MODEL_TYPE_GLM53_NOPE" not in flashinfer_source.read_text(encoding="utf-8"):
        raise RuntimeError("pinned FlashInfer package lacks GLM53_NOPE")
    flashinfer_core = site_root / "flashinfer/mla/_core.py"
    sparse_mla_parameters = function_parameters(
        flashinfer_core,
        "trtllm_batch_decode_with_kv_cache_mla",
    )
    missing_parameters = required_sparse_mla_parameters - sparse_mla_parameters
    if missing_parameters:
        raise RuntimeError(
            "pinned FlashInfer sparse MLA API lacks parameters: "
            f"{sorted(missing_parameters)}"
        )

    topk_module = site_root / "flashinfer/data/aot/topk/topk.so"
    if not topk_module.is_file():
        raise RuntimeError("matching FlashInfer TopK module is missing")
    require_cuda_architecture(topk_module, "sm_121")
    stale_sparse_mla = site_root / "flashinfer/data/aot/sparse_mla_sm120"
    stale_release_sparse_mla = (
        site_root
        / "flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so"
    )
    if stale_sparse_mla.exists() or stale_release_sparse_mla.exists():
        raise RuntimeError("stale release sparse-MLA AOT module is still installed")
    try:
        importlib.metadata.version("flashinfer-jit-cache")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("incompatible release FlashInfer JIT cache is installed")
    provenance = json.loads(
        Path("/opt/glm53-runner-provenance.json").read_text(encoding="utf-8")
    )
    if provenance.get("schema_version") != 2:
        raise RuntimeError("runner provenance schema mismatch")
    runner = provenance.get("runner", {})
    if runner.get("version") != "3.2" or not runner.get("source"):
        raise RuntimeError("runner release provenance mismatch")
    for model_name in ("target", "dflash"):
        model = provenance.get("models", {}).get(model_name, {})
        for stage in ("artifact", "source"):
            identity = model.get(stage, {})
            if (
                not identity.get("repository")
                or len(identity.get("revision", "")) != 40
            ):
                raise RuntimeError(
                    f"incomplete {model_name} {stage} provenance"
                )
    flashinfer_commit = Path(os.environ.get("FLASHINFER_WORKSPACE_BASE", "")).name
    if flashinfer_commit != provenance["external_revisions"]["flashinfer"]:
        raise RuntimeError("FlashInfer JIT workspace is not keyed by source revision")

    trellis_root = site_root / "sparkinfer/moe/trellis_moe"
    trellis_text = "\n".join(
        path.read_text(encoding="utf-8") for path in trellis_root.glob("*.py")
    )
    missing = [
        name for name in required_trellis_symbols if f'"{name}"' not in trellis_text
    ]
    if missing:
        raise RuntimeError(f"missing Sparkinfer Trellis definitions: {missing}")
    if "def scratch_specs(" not in trellis_text:
        raise RuntimeError("Sparkinfer Trellis Plan lacks scratch_specs")
    exllamav3_extension = require_one(site_root, "exllamav3_ext*.so")
    require_cuda_architecture(exllamav3_extension, "sm_121a")

    b12x_root = site_root / "b12x"
    for relative in (
        "gemm/mxfp8_linear/__init__.py",
        "gemm/blockscaled/__init__.py",
    ):
        if not (b12x_root / relative).is_file():
            raise RuntimeError(f"pinned B12X package lacks {relative}")
    b12x_api = b12x_root / "gemm/mxfp8_linear/api.py"
    b12x_api_text = b12x_api.read_text(encoding="utf-8")
    if "kernel_supported, _ = _kernel_is_supported()" not in b12x_api_text:
        raise RuntimeError("B12X support predicate does not unpack the kernel result")
    if importlib.metadata.version("b12x") != B12X_VERSION:
        raise RuntimeError(f"expected B12X {B12X_VERSION}")
    b12x_linear = b12x_root / "gemm/blockscaled/_linear.py"
    if source_signature(b12x_linear, "pack_weight") != B12X_PACK_WEIGHT_SIGNATURE:
        raise RuntimeError("B12X pack_weight source signature mismatch")
    if source_signature(b12x_linear, "blockscaled_mm") != B12X_MM_SIGNATURE:
        raise RuntimeError("B12X mm source signature mismatch")

    if provenance.get("runtime_contract") != REQUIRED_RUNTIME_CONTRACT:
        raise RuntimeError("packaged GLM runner contract mismatch")
    template_path = Path(provenance["chat_template"]["image_path"])
    if sha256(template_path) != provenance["chat_template"]["sha256"]:
        raise RuntimeError("packaged GLM chat template hash mismatch")

    return {
        "vllm": importlib.metadata.version("vllm"),
        "flashinfer-python": importlib.metadata.version("flashinfer-python"),
        "sparkinfer": importlib.metadata.version("sparkinfer"),
        "exllamav3-ext-vllm": importlib.metadata.version("exllamav3-ext-vllm"),
        "nvidia-cutlass-dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "triton": importlib.metadata.version("triton"),
        "b12x": importlib.metadata.version("b12x"),
    }


def verify_runtime() -> dict[str, str]:
    import torch
    from b12x.gemm import mxfp8_linear
    from flashinfer.mla import _sparse_mla_sm120 as sparse_mla_sm120
    from flashinfer.topk import get_topk_module
    from sparkinfer.moe import trellis_moe

    importlib.import_module("vllm._C_stable_libtorch")
    importlib.import_module("vllm._flashkda_C")
    importlib.import_module("vllm._rust_tool_parser")
    flashkda_fwd = torch.ops._flashkda_C.fwd.default
    flashkda_arguments = tuple(arg.name for arg in flashkda_fwd._schema.arguments)
    if flashkda_arguments != FLASHKDA_FWD_ARGUMENTS:
        raise RuntimeError(
            "FlashKDA fwd schema mismatch: "
            f"expected {FLASHKDA_FWD_ARGUMENTS}, got {flashkda_arguments}"
        )
    if not hasattr(torch.ops._flashkda_C, "get_workspace_size"):
        raise RuntimeError("FlashKDA workspace query is missing")
    missing = [
        name for name in required_trellis_symbols if not hasattr(trellis_moe, name)
    ]
    if missing:
        raise RuntimeError(f"missing Sparkinfer Trellis symbols: {missing}")
    if not hasattr(trellis_moe.Plan, "scratch_specs"):
        raise RuntimeError("Sparkinfer Trellis Plan lacks scratch_specs")
    if not hasattr(sparse_mla_sm120, "_MODEL_TYPE_GLM53_NOPE"):
        raise RuntimeError("pinned FlashInfer package lacks GLM53_NOPE")
    topk_module = get_topk_module()
    missing_topk = [
        name for name in REQUIRED_TOPK_EXPORTS if not hasattr(topk_module, name)
    ]
    if missing_topk:
        raise RuntimeError(
            f"matching FlashInfer TopK exports are missing: {missing_topk}"
        )
    for name in ("is_supported", "pack_weight", "mm"):
        if not hasattr(mxfp8_linear, name):
            raise RuntimeError(f"pinned B12X MXFP8 API lacks {name}")
    require_signature(
        mxfp8_linear.pack_weight,
        B12X_PACK_WEIGHT_SIGNATURE,
        "B12X pack_weight",
    )
    require_signature(mxfp8_linear.mm, B12X_MM_SIGNATURE, "B12X mm")
    if not mxfp8_linear.is_supported():
        raise RuntimeError("B12X MXFP8 kernel is unsupported on this GPU")

    try:
        import exllamav3_ext
    except ImportError as exc:
        raise RuntimeError("ExLlamaV3 extension import failed") from exc
    if not hasattr(exllamav3_ext, "exl3_gemm"):
        raise RuntimeError("ExLlamaV3 extension lacks exl3_gemm")

    versions = verify_build()
    versions["torch"] = torch.__version__
    return versions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    versions = verify_build() if args.build_only else verify_runtime()
    Path("/opt/glm53-runner-versions.json").write_text(
        json.dumps(versions, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(versions, sort_keys=True))


if __name__ == "__main__":
    main()
