# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU source contracts for compact MLA rows, without Torch or CUDA imports.

Execute the source methods and dataclasses in a small standard-library namespace.
The native check compiles the unchanged FP8 validation and tail branch as host C++.
This checks row ownership and bounds, not CUDA numerical execution.
"""

import ast
import copy
import os
import subprocess
from collections import Counter
from dataclasses import dataclass, field, fields, replace
from enum import Enum, IntEnum
from fractions import Fraction
from math import prod
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

ROOT = Path(os.environ.get("VLLM_COMPACT_TEST_SOURCE", Path(__file__).parents[3]))


def _method(path, class_name, name):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    return copy.deepcopy(
        next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    )


def _execute(nodes, namespace):
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    exec(compile(module, str(ROOT), "exec"), namespace)


def _row_classes():
    base = ast.ClassDef(
        name="MLAAttentionImpl",
        bases=[],
        keywords=[],
        body=[
            _method(
                "vllm/v1/attention/backend.py",
                "MLAAttentionImpl",
                "get_fp8_ds_mla_row_bytes",
            )
        ],
        decorator_list=[],
    )
    impl = ast.ClassDef(
        name="SM120Impl",
        bases=[ast.Name(id="MLAAttentionImpl", ctx=ast.Load())],
        keywords=[],
        body=[
            _method(
                "vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
                "FlashInferMLASparseSM120Impl",
                "get_fp8_ds_mla_row_bytes",
            )
        ],
        decorator_list=[],
    )
    namespace: dict[str, Any] = {"__name__": __name__}
    _execute([base, impl], namespace)
    return namespace


def _spec_namespace():
    selected = {
        "KVQuantMode",
        "SparseCacheRole",
        "KVCacheSpec",
        "AttentionSpec",
        "FullAttentionSpec",
        "MLAAttentionSpec",
        "SlidingWindowSpec",
        "SlidingWindowMLASpec",
        "get_kv_quant_mode",
        "_apply_alignment_padding",
        "get_mla_state_content_bytes",
    }
    tree = ast.parse((ROOT / "vllm/v1/kv_cache_interface.py").read_text())
    namespace = {
        "__name__": __name__,
        "dataclass": dataclass,
        "field": field,
        "fields": fields,
        "replace": replace,
        "Enum": Enum,
        "IntEnum": IntEnum,
        "Fraction": Fraction,
        "prod": prod,
        "Counter": Counter,
        "get_dtype_size": lambda dtype: dtype,
        "cdiv": lambda a, b: (a + b - 1) // b,
        "round_up": lambda a, b: (a + b - 1) // b * b,
    }
    _execute([n for n in tree.body if getattr(n, "name", None) in selected], namespace)
    namespace.update(_row_classes())
    namespace.update({"cast": cast, "kv_cache_dtype_str_to_dtype": lambda *_: 1})
    method = _method(
        "vllm/model_executor/layers/attention/mla_attention.py",
        "MLAAttention",
        "get_kv_cache_spec",
    )
    _execute([method], namespace)
    return namespace


@pytest.mark.parametrize("rope_dim,expected", [(0, 528), (64, 656)])
@pytest.mark.parametrize("block_size", [64, 4608])
@pytest.mark.parametrize("flat", [False, True])
def test_backend_owned_rows_determine_allocated_page_bytes(
    rope_dim, expected, block_size, flat
):
    """The actual allocator must consume the compact backend size, not 656."""
    ns = _spec_namespace()
    impl = ns["SM120Impl"]()
    impl.kv_lora_rank, impl.qk_rope_head_dim = 512, rope_dim
    layer = SimpleNamespace(
        impl=impl,
        head_size=512 + rope_dim,
        kv_cache_dtype="fp8_ds_mla",
        sliding_window=None,
        indexer=None,
        non_causal_multi_token_decode=False,
        attn_backend=SimpleNamespace(get_name=lambda: "FLASHINFER_MLA_SPARSE_SM120"),
        _uses_flat_kv_cache=lambda: flat,
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size), model_config=None
    )
    spec = ns["get_kv_cache_spec"](layer, config)
    assert spec.state_content_size_bytes == expected
    assert spec.page_size_bytes == block_size * expected
    assert spec.block_stride_alignment == (expected if flat else None)


def test_other_backends_keep_standard_rope_rows():
    assert _row_classes()["MLAAttentionImpl"]().get_fp8_ds_mla_row_bytes() == 656


def test_hybrid_alignment_keeps_v030_standard_packed_size():
    ns = _spec_namespace()
    assert ns["get_mla_state_content_bytes"]("fp8_ds_mla") == 656
    assert ns["get_mla_state_content_bytes"]("nvfp4_ds_mla") == 352
    assert ns["get_mla_state_content_bytes"]("auto") is None
    method = _method(
        "vllm/platforms/interface.py", "Platform", "_align_hybrid_block_size"
    )
    calls = [n for n in ast.walk(method) if isinstance(n, ast.Call)]
    mla = next(
        n
        for n in calls
        if isinstance(n.func, ast.Name) and n.func.id == "MLAAttentionSpec"
    )
    value = next(k.value for k in mla.keywords if k.arg == "state_content_bytes")
    assert (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "get_mla_state_content_bytes"
    )


def test_native_compact_writer_does_not_touch_next_row(tmp_path):
    """Execute source validation and RoPE-tail writes with guard bytes on CPU."""
    text = (ROOT / "csrc/libtorch_stable/cache_kernels.cu").read_text()
    start = text.index("void concat_and_cache_mla(")
    validation = text[text.index('  if (kv_cache_dtype == "fp8_ds_mla") {', start) :]
    validation = validation[: validation.index("  } else if (is_nvfp4_ds_mla)")] + "  }"
    start = text.index("__global__ void concat_and_cache_ds_mla_kernel(")
    tail = text[text.index("  if (threadIdx.x >= 64) {", start) :]
    tail = tail[: tail.index("\n  // The first two warps")]
    # Keep the exact source code; only Tensor metadata and CUDA lane ids are supplied.
    deoperatortions = ""
    if "has_rope_tail" in validation + tail:
        deoperatortions = (
            "const int64_t row_bytes = "
            "kv_cache.size(2) * kv_cache.element_size(); "
            "const bool has_rope_tail = row_bytes == 656;"
        )
    program = r"""
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <algorithm>
#define STD_TORCH_CHECK(condition, ...) do { \
    if (!(condition)) throw std::runtime_error("shape"); \
} while (0)
struct Tensor {
    int bytes;
    int size(int) const { return bytes; }
    int element_size() const { return 1; }
};
struct HalfTensor { int element_size() const { return 2; } };
bool accepts(int bytes, int pe_dim) {
    Tensor kv_cache{bytes}; HalfTensor kv_c, k_pe;
    int kv_lora_rank = 512; const char* unused = nullptr; (void)unused;
    std::string kv_cache_dtype = "fp8_ds_mla";
    DEoperatorTIONS
    try { VALIDATION } catch (std::runtime_error&) { return false; }
    return true;
}
void write_tail(int lane, int bytes, int pe_dim, uint8_t* kv_cache, uint16_t* k_pe) {
    using scalar_t = uint16_t;
    struct { int x; } threadIdx{lane};
    int kv_lora_rank = 512, dst_idx_start = 0, token_idx = 0, k_pe_stride = pe_dim;
    bool has_rope_tail = bytes == 656; (void)has_rope_tail;
    scalar_t* kv_cache_16bit = reinterpret_cast<scalar_t*>(kv_cache);
    TAIL
}
int main() {
    if (!accepts(528,0) || !accepts(656,0) || !accepts(656,64)) return 1;
    if (accepts(528,64) || accepts(576,0) ||
        accepts(512,0) || accepts(656,32)) return 2;
    uint16_t rope[64]; for (int i=0;i<64;++i) rope[i] = i+1;
    for (int bytes: {528,656}) for (int pe: {0,64}) {
        if (!accepts(bytes,pe)) continue;
        std::vector<uint8_t> storage(bytes+128,0xab);
        for(int lane=64;lane<96;++lane) write_tail(lane,bytes,pe,storage.data(),rope);
        if (!std::all_of(storage.begin(),storage.begin()+528,
                         [](uint8_t v){return v==0xab;})) return 3;
        if (!std::all_of(storage.begin()+bytes,storage.end(),
                         [](uint8_t v){return v==0xab;})) return 4;
        if(bytes==656 && pe==0 &&
           !std::all_of(storage.begin()+528,storage.begin()+656,
                        [](uint8_t v){return v==0;})) return 5;
        if(bytes==656 && pe==64) for(int i=0;i<64;++i)
            if(reinterpret_cast<uint16_t*>(storage.data()+528)[i]!=rope[i])
                return 6;
    }
}
"""
    program = (
        program.replace("DEoperatorTIONS", deoperatortions)
        .replace("VALIDATION", validation)
        .replace("TAIL", tail)
    )
    source, binary = tmp_path / "compact.cpp", tmp_path / "compact"
    source.write_text(program)
    subprocess.run(
        ["c++", "-std=c++17", "-O0", str(source), "-o", str(binary)],
        check=True,
        timeout=30,
    )
    subprocess.run([str(binary)], check=True, timeout=5)


@pytest.mark.parametrize("row_bytes", [528, 656])
def test_hisparse_registration_keeps_backend_row_width(row_bytes):
    namespace: dict[str, Any] = {"__name__": __name__}
    nodes = []
    for name, parent in [
        ("SparseMLAIndexGroup", None),
        ("HiSparseMLAIndexGroup", "SparseMLAIndexGroup"),
    ]:
        nodes.append(
            ast.ClassDef(
                name=name,
                bases=[] if parent is None else [ast.Name(id=parent, ctx=ast.Load())],
                keywords=[],
                decorator_list=[],
                body=[
                    _method(
                        "vllm/v1/attention/backends/mla/index_group.py",
                        name,
                        "register_layer",
                    )
                ],
            )
        )
    recorded = []

    def create_handle(*args, **kwargs):
        recorded.append(kwargs)
        return SimpleNamespace()

    namespace.update(
        {
            "torch": SimpleNamespace(uint8=1),
            "create_hisparse_cache_handle": create_handle,
            "_create_event": lambda: None,
        }
    )
    _execute(nodes, namespace)
    group = namespace["HiSparseMLAIndexGroup"]()
    group.num_layers, group.prefill_stream = 0, object()
    group.logical_topk_indices = SimpleNamespace(shape=(1, 2048))
    group.caches, group.prefill_ready_events = [], []
    assert (
        group.register_layer(object(), row_width=row_bytes, kv_cache_dtype="fp8_ds_mla")
        == 0
    )
    assert recorded[0]["row_width"] == row_bytes
    assert recorded[0]["kv_dtype"] == 1


def test_hisparse_shared_staging_rejects_mixed_row_widths_before_allocation():
    namespace: dict[str, Any] = {"__name__": __name__}
    method = next(
        n
        for n in ast.parse((ROOT / "vllm/v1/hisparse/runtime.py").read_text()).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "initialize_hisparse_runtime_buffers"
    )
    _execute([method], namespace)
    handles = [
        SimpleNamespace(
            view=SimpleNamespace(
                block_size=64, cache=SimpleNamespace(shape=(1, 64, width), dtype=1)
            )
        )
        for width in (528, 656)
    ]
    with pytest.raises(ValueError, match="identical KV cache row widths"):
        namespace["initialize_hisparse_runtime_buffers"](
            handles, max_num_reqs=1, max_num_batched_tokens=1
        )


def test_dflash_config_preserves_resolved_layout_and_draft_isolation():
    layout_class = next(
        n
        for n in ast.parse((ROOT / "vllm/config/cache.py").read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "CacheConfig"
    )
    layout_field = next(
        n
        for n in layout_class.body
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id == "kv_cache_layout"
    )
    cache_class = ast.ClassDef(
        name="Cache",
        bases=[],
        keywords=[],
        decorator_list=[ast.Name(id="dataclass", ctx=ast.Load())],
        body=[ast.parse('cache_dtype: str = "fp8_ds_mla"').body[0], layout_field],
    )
    base = ast.parse(
        "class Base:\n @property\n def attn_vllm_config(self):\n"
        "  return self.vllm_config"
    ).body[0]
    impl = ast.ClassDef(
        name="DFlashSpeculator",
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        decorator_list=[],
        body=[
            _method(
                "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py",
                "DFlashSpeculator",
                "attn_vllm_config",
            )
        ],
    )
    namespace: dict[str, Any] = {
        "dataclass": dataclass,
        "field": field,
        "copy": copy,
        "replace": replace,
    }
    _execute([cache_class, base, impl], namespace)
    cache = namespace["Cache"]()
    cache.kv_cache_layout = "LBHNC"
    cache.effective_attention_block_size = 4608
    obj = namespace["DFlashSpeculator"]()

    @dataclass
    class Attention:
        backend: str = "FLASHINFER_MLA_SPARSE_SM120"
        use_non_causal: bool = False

    obj.vllm_config = SimpleNamespace(cache_config=cache, attention_config=Attention())
    obj.speculative_config = SimpleNamespace(
        kv_cache_dtype="fp8_e4m3", attention_backend="FLASHINFER"
    )
    obj.requires_non_causal = True
    draft = obj.attn_vllm_config
    assert draft.cache_config is not cache
    assert draft.cache_config.kv_cache_layout == "LBHNC"
    assert draft.cache_config.effective_attention_block_size == 4608
    assert draft.cache_config.cache_dtype == "fp8_e4m3"
    assert cache.cache_dtype == "fp8_ds_mla"
    assert draft.attention_config.backend == "FLASHINFER"
    assert obj.vllm_config.attention_config.backend == "FLASHINFER_MLA_SPARSE_SM120"
