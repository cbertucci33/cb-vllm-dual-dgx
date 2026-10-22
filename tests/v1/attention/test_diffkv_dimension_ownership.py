# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-owned DiffKV dimensions, with native kernels replaced at the call boundary."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CompilationConfig
from vllm.model_executor.layers.attention.attention import Attention
from vllm.platforms.cpu import CpuPlatform
from vllm.v1.attention.backends import flash_attn as flash_parent
from vllm.v1.attention.backends import flash_attn_diffkv as flash
from vllm.v1.attention.backends import triton_attn as triton_parent
from vllm.v1.attention.backends import triton_attn_diffkv as triton
from vllm.v1.kv_cache_interface import FullAttentionSpec


@pytest.fixture
def cpu_boundaries(monkeypatch):
    import vllm.model_executor.layers.attention.attention as attention

    monkeypatch.setattr(attention, "current_platform", CpuPlatform())
    monkeypatch.setattr(triton_parent, "current_platform", CpuPlatform())
    monkeypatch.setattr(flash_parent, "current_platform", CpuPlatform())
    monkeypatch.setattr(flash_parent, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(flash_parent, "get_flash_attn_version", lambda **kw: 4)
    monkeypatch.setattr(flash, "get_flash_attn_version", lambda **kw: 4)
    monkeypatch.setattr(
        flash_parent, "flash_attn_supports_kv_cache_dtype", lambda *a, **kw: True
    )
    monkeypatch.setattr(
        flash_parent, "flash_attn_supports_quant_query_input", lambda: False
    )


def _config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_mm_prefix_lm=False,
            get_num_attention_heads=lambda _: 2,
            get_num_kv_heads=lambda _: 2,
            get_head_size=lambda: 192,
            rswa_window=None,
        ),
        parallel_config=SimpleNamespace(),
        compilation_config=CompilationConfig(),
    )


@pytest.mark.parametrize("head_size_v", [64, 128, 256])
def test_diffkv_builder_uses_group_value_width(cpu_boundaries, head_size_v):
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=192,
        head_size_v=head_size_v,
        dtype=torch.float32,
    )
    builder = triton.TritonAttentionDiffKVMetadataBuilder(
        spec, [], _config(), torch.device("cpu")
    )
    assert builder.softmax_segm_output.shape[-1] == head_size_v


@pytest.mark.parametrize("module", [triton, flash])
@pytest.mark.parametrize("head_size_v", [64, 128, 256])
def test_diffkv_constructor_owns_value_width(cpu_boundaries, module, head_size_v):
    impl_cls = (
        triton.TritonAttentionDiffKVImpl
        if module is triton
        else flash.FlashAttentionDiffKVImpl
    )
    impl = impl_cls(2, 192, 192**-0.5, 2, None, None, "auto", head_size_v=head_size_v)
    other = impl_cls(2, 192, 192**-0.5, 2, None, None, "auto", head_size_v=32)
    assert impl.head_size_v == head_size_v
    assert other.head_size_v == 32


@pytest.mark.parametrize(
    "backend",
    [
        triton.TritonAttentionDiffKVBackend,
        flash.FlashAttentionDiffKVBackend,
        triton_parent.TritonAttentionBackend,
    ],
)
def test_attention_binds_width_only_to_diffkv(cpu_boundaries, monkeypatch, backend):
    import vllm.model_executor.layers.attention.attention as attention

    monkeypatch.setattr(attention, "get_current_vllm_config", _config)
    is_diffkv = backend is not triton_parent.TritonAttentionBackend
    width = 64 if is_diffkv else 192
    layer = Attention(
        2, 192, 192**-0.5, num_kv_heads=2, attn_backend=backend, head_size_v=width
    )
    assert layer.head_size_v == width
    if is_diffkv:
        assert layer.impl.head_size_v == width
    else:
        assert layer.impl.head_size == width


def _metadata():
    return SimpleNamespace(
        use_cascade=False,
        causal=True,
        mm_prefix_range=None,
        mm_prefix_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
        num_actual_tokens=1,
        query_start_loc=torch.tensor([0, 1]),
        seq_lens=torch.tensor([1]),
        max_query_len=1,
        max_seq_len=1,
        block_table=torch.tensor([[0]]),
        seq_threshold_3D=64,
        num_par_softmax_segments=16,
        softmax_segm_output=None,
        softmax_segm_max=None,
        softmax_segm_expsum=None,
        scheduler_metadata=None,
        max_num_splits=0,
    )


@pytest.mark.parametrize("module", [triton, flash])
def test_diffkv_forward_ignores_legacy_global_width(
    cpu_boundaries, monkeypatch, module
):
    impl_cls = (
        triton.TritonAttentionDiffKVImpl
        if module is triton
        else flash.FlashAttentionDiffKVImpl
    )
    backend = (
        triton.TritonAttentionDiffKVBackend
        if module is triton
        else flash.FlashAttentionDiffKVBackend
    )
    impl = impl_cls(2, 192, 192**-0.5, 2, None, None, "auto")
    monkeypatch.setattr(backend, "head_size_v", 32, raising=False)
    _assert_forward_views(monkeypatch, module, impl, 128)


def test_flash_diffkv_value_wider_than_key(cpu_boundaries, monkeypatch):
    impl = flash.FlashAttentionDiffKVImpl(
        2, 192, 192**-0.5, 2, None, None, "auto", head_size_v=256
    )
    _assert_forward_views(monkeypatch, flash, impl, 256)


def _assert_forward_views(monkeypatch, module, impl, width):
    captured = []
    kernel = (
        "unified_attention_diffkv" if module is triton else "flash_attn_varlen_func"
    )
    monkeypatch.setattr(module, kernel, lambda **kw: captured.append(kw), raising=False)
    q = torch.empty(1, 2, 192)
    k = torch.empty(1, 2, 192)
    v = torch.empty(1, 2, width)
    cache = torch.arange(2 * 16 * (192 + width), dtype=torch.float32).reshape(
        1, 2, 16, 192 + width
    )
    out = torch.empty(1, 2, width)
    layer = SimpleNamespace(
        _q_scale=torch.tensor(1.0),
        _k_scale=torch.tensor(1.0),
        _v_scale=torch.tensor(1.0),
    )
    assert impl.forward(layer, q, k, v, cache, _metadata(), out) is out
    assert len(captured) == 1
    call = captured[0]
    torch.testing.assert_close(call["k"], cache.transpose(1, 2)[..., :192])
    torch.testing.assert_close(call["v"], cache.transpose(1, 2)[..., 192:])
    assert call["out"].shape == (1, 2, width)


@pytest.mark.parametrize(
    "backend", [triton.TritonAttentionDiffKVBackend, flash.FlashAttentionDiffKVBackend]
)
def test_benchmark_dimensions_reach_impl_spec_and_cache(
    cpu_boundaries, monkeypatch, backend
):
    from pathlib import Path

    from vllm.v1.kv_cache_interface import KVCacheLayout

    bench_path = Path(__file__).resolve().parents[3] / "benchmarks/attention_benchmarks"
    monkeypatch.syspath_prepend(str(bench_path))
    from common import BenchmarkConfig

    from benchmarks.attention_benchmarks.runner import (
        _create_backend_impl,
        _create_kv_cache,
    )

    config = BenchmarkConfig(
        backend=backend.get_name(),
        batch_spec="1q1s1k",
        num_layers=1,
        head_dim=192,
        v_head_dim=64,
        num_q_heads=2,
        num_kv_heads=2,
        block_size=16,
        device="cpu",
    )
    _, impl, layer = _create_backend_impl(
        {"backend_class": backend}, config, torch.device("cpu"), torch.float32
    )
    assert impl.head_size_v == 64
    assert layer.get_kv_cache_spec().head_size_v == 64
    cache = _create_kv_cache(
        config, 1, backend, torch.device("cpu"), torch.float32, KVCacheLayout.LBHNC
    )
    assert len(cache) == 1
    assert cache[0].shape == (1, 2, 16, 256)


@pytest.mark.parametrize("num_kv_heads,head_size", [(2, 192), (4, 128), (1, 64)])
@pytest.mark.parametrize("diffkv", [False, True])
@pytest.mark.parametrize("full_graph", [False, True])
def test_triton_builder_uses_cache_group_geometry(
    cpu_boundaries, num_kv_heads, head_size, diffkv, full_graph
):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.utils.math_utils import next_power_of_2

    cfg = _config()  # Target geometry is KV2 / QK192, unlike the draft/group.
    cfg.model_config.get_num_attention_heads = lambda _: 8
    captures = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    cfg.compilation_config.cudagraph_mode = (
        CUDAGraphMode.FULL_DECODE_ONLY if full_graph else CUDAGraphMode.NONE
    )
    cfg.compilation_config.cudagraph_capture_sizes = captures
    width_v = 64 if diffkv else head_size
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        head_size_v=width_v,
        dtype=torch.float32,
    )
    builder_cls = (
        triton.TritonAttentionDiffKVMetadataBuilder
        if diffkv
        else triton_parent.TritonAttentionMetadataBuilder
    )
    builder = builder_cls(spec, [], cfg, torch.device("cpu"))
    assert builder.num_heads_kv == num_kv_heads
    assert builder.headdim == head_size
    threshold = triton_parent.MIN_LAUNCH_GRID_SIZE_2D // num_kv_heads
    if full_graph:
        threshold = min(captures, key=lambda size: abs(size - threshold))
    assert builder.seq_threshold_3D == threshold
    assert builder.softmax_segm_output.shape == (
        threshold,
        8,
        builder.num_par_softmax_segments,
        next_power_of_2(width_v),
    )


@pytest.mark.parametrize("head_size_v", [64, 128, 256])
def test_diffkv_builder_does_not_allocate_discarded_qk_scratch(
    cpu_boundaries, monkeypatch, head_size_v
):
    allocations = []
    real_empty = torch.empty

    def track_empty(*args, **kwargs):
        result = real_empty(*args, **kwargs)
        if result.ndim == 4:
            allocations.append(result)
        return result

    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=192,
        head_size_v=head_size_v,
        dtype=torch.float32,
    )
    monkeypatch.setattr(torch, "empty", track_empty)
    builder = triton.TritonAttentionDiffKVMetadataBuilder(
        spec, [], _config(), torch.device("cpu")
    )
    assert len(allocations) == 1
    assert allocations[0] is builder.softmax_segm_output
    assert allocations[0].shape[-1] == head_size_v
