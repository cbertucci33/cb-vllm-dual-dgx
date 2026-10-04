# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the Triton DiffKV unified-attention kernel.
"""

from unittest.mock import Mock

import pytest
import torch

from tests.kernels.attention.test_triton_unified_attention import ref_paged_attn
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    set_random_seed,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv,
)
from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    kernel_unified_attention_diffkv,
    unified_attention_diffkv,
)

pytestmark = pytest.mark.skip_global_cleanup

DEVICE_TYPE = current_platform.device_type

# (num_query_heads, num_kv_heads): MHA, GQA, and the num_kv_heads==1
# (degenerate-stride) case.
NUM_HEADS = [(4, 4), (8, 2), (5, 1)]
# (head_size_qk, head_size_v).  (192, 128) is the canonical asymmetric
# DiffKV shape; FA4 on Blackwell only supports head_size>128 when it is
# 192, and FA3 on Hopper supports it too -- so this pair is runnable on
# both.  (128, 128) keeps the equal-dim path covered through the DiffKV
# kernel.
HEAD_SIZES = [(128, 128), (192, 128)]
BLOCK_SIZES = [16]
DTYPES = [torch.bfloat16]

NUM_BLOCKS = 2048

# 0: 2D decode kernel; 8: 3D (split-KV) decode kernel.
SEQ_THRESHOLD_3D_VALUES = [0, 8]

NUM_PAR_SOFTMAX_SEGMENTS = 16


def _alloc_segm_buffers(seq_threshold_3D: int, num_query_heads: int, head_size_v: int):
    """Allocate the split-KV softmax scratch (last dim == head_size_v)."""
    head_size_v_padded = next_power_of_2(head_size_v)
    segm_output = torch.empty(
        (
            seq_threshold_3D,
            num_query_heads,
            NUM_PAR_SOFTMAX_SEGMENTS,
            head_size_v_padded,
        ),
        dtype=torch.float32,
    )
    segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, NUM_PAR_SOFTMAX_SEGMENTS),
        dtype=torch.float32,
    )
    segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, NUM_PAR_SOFTMAX_SEGMENTS),
        dtype=torch.float32,
    )
    return segm_output, segm_max, segm_expsum


# MiMo TP2 global layer; forcing SM12x lets CUDA CI take BLOCK_M=32.
@pytest.mark.skipif(not current_platform.is_cuda(), reason="SM12x tuning is CUDA-only")
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize(
    ("query_len", "expected_block_m"), [(63, 16), (64, 32), (65, 32), (257, 32)]
)
@torch.inference_mode()
def test_triton_unified_attn_diffkv_prefill_block_m(
    block_size: int,
    query_len: int,
    expected_block_m: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(current_platform, "is_device_capability_family", lambda _: True)
    kernel_run = Mock(wraps=kernel_unified_attention_diffkv.run)
    monkeypatch.setattr(kernel_unified_attention_diffkv, "run", kernel_run)
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    query_lens = [1, query_len]
    kv_lens = [1344, 294]
    num_query_heads, num_kv_heads = 32, 2
    head_size_qk, head_size_v = 192, 128
    dtype = torch.bfloat16
    scale = head_size_qk**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size_qk, dtype=dtype)
    kv_cache = torch.randn(
        NUM_BLOCKS,
        block_size,
        num_kv_heads,
        head_size_qk + head_size_v,
        dtype=dtype,
    )
    key_cache = kv_cache[..., :head_size_qk]
    value_cache = kv_cache[..., head_size_qk:]

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max(kv_lens) + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS, (len(query_lens), max_num_blocks_per_seq), dtype=torch.int32
    )

    ref_out = ref_paged_attn(
        query.float(),
        key_cache.float(),
        value_cache.float(),
        query_lens,
        kv_lens,
        block_tables,
        scale,
    ).to(dtype)

    triton_out = torch.empty(sum(query_lens), num_query_heads, head_size_v, dtype=dtype)
    unified_attention_diffkv(
        q=query,
        k=key_cache,
        v=value_cache,
        out=triton_out,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        max_seqlen_q=max(query_lens),
    )

    assert kernel_run.call_args.kwargs["BLOCK_M"] == expected_block_m
    torch.testing.assert_close(triton_out, ref_out, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(89),
    reason="FP8 DiffKV requires CUDA SM89+",
)
@pytest.mark.parametrize(
    ("query_lens", "kv_lens", "expected_block_q"),
    [
        ([2], [8193], 2),
        ([4], [8195], 4),
        ([4], [62287], 4),
        ([2, 4], [8193, 777], 4),
    ],
)
@torch.inference_mode()
def test_triton_unified_attn_diffkv_mimo_fp8_whole_verify_3d(
    query_lens: list[int],
    kv_lens: list[int],
    expected_block_q: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the production MiMo FP8 whole-verify tensor contract."""
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    num_query_heads, num_kv_heads = 32, 2
    head_size_qk, head_size_v = 192, 128
    block_size = 16
    num_tokens = sum(query_lens)
    num_seqs = len(query_lens)
    max_kv_len = max(kv_lens)
    num_blocks = (sum(kv_lens) + block_size - 1) // block_size + num_seqs
    fp8_dtype = current_platform.fp8_dtype()

    query_bf16 = torch.randn(
        num_tokens, num_query_heads, head_size_qk, dtype=torch.bfloat16
    )
    kv_bf16 = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size_qk + head_size_v,
        dtype=torch.bfloat16,
    )

    # Non-power-of-two scales ensure that all three descales affect the result.
    q_descale = torch.tensor(0.2, dtype=torch.float32)
    k_descale = torch.tensor(0.3, dtype=torch.float32)
    v_descale = torch.tensor(0.7, dtype=torch.float32)
    query_fp8 = (query_bf16 / q_descale).to(fp8_dtype)
    key_fp8 = (kv_bf16[..., :head_size_qk] / k_descale).to(fp8_dtype)
    value_fp8 = (kv_bf16[..., head_size_qk:] / v_descale).to(fp8_dtype)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.empty(num_seqs, max_num_blocks_per_seq, dtype=torch.int32)
    next_block = 0
    for seq_idx, kv_len in enumerate(kv_lens):
        blocks_for_seq = (kv_len + block_size - 1) // block_size
        block_tables[seq_idx, :blocks_for_seq] = torch.arange(
            next_block, next_block + blocks_for_seq, dtype=torch.int32
        )
        block_tables[seq_idx, blocks_for_seq:] = 0
        next_block += blocks_for_seq

    ref_out = ref_paged_attn(
        query_fp8.float() * q_descale,
        key_fp8.float() * k_descale,
        value_fp8.float() * v_descale,
        query_lens,
        kv_lens,
        block_tables,
        head_size_qk**-0.5,
    ).to(torch.bfloat16)

    scratch_tokens = 64
    segm_output, segm_max, segm_expsum = _alloc_segm_buffers(
        scratch_tokens, num_query_heads, head_size_v
    )
    triton_out = torch.full_like(ref_out, float("nan"))
    kernel_run = Mock(wraps=kernel_unified_attention_diffkv.run)
    monkeypatch.setattr(kernel_unified_attention_diffkv, "run", kernel_run)

    unified_attention_diffkv(
        q=query_fp8,
        k=key_fp8,
        v=value_fp8,
        out=triton_out,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        softmax_scale=head_size_qk**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        max_seqlen_q=max(query_lens),
        seq_threshold_3D=scratch_tokens,
        num_par_softmax_segments=NUM_PAR_SOFTMAX_SEGMENTS,
        softmax_segm_output=segm_output,
        softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_expsum,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )

    launch = kernel_run.call_args.kwargs
    assert launch["IS_3D"] is True
    assert launch["BLOCK_Q"] == expected_block_q
    assert launch["BLOCK_M"] == expected_block_q * 16
    assert launch["TILE_SIZE"] == 32
    assert launch["USE_Q_SCALE"] is True
    assert launch["USE_KV_SCALES"] is True
    assert not torch.isnan(triton_out).any()
    torch.testing.assert_close(triton_out, ref_out, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(89),
    reason="FP8 DiffKV requires CUDA SM89+",
)
@torch.inference_mode()
def test_mimo_static_fp8_query_quantization(default_vllm_config) -> None:
    """Match the scalar-scale query quantization used by target attention."""
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)
    query = torch.randn(4, 32 * 192, dtype=torch.bfloat16)
    scale = torch.tensor(0.037, dtype=torch.float32)
    query[0, 0] = 0
    query[0, 1] = 20
    query[0, 2] = -20
    quant = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)

    actual, returned_scale = quant(query, scale)

    assert actual.dtype == current_platform.fp8_dtype()
    assert actual.shape == query.shape
    assert actual.stride() == query.stride()
    assert torch.isfinite(actual.float()).all()
    assert actual[0, 0].float().item() == 0
    torch.testing.assert_close(returned_scale, scale)
    expected = query.float().clamp(-448 * float(scale), 448 * float(scale))
    torch.testing.assert_close(
        actual.float() * scale,
        expected,
        atol=float(scale) * 0.55,
        rtol=0.08,
    )


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(89),
    reason="FP8 DiffKV requires CUDA SM89+",
)
@torch.inference_mode()
def test_mimo_fp8_diffkv_cache_write_contract() -> None:
    """Write asymmetric K/V across block boundaries with production scales."""
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)
    num_tokens, num_kv_heads, block_size = 5, 2, 16
    key = torch.randn(num_tokens, num_kv_heads, 192, dtype=torch.bfloat16)
    value = torch.empty(num_tokens, num_kv_heads, 128, dtype=torch.bfloat16).uniform_(
        -2.5, 2.5
    )
    slots = torch.tensor([0, 15, 16, 31, -1], dtype=torch.int64)
    k_scale = torch.tensor(0.041, dtype=torch.float32)
    v_scale = torch.tensor(0.007, dtype=torch.float32)
    cache_bytes = torch.zeros(2, block_size, num_kv_heads, 192 + 128, dtype=torch.uint8)

    triton_reshape_and_cache_flash_diffkv(
        key,
        value,
        cache_bytes,
        slots,
        "fp8_e4m3",
        k_scale,
        v_scale,
    )

    cache = cache_bytes.view(current_platform.fp8_dtype())
    for token_idx, slot in enumerate(slots[:-1].tolist()):
        block_idx, block_offset = divmod(slot, block_size)
        torch.testing.assert_close(
            cache[block_idx, block_offset, :, :192].float() * k_scale,
            key[token_idx].float(),
            atol=float(k_scale),
            rtol=0.125,
        )
        torch.testing.assert_close(
            cache[block_idx, block_offset, :, 192:].float() * v_scale,
            value[token_idx].float(),
            atol=float(v_scale),
            rtol=0.125,
        )
    selected = torch.zeros(2, block_size, dtype=torch.bool)
    for slot in slots[:-1].tolist():
        selected[slot // block_size, slot % block_size] = True
    assert torch.count_nonzero(cache_bytes[~selected]).item() == 0


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(89),
    reason="FP8 DiffKV requires CUDA SM89+",
)
@pytest.mark.parametrize(
    ("query_lens", "kv_lens", "window_size", "seq_threshold_3d"),
    [
        ([63], [294], (-1, -1), 0),
        ([1], [2011], (-1, -1), 0),
        ([1], [8193], (-1, -1), 64),
        ([4], [8195], (127, 0), 64),
    ],
)
@torch.inference_mode()
def test_mimo_fp8_diffkv_route_parity(
    query_lens: list[int],
    kv_lens: list[int],
    window_size: tuple[int, int],
    seq_threshold_3d: int,
) -> None:
    """Cover FP8 prefill, 2D/3D decode, and the sliding-window route."""
    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)
    num_query_heads, num_kv_heads = 32, 2
    head_size_qk, head_size_v, block_size = 192, 128, 16
    num_blocks = (sum(kv_lens) + block_size - 1) // block_size + len(kv_lens)
    fp8_dtype = current_platform.fp8_dtype()
    q_scale = torch.tensor(0.2, dtype=torch.float32)
    k_scale = torch.tensor(0.3, dtype=torch.float32)
    v_scale = torch.tensor(0.7, dtype=torch.float32)
    query = (torch.randn(sum(query_lens), num_query_heads, head_size_qk) / q_scale).to(
        fp8_dtype
    )
    kv = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size_qk + head_size_v,
    )
    key = (kv[..., :head_size_qk] / k_scale).to(fp8_dtype)
    value = (kv[..., head_size_qk:] / v_scale).to(fp8_dtype)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(0)
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)
    max_blocks = (max(kv_lens) + block_size - 1) // block_size
    block_tables = torch.zeros(len(kv_lens), max_blocks, dtype=torch.int32)
    next_block = 0
    for seq_idx, kv_len in enumerate(kv_lens):
        blocks = (kv_len + block_size - 1) // block_size
        block_tables[seq_idx, :blocks] = torch.arange(next_block, next_block + blocks)
        next_block += blocks
    sliding_window = None if window_size == (-1, -1) else window_size[0] + 1
    reference = ref_paged_attn(
        query.float() * q_scale,
        key.float() * k_scale,
        value.float() * v_scale,
        query_lens,
        kv_lens,
        block_tables,
        head_size_qk**-0.5,
        sliding_window=sliding_window,
    ).to(torch.bfloat16)
    segm_output, segm_max, segm_expsum = _alloc_segm_buffers(
        seq_threshold_3d, num_query_heads, head_size_v
    )
    actual = torch.empty_like(reference)
    unified_attention_diffkv(
        q=query,
        k=key,
        v=value,
        out=actual,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        softmax_scale=head_size_qk**-0.5,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=0,
        max_seqlen_q=max(query_lens),
        seq_threshold_3D=seq_threshold_3d,
        num_par_softmax_segments=NUM_PAR_SOFTMAX_SEGMENTS,
        softmax_segm_output=segm_output,
        softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_expsum,
        q_descale=q_scale,
        k_descale=k_scale,
        v_descale=v_scale,
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 1328), (5, 18), (129, 463)],  # mixed prefill + decode
        [(1, 523), (1, 37), (1, 2011)],  # decode-only (exercises 3D path)
    ],
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_sizes", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 128])
@pytest.mark.parametrize("soft_cap", [None, 50.0])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn_diffkv_vs_fa(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_sizes: tuple[int, int],
    sliding_window: int | None,
    soft_cap: float | None,
    dtype: torch.dtype,
    block_size: int,
    seq_threshold_3D: int,
) -> None:
    head_size_qk, head_size_v = head_sizes

    # DiffKV requires FA3 (Hopper) / FA4 (Blackwell) as the reference.
    fa_version = get_flash_attn_version(head_size=head_size_qk, head_size_v=head_size_v)
    if not is_flash_attn_varlen_func_available() or fa_version not in (3, 4):
        pytest.skip(f"FA DiffKV needs FA3/FA4 (got version {fa_version}).")

    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    torch.set_default_device(DEVICE_TYPE)
    set_random_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size_qk**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size_qk, dtype=dtype)
    # Packed KV cache: [num_blocks, block_size, num_kv_heads, hqk + hv].
    kv_cache = torch.randn(
        NUM_BLOCKS,
        block_size,
        num_kv_heads,
        head_size_qk + head_size_v,
        dtype=dtype,
    )
    key_cache = kv_cache[..., :head_size_qk]
    value_cache = kv_cache[..., head_size_qk:]

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    # ---- FlashAttention DiffKV (ground truth) ---------------------------
    # Mirror the backend: fix degenerate strides on size-1 dims so FA's
    # TMA path sees ≥16-byte-aligned strides (matters for num_kv_heads==1).
    fa_k = canonicalize_singleton_dim_strides(key_cache)
    fa_v = canonicalize_singleton_dim_strides(value_cache)
    fa_out = torch.empty(sum(query_lens), num_query_heads, head_size_v, dtype=dtype)
    flash_attn_varlen_func(
        q=query,
        k=fa_k,
        v=fa_v,
        out=fa_out,
        cu_seqlens_q=cu_query_lens,
        max_seqlen_q=max_query_len,
        seqused_k=kv_lens_t,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=list(window_size),
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        fa_version=fa_version,
    )

    # ---- Triton DiffKV --------------------------------------------------
    segm_output, segm_max, segm_expsum = _alloc_segm_buffers(
        seq_threshold_3D, num_query_heads, head_size_v
    )
    triton_out = torch.empty(sum(query_lens), num_query_heads, head_size_v, dtype=dtype)
    unified_attention_diffkv(
        q=query,
        k=key_cache,
        v=value_cache,
        out=triton_out,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        max_seqlen_q=max_query_len,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=NUM_PAR_SOFTMAX_SEGMENTS,
        softmax_segm_output=segm_output,
        softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_expsum,
    )

    (
        torch.testing.assert_close(triton_out, fa_out, atol=2e-2, rtol=2e-2),
        f"triton vs FA max abs diff: {torch.max(torch.abs(triton_out - fa_out))}",
    )
