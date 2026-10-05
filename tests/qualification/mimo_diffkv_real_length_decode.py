#!/usr/bin/env python3
"""Validate MiMo's q=1 BF16 3D DiffKV path at the failed request lengths.

This is a qualification harness, not production runtime code.  It compares the
installed 3D split-KV kernel with both an independent FP32 paged-attention
oracle and the installed 2D kernel, then repeats the 3D check through CUDA graph
capture/replay while mutating the live query and sequence length tensors.
"""

from __future__ import annotations

import json
import math

import torch

from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    unified_attention_diffkv,
)


NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 2
HEAD_SIZE_QK = 192
HEAD_SIZE_V = 128
BLOCK_SIZE = 16
NUM_SEGMENTS = 16
SCRATCH_TOKENS = 64
SEQUENCE_LENGTHS = (26471, 26620)
ATOL = 0.02
RTOL = 0.02


def _scratch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = torch.empty(
        SCRATCH_TOKENS,
        NUM_QUERY_HEADS,
        NUM_SEGMENTS,
        HEAD_SIZE_V,
        dtype=torch.float32,
        device="cuda",
    )
    maxes = torch.empty(
        SCRATCH_TOKENS,
        NUM_QUERY_HEADS,
        NUM_SEGMENTS,
        dtype=torch.float32,
        device="cuda",
    )
    sums = torch.empty_like(maxes)
    return output, maxes, sums


def _oracle(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    blocks = math.ceil(seq_len / BLOCK_SIZE)
    physical = block_table[0, :blocks].to(torch.long)
    logical_key = key[physical].reshape(-1, NUM_KV_HEADS, HEAD_SIZE_QK)[:seq_len]
    logical_value = value[physical].reshape(-1, NUM_KV_HEADS, HEAD_SIZE_V)[:seq_len]
    result = torch.empty(
        1, NUM_QUERY_HEADS, HEAD_SIZE_V, dtype=torch.float32, device="cuda"
    )
    scale = HEAD_SIZE_QK**-0.5
    group = NUM_QUERY_HEADS // NUM_KV_HEADS
    for kv_head in range(NUM_KV_HEADS):
        head_slice = slice(kv_head * group, (kv_head + 1) * group)
        q = query[0, head_slice].float() * scale
        scores = q @ logical_key[:, kv_head].float().T
        probs = torch.softmax(scores, dim=-1)
        result[0, head_slice] = probs @ logical_value[:, kv_head].float()
    return result.to(torch.bfloat16)


def _run(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: torch.Tensor,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> None:
    if scratch is None:
        segm_output = segm_max = segm_sum = None
        threshold = None
        segments = None
    else:
        segm_output, segm_max, segm_sum = scratch
        threshold = SCRATCH_TOKENS
        segments = NUM_SEGMENTS
    unified_attention_diffkv(
        q=query,
        k=key,
        v=value,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seq_len,
        softmax_scale=HEAD_SIZE_QK**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0,
        max_seqlen_q=1,
        seq_threshold_3D=threshold,
        num_par_softmax_segments=segments,
        softmax_segm_output=segm_output,
        softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_sum,
    )


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "relative_rms": (
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        ).item(),
    }


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(20261004)
    torch.cuda.manual_seed_all(20261004)
    max_len = max(SEQUENCE_LENGTHS)
    used_blocks = math.ceil(max_len / BLOCK_SIZE)
    total_blocks = used_blocks + 17
    packed = torch.randn(
        total_blocks,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_SIZE_QK + HEAD_SIZE_V,
        dtype=torch.bfloat16,
        device="cuda",
    )
    key = packed[..., :HEAD_SIZE_QK]
    value = packed[..., HEAD_SIZE_QK:]
    block_table = torch.randperm(total_blocks, device="cuda", dtype=torch.int64)[
        :used_blocks
    ].to(torch.int32)[None, :]
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    static_query = torch.randn(
        1,
        NUM_QUERY_HEADS,
        HEAD_SIZE_QK,
        dtype=torch.bfloat16,
        device="cuda",
    )
    static_seq_len = torch.tensor(
        [SEQUENCE_LENGTHS[0]], dtype=torch.int32, device="cuda"
    )
    output_3d = torch.empty(
        1, NUM_QUERY_HEADS, HEAD_SIZE_V, dtype=torch.bfloat16, device="cuda"
    )
    output_2d = torch.empty_like(output_3d)
    scratch = _scratch()
    results: list[dict[str, object]] = []

    for index, length in enumerate(SEQUENCE_LENGTHS):
        query = torch.randn_like(static_query) if index else static_query.clone()
        seq_len = torch.tensor([length], dtype=torch.int32, device="cuda")
        _run(
            query,
            key,
            value,
            output_3d,
            cu_seqlens_q,
            block_table,
            seq_len,
            scratch,
        )
        _run(
            query,
            key,
            value,
            output_2d,
            cu_seqlens_q,
            block_table,
            seq_len,
            None,
        )
        reference = _oracle(query, key, value, block_table, length)
        torch.cuda.synchronize()
        torch.testing.assert_close(output_3d, reference, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(output_3d, output_2d, atol=ATOL, rtol=RTOL)
        results.append(
            {
                "mode": "eager",
                "seq_len": length,
                "3d_vs_fp32": _error(output_3d, reference),
                "3d_vs_2d": _error(output_3d, output_2d),
            }
        )

    static_seq_len.fill_(SEQUENCE_LENGTHS[0])
    _run(
        static_query,
        key,
        value,
        output_3d,
        cu_seqlens_q,
        block_table,
        static_seq_len,
        scratch,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(
            static_query,
            key,
            value,
            output_3d,
            cu_seqlens_q,
            block_table,
            static_seq_len,
            scratch,
        )

    for length in SEQUENCE_LENGTHS:
        replay_query = torch.randn_like(static_query)
        static_query.copy_(replay_query)
        static_seq_len.fill_(length)
        graph.replay()
        torch.cuda.synchronize()
        reference = _oracle(replay_query, key, value, block_table, length)
        torch.testing.assert_close(output_3d, reference, atol=ATOL, rtol=RTOL)
        results.append(
            {
                "mode": "cuda_graph_replay",
                "seq_len": length,
                "3d_vs_fp32": _error(output_3d, reference),
            }
        )

    print(json.dumps({"status": "PASS", "results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
