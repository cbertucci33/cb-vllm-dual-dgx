#!/usr/bin/env python3
"""Two-node MiMo TP2 runner agreement contract for clean-image qualification."""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

from vllm.platforms import current_platform
from vllm.v1.attention.backends.triton_attn_diffkv import (
    DIFFKV_NUM_PAR_SOFTMAX_SEGMENTS,
)
from vllm.v1.attention.ops.triton_unified_attention_diffkv import (
    unified_attention_diffkv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--master", required=True)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master}:{args.port}",
        rank=args.rank,
        world_size=2,
    )
    try:
        assert torch.cuda.get_device_capability() == (12, 1)
        fp8_dtype = current_platform.fp8_dtype()
        query_len, kv_len = 4, 257
        num_query_heads, num_kv_heads = 32, 2
        head_size_qk, head_size_v, block_size = 192, 128, 16
        num_segments = DIFFKV_NUM_PAR_SOFTMAX_SEGMENTS
        q_scale = torch.tensor(0.037, dtype=torch.float32, device=device)
        k_scale = torch.tensor(0.041, dtype=torch.float32, device=device)
        v_scale = torch.tensor(0.007, dtype=torch.float32, device=device)
        num_blocks = (kv_len + block_size - 1) // block_size

        generator = torch.Generator(device="cpu").manual_seed(20261004)
        query = (
            torch.randn(
                query_len,
                num_query_heads,
                head_size_qk,
                generator=generator,
            ).to(device)
            / q_scale
        ).to(fp8_dtype)
        key = (
            torch.randn(
                num_blocks,
                block_size,
                num_kv_heads,
                head_size_qk,
                generator=generator,
            ).to(device)
            / k_scale
        ).to(fp8_dtype)
        value = (
            torch.randn(
                num_blocks,
                block_size,
                num_kv_heads,
                head_size_v,
                generator=generator,
            ).to(device)
            / v_scale
        ).to(fp8_dtype)
        block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(
            1, -1
        )
        cu_query_lens = torch.tensor([0, query_len], dtype=torch.int32, device=device)
        kv_lens = torch.tensor([kv_len], dtype=torch.int32, device=device)
        segm_output = torch.empty(
            64,
            num_query_heads,
            num_segments,
            head_size_v,
            dtype=torch.float32,
            device=device,
        )
        segm_max = torch.empty(
            64,
            num_query_heads,
            num_segments,
            dtype=torch.float32,
            device=device,
        )
        segm_expsum = torch.empty_like(segm_max)
        output = torch.empty(
            query_len,
            num_query_heads,
            head_size_v,
            dtype=torch.bfloat16,
            device=device,
        )

        unified_attention_diffkv(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_query_lens,
            seqused_k=kv_lens,
            softmax_scale=head_size_qk**-0.5,
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            softcap=0,
            max_seqlen_q=query_len,
            seq_threshold_3D=64,
            num_par_softmax_segments=num_segments,
            softmax_segm_output=segm_output,
            softmax_segm_max=segm_max,
            softmax_segm_expsum=segm_expsum,
            q_descale=q_scale,
            k_descale=k_scale,
            v_descale=v_scale,
        )
        torch.cuda.synchronize()

        geometry = torch.tensor(
            [
                query_len,
                kv_len,
                num_query_heads,
                num_kv_heads,
                head_size_qk,
                head_size_v,
                block_size,
                num_segments,
            ],
            dtype=torch.int64,
            device=device,
        )
        scales = torch.stack((q_scale, k_scale, v_scale))
        gathered_geometry = [torch.empty_like(geometry) for _ in range(2)]
        gathered_scales = [torch.empty_like(scales) for _ in range(2)]
        gathered_output = [torch.empty_like(output) for _ in range(2)]
        dist.all_gather(gathered_geometry, geometry)
        dist.all_gather(gathered_scales, scales)
        dist.all_gather(gathered_output, output)

        assert torch.equal(gathered_geometry[0], gathered_geometry[1])
        torch.testing.assert_close(
            gathered_scales[0], gathered_scales[1], rtol=0, atol=0
        )
        torch.testing.assert_close(
            gathered_output[0], gathered_output[1], rtol=0, atol=0
        )
        assert torch.isfinite(output).all()

        collective = torch.tensor([float(args.rank + 1)], device=device)
        dist.all_reduce(collective)
        assert collective.item() == 3.0
        max_rank_diff = (
            (gathered_output[0].float() - gathered_output[1].float()).abs().max().item()
        )
        print(
            f"PASS tp2-rank={args.rank} geometry={geometry.tolist()} "
            f"max-rank-diff={max_rank_diff} "
            f"collective={collective.item()}"
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
