#!/usr/bin/env python3
"""Exercise MiMo EXL3 dense and rank-sliced Trellis arithmetic directly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.quantization.exl3 import Exl3LinearMethod

# vLLM's canonical FP16 kernel comparison contract.  These values are kept in
# tests/kernels/allclose_default.py and sourced there to PyTorch's transformer
# tests.  Batched and rowwise calls select different GEMM row tilings, so exact
# equality is not a valid arithmetic contract for this comparison.
FP16_ATOL = 1e-3
FP16_RTOL = 1e-3


def _assert_fp16_tiling_close(
    name: str, rows: int, batched: torch.Tensor, rowwise: torch.Tensor
) -> None:
    difference = (batched.float() - rowwise.float()).abs()
    print(
        json.dumps(
            {
                "contract": name,
                "rows": rows,
                "max_abs_error": difference.max().item(),
                "mean_abs_error": difference.mean().item(),
                "different_elements": torch.count_nonzero(difference).item(),
                "elements": difference.numel(),
                "atol": FP16_ATOL,
                "rtol": FP16_RTOL,
            },
            sort_keys=True,
        )
    )
    torch.testing.assert_close(
        batched, rowwise, rtol=FP16_RTOL, atol=FP16_ATOL
    )


def _load_tensor(model: Path, index: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(
        str(model / index[name]), framework="pt", device="cpu"
    ) as safetensor_file:
        return safetensor_file.get_tensor(name).clone()


def _holder(tensors: dict) -> SimpleNamespace:
    return SimpleNamespace(exl3_tensors=tensors)


def contract_dense(model: Path, index: dict[str, str], rank: int) -> None:
    layer = object.__new__(QKVParallelLinear)
    torch.nn.Module.__init__(layer)
    prefixes = {
        shard: f"model.layers.0.self_attn.{shard}_proj" for shard in ("q", "k", "v")
    }
    layer.trellis = _holder(
        {
            shard: _load_tensor(model, index, f"{prefix}.trellis").cuda()
            for shard, prefix in prefixes.items()
        }
    )
    layer.suh = _holder(
        {
            shard: _load_tensor(model, index, f"{prefix}.suh").cuda()
            for shard, prefix in prefixes.items()
        }
    )
    layer.svh = _holder(
        {
            shard: _load_tensor(model, index, f"{prefix}.svh").cuda()
            for shard, prefix in prefixes.items()
        }
    )
    layer.mcg = _holder(
        {
            shard: _load_tensor(model, index, f"{prefix}.mcg").cuda()
            for shard, prefix in prefixes.items()
        }
    )
    layer.mul1 = _holder({})
    layer.exl3_shard_ids = ["q", "k", "v"]
    layer.exl3_output_partition_sizes = [6144, 384, 256]
    layer.exl3_tp_size = 2
    layer.exl3_tp_rank = rank
    layer.exl3_parallel_mode = "column"
    layer.exl3_input_size_per_partition = 4096
    layer.num_kv_head_replicas = 1
    layer.total_num_kv_heads = 4
    layer.num_kv_heads = 2
    layer.head_size = 192
    layer.v_head_size = 128

    Exl3LinearMethod._validate_loaded_tensors(layer)
    Exl3LinearMethod._shard_tensors_for_tensor_parallel(layer)
    Exl3LinearMethod._validate_loaded_tensors(layer)
    method = object.__new__(Exl3LinearMethod)
    generator = torch.Generator(device="cpu").manual_seed(700 + rank)
    for rows in (1, 2, 4, 5):
        x = torch.randn(rows, 4096, generator=generator, dtype=torch.float16).cuda()
        batched = method.apply(layer, x)
        rowwise = torch.cat([method.apply(layer, row[None]) for row in x], dim=0)
        assert batched.shape == (rows, 6784)
        assert batched.dtype == torch.float16
        assert torch.isfinite(batched).all()
        _assert_fp16_tiling_close("exl3-dense", rows, batched, rowwise)


def _load_expert(
    model: Path,
    index: dict[str, str],
    rank: int,
    projection: str,
    suffix: str,
) -> torch.Tensor:
    name = f"model.layers.1.mlp.experts.0.{projection}.rank{rank}.{suffix}"
    return _load_tensor(model, index, name).cuda().contiguous()


def contract_trellis(model: Path, index: dict[str, str], rank: int) -> None:
    from sparkinfer.moe import trellis_moe

    gate_trellis = _load_expert(model, index, rank, "gate_proj", "trellis")
    up_trellis = _load_expert(model, index, rank, "up_proj", "trellis")
    down_trellis = _load_expert(model, index, rank, "down_proj", "trellis")
    gate_suh = _load_expert(model, index, rank, "gate_proj", "suh")[None]
    up_suh = _load_expert(model, index, rank, "up_proj", "suh")[None]
    gate_svh = _load_expert(model, index, rank, "gate_proj", "svh")
    up_svh = _load_expert(model, index, rank, "up_proj", "svh")
    down_suh = _load_expert(model, index, rank, "down_proj", "suh")
    down_svh = _load_expert(model, index, rank, "down_proj", "svh")[None]
    marker = _load_expert(model, index, rank, "gate_proj", "mcg")
    w13 = torch.stack((gate_trellis, up_trellis), dim=0)[:, None]
    w2 = down_trellis[None]
    intermediate_rotations = torch.cat((gate_svh, up_svh, down_suh))[None]
    tile_config = (64, 256, 64, 256)
    weights = trellis_moe.prepare_weights(
        w13,
        w2,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        codebook="mcg",
        mcg=marker,
        tile_config=tile_config,
    )
    caps = trellis_moe.Caps(
        max_tokens=32,
        num_topk=1,
        num_experts=1,
        hidden_size=4096,
        intermediate_size=1024,
        route_num_experts=1,
        block_size_m=8,
        trellis_bits=4,
        tile_config=tile_config,
        input_dtype=torch.float16,
        device=torch.device("cuda"),
    )
    plan = trellis_moe.plan(caps)
    scratch_spec = plan.scratch_specs()[0]
    scratch = torch.empty(
        scratch_spec.shape,
        dtype=scratch_spec.dtype,
        device=scratch_spec.device,
    )
    generator = torch.Generator(device="cpu").manual_seed(900 + rank)

    def run(x: torch.Tensor) -> torch.Tensor:
        rows = x.shape[0]
        binding = trellis_moe.bind(
            plan,
            scratch=scratch,
            a=x,
            weights=weights,
            topk_weights=torch.ones(rows, 1, dtype=torch.float32, device="cuda"),
            topk_ids=torch.zeros(rows, 1, dtype=torch.long, device="cuda"),
        )
        return trellis_moe.run(binding=binding)

    for rows in (1, 2, 4, 5):
        x = torch.randn(rows, 4096, generator=generator, dtype=torch.float16).cuda()
        # trellis_moe.run returns a view into the binding's shared scratch
        # arena.  Preserve each observation before the next binding reuses it.
        batched = run(x).clone()
        rowwise = torch.cat([run(row[None]).clone() for row in x], dim=0)
        assert batched.shape == (rows, 4096)
        assert torch.isfinite(batched).all()
        _assert_fp16_tiling_close("sparkinfer-trellis", rows, batched, rowwise)

    static_x = torch.randn(4, 4096, generator=generator, dtype=torch.float16).cuda()
    static_weights = torch.ones(4, 1, dtype=torch.float32, device="cuda")
    static_ids = torch.zeros(4, 1, dtype=torch.long, device="cuda")
    binding = trellis_moe.bind(
        plan,
        scratch=scratch,
        a=static_x,
        weights=weights,
        topk_weights=static_weights,
        topk_ids=static_ids,
    )
    eager = trellis_moe.run(binding=binding).clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = trellis_moe.run(binding=binding)
    torch.testing.assert_close(captured_output, eager, rtol=0, atol=0)
    replacement = torch.randn(4, 4096, generator=generator, dtype=torch.float16).cuda()
    static_x.copy_(replacement)
    graph.replay()
    torch.cuda.synchronize()
    replay = captured_output.clone()
    eager_replay = run(replacement)
    torch.testing.assert_close(replay, eager_replay, rtol=0, atol=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--rank", required=True, type=int, choices=(0, 1))
    args = parser.parse_args()
    assert torch.cuda.is_available()
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    contract_dense(args.model, index, args.rank)
    print(f"PASS exl3-dense-rank={args.rank}-m1-m2-m4-m5-qkv-vtrim")
    contract_trellis(args.model, index, args.rank)
    print(f"PASS sparkinfer-trellis-rank={args.rank}-m1-m2-m4-m5-graph")


if __name__ == "__main__":
    main()
