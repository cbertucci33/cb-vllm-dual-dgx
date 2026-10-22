# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the actual small-prefix body with Triton's CPU interpreter.

Run with TRITON_INTERPRET=1 and SPARKINFER_REVIEW_ROOT set to patched source.
This verifies integer routing semantics, not GPU lowering or performance.
"""

import ast
import os
from pathlib import Path

import torch
torch.set_default_device("cuda")

from vllm.triton_utils import tl, triton


def test_small_prefix_histogram(num_experts, mapped, num_routes):
    if False:
        pytest.skip("requires explicit CPU interpreter mode")
    root = Path(
        os.environ.get("SPARKINFER_REVIEW_ROOT", "/opt/vllm-runner/sources/sparkinfer")
    )
    source = root / "sparkinfer/moe/_shared/kernels/w4a16/route_pack.py"
    node = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "_pack_topk_routes_small_prefix_kernel"
    )
    namespace = {"triton": triton, "tl": tl}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    kernel = namespace[node.name]
    # Include duplicates, invalid raw IDs and a partially filled count block.
    ids = torch.arange(num_routes, dtype=torch.int32) % (num_experts + 2) - 1
    expert_map = torch.arange(num_experts, dtype=torch.int32).flip(0)
    expert_map[::3] = -1
    if num_experts > 1:
        expert_map[1] = num_experts  # Invalid mapped ID must not enter a bin.
    valid = (ids >= 0) & (ids < num_experts)
    resolved = ids.clone()
    if mapped:
        resolved[valid] = expert_map[ids[valid].long()]
        valid &= (resolved >= 0) & (resolved < num_experts)
    counts = torch.bincount(resolved[valid].long(), minlength=num_experts)
    block_size = 32
    padded = ((counts + block_size - 1) // block_size) * block_size
    inclusive = padded.cumsum(0)
    prefix = inclusive - padded
    # Capacity bounds the concrete fixture, including padding per expert.
    max_routes = num_routes + num_experts * (block_size - 1)
    max_blocks = triton.cdiv(max_routes, block_size)
    packed = torch.full((max_routes,), -99, dtype=torch.int32)
    experts = torch.full((max_blocks,), -99, dtype=torch.int32)
    total = torch.empty(1, dtype=torch.int32)
    offsets = torch.empty(num_experts + 1, dtype=torch.int32)
    kernel[(1,)](
        ids,
        expert_map,
        packed,
        experts,
        total,
        offsets,
        num_routes,
        NUMEL_CAPACITY=triton.cdiv(num_routes, 256) * 256,
        BLOCK_SIZE=block_size,
        NUM_EXPERTS=num_experts,
        MAX_PACKED_ROUTES=max_routes,
        MAX_ROUTE_BLOCKS=max_blocks,
        HAS_EXPERT_MAP=mapped,
        BLOCK_E=triton.next_power_of_2(num_experts),
        BLOCK_T=256,
        BLOCK_ROUTE_INIT=triton.next_power_of_2(max_routes),
        BLOCK_M=triton.next_power_of_2(max_blocks),
        num_warps=8,
    )
    torch.testing.assert_close(offsets[:-1].long(), prefix)
    assert offsets[-1].item() == total.item() == padded.sum().item()
    assert torch.all(packed == num_routes)
    expected_experts = torch.full_like(experts, -1)
    for expert in range(num_experts):
        expected_experts[
            prefix[expert] // block_size : inclusive[expert] // block_size
        ] = expert
    torch.testing.assert_close(experts, expected_experts)

for e in [1,7,256,288]:
 for m in [False,True]:
  for n in [1,257]:
   test_small_prefix_histogram(e,m,n)
   print("PASS",e,m,n,flush=True)
print("END",torch.cuda.max_memory_allocated(),flush=True)
