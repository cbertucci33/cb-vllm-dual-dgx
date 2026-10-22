# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-interpreter regression for the FP8 grouped-tile/block-table boundary."""

import ast
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import triton
import triton.language as tl
from triton.runtime.interpreter import InterpreterBuilder

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="Requires the Triton CPU interpreter, not a GPU kernel launch.",
)


def _load_functions(path: Path, globals_: dict) -> dict:
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id.startswith("_SPEC_3D_")
                for target in node.targets
            )
        )
    ]
    namespace = dict(torch=torch, triton=triton, tl=tl, Any=Any)
    namespace.update(globals_)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float32])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("window", [(-1, -1), (1023, 0)])
def test_diffkv_partial_table_load(
    dtype: torch.dtype, block_size: int, window: tuple[int, int]
) -> None:
    root = Path(__file__).resolve().parents[3]
    ops = root / "vllm/v1/attention/ops"
    source = ops / "triton_unified_attention_diffkv.py"
    helpers = _load_functions(ops / "triton_attention_helpers.py", {})
    helpers.update(
        current_platform=SimpleNamespace(
            is_cuda=lambda: True,
            is_device_capability_family=lambda family: family == 120,
        ),
        is_batch_invariant=False,
        # Trigger the existing local grouped-verify wave gate on CPU.
        num_compute_units=lambda _: 1,
    )
    production = _load_functions(source, helpers)
    tree = ast.parse(source.read_text())
    block_load = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "physical_block_idx"
            for target in node.targets
        )
    )

    block_start = block_load.lineno
    block_end = block_load.end_lineno
    assert block_start is not None and block_end is not None

    torch.manual_seed(6)
    query = (torch.randn(8, 64, 192) * 0.1).to(dtype)
    packed = (torch.randn(1, block_size, 4, 320) * 0.1).to(dtype)
    key, value = packed[..., :192], packed[..., 192:]
    output = torch.empty(8, 64, 128)
    table = torch.tensor([[0]], dtype=torch.int32)
    table_start = table.data_ptr()
    table_end = table_start + table.numel() * table.element_size()
    original_load = InterpreterBuilder.create_masked_load
    load_count = 0

    def checked_load(builder, pointers, mask, other, *args):
        nonlocal load_count
        if any(
            frame.function == "kernel_unified_attention_diffkv"
            and frame.filename == str(source)
            and frame.lineno is not None
            and block_start <= frame.lineno <= block_end
            for frame in inspect.stack()
        ):
            active = pointers.data[mask.data]
            assert ((active >= table_start) & (active < table_end)).all()
            load_count += 1
        return original_load(builder, pointers, mask, other, *args)

    InterpreterBuilder.create_masked_load = checked_load
    try:
        production["unified_attention_diffkv"](
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=torch.tensor([0, 8], dtype=torch.int32),
            seqused_k=torch.tensor([8], dtype=torch.int32),
            softmax_scale=192**-0.5,
            causal=True,
            window_size=window,
            block_table=table,
            softcap=0.0,
            max_seqlen_q=8,
            seq_threshold_3D=8,
            num_par_softmax_segments=4,
            softmax_segm_output=torch.empty(8, 64, 4, 128),
            softmax_segm_max=torch.empty(8, 64, 4),
            softmax_segm_expsum=torch.empty(8, 64, 4),
        )
    finally:
        InterpreterBuilder.create_masked_load = original_load

    key_heads = key[0, :8].float().repeat_interleave(16, dim=1)
    value_heads = value[0, :8].float().repeat_interleave(16, dim=1)
    scores = torch.einsum("thd,shd->hts", query.float(), key_heads) * 192**-0.5
    scores.masked_fill_(~torch.ones(8, 8, dtype=torch.bool).tril(), -float("inf"))
    probability = (scores - scores.max(dim=-1, keepdim=True).values).exp()
    denominator = probability.sum(dim=-1)
    # Match the kernel's explicit P-to-V-dtype cast before the P @ V dot.
    reference = torch.einsum(
        "hts,shv->thv", probability.to(dtype).float(), value_heads
    ) / denominator.transpose(0, 1).unsqueeze(-1)
    torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-5)
    assert load_count > 0
    assert not torch.cuda.is_initialized()
