# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Previous acceptance belongs to the stored window, not the new query length."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


def _update():
    if not INTERPRET:
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
            causal_conv1d_update,
        )

        return causal_conv1d_update
    import triton
    import triton.language as tl

    # Execute the NVIDIA wrapper/kernel, not the module's CPU override.
    path = ROOT / "vllm/model_executor/layers/mamba/ops/causal_conv1d.py"
    nodes = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"causal_conv1d_update", "_causal_conv1d_update_kernel"}
    ]
    assert len(nodes) == 2
    ns = {
        "torch": torch,
        "triton": triton,
        "tl": tl,
        "NULL_BLOCK_ID": 0,
        "current_platform": SimpleNamespace(is_arch_support_pdl=lambda: False),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns["causal_conv1d_update"]


@pytest.fixture(scope="module")
def update():
    if not INTERPRET and not torch.cuda.is_available():
        pytest.skip("requires NVIDIA CUDA or explicit Triton CPU interpreter")
    return _update()


@pytest.mark.parametrize("num_drafts", range(1, 8))
@pytest.mark.parametrize("width", [2, 4, 6])
@pytest.mark.parametrize("varlen", [False, True])
def test_previous_acceptance_can_exceed_current_query(
    update, num_drafts, width, varlen
):
    device = "cpu" if INTERPRET else "cuda"
    capacity = width - 1 + num_drafts
    dim = 8
    maximum = num_drafts + 1
    for current_len in range(1, maximum + 1):
        for accepted in range(1, maximum + 1):
            state = (
                torch.arange(
                    3 * dim * capacity, device=device, dtype=torch.float32
                ).reshape(3, dim, capacity)
                / 32
                + 1
            )
            before = state.clone()
            weight = (
                torch.arange(1, width + 1, device=device, dtype=torch.float32)
                .expand(dim, -1)
                .contiguous()
            )
            x = (
                torch.arange(
                    dim * current_len, device=device, dtype=torch.float32
                ).reshape(1, dim, current_len)
                + 1
            ) / 8
            history = before[1, :, accepted - 1 : accepted - 1 + width - 1]
            joined = torch.cat((history, x[0]), dim=-1)
            expected = (
                (joined.unfold(-1, width, 1) * weight[:, None, :]).sum(-1).unsqueeze(0)
            )
            if varlen:
                query = x[0].t().contiguous()
                kwargs = {
                    "query_start_loc": torch.tensor(
                        [0, current_len], dtype=torch.int32, device=device
                    ),
                    "max_query_len": maximum,
                }
                expected = expected[0].t()
            else:
                query, kwargs = x.clone(), {}
            out = torch.full_like(query, torch.nan)
            actual = update(
                query,
                state,
                weight,
                conv_state_indices=torch.tensor([1], dtype=torch.int32, device=device),
                num_accepted_tokens=torch.tensor(
                    [accepted], dtype=torch.int32, device=device
                ),
                out=out,
                **kwargs,
            )
            torch.testing.assert_close(actual, expected)
            written = width - 2 + current_len
            torch.testing.assert_close(state[1, :, :written], joined[:, 1:])
            torch.testing.assert_close(state[1, :, written:], before[1, :, written:])
            torch.testing.assert_close(state[[0, 2]], before[[0, 2]])


@pytest.mark.parametrize("accepted", [0, 9])
@pytest.mark.parametrize("varlen", [False, True])
def test_acceptance_outside_stored_window_does_not_touch_state(
    update, accepted, varlen
):
    device = "cpu" if INTERPRET else "cuda"
    state = torch.ones(3, 8, 10, device=device)
    before = state.clone()
    weight = torch.ones(8, 4, device=device)
    query = torch.ones((2, 8) if varlen else (1, 8, 2), device=device)
    kwargs = (
        {
            "query_start_loc": torch.tensor([0, 2], dtype=torch.int32, device=device),
            "max_query_len": 8,
        }
        if varlen
        else {}
    )
    actual = update(
        query,
        state,
        weight,
        conv_state_indices=torch.tensor([1], dtype=torch.int32, device=device),
        num_accepted_tokens=torch.tensor([accepted], dtype=torch.int32, device=device),
        out=torch.empty_like(query),
        **kwargs,
    )
    torch.testing.assert_close(actual, torch.zeros_like(actual))
    torch.testing.assert_close(state, before)
