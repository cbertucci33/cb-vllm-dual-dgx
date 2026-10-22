# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host staging allocations must ignore a caller's default device context."""

import ast
import weakref
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch


@pytest.mark.parametrize("default_device", ["cpu", "meta"])
@pytest.mark.parametrize("registration", ["success", "unavailable", "failure"])
def test_pinned_staging_always_allocates_on_host(default_device, registration):
    path = Path(__file__).resolve().parents[2] / "vllm/utils/torch_utils.py"
    function = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_pinned_uint8_buffer"
    )
    allocations = []
    registrations = []

    def empty(*args, **kwargs):
        allocations.append(kwargs.copy())
        # Exercise real device-context dispatch without allocating pinned CUDA
        # resources on a CPU-only test host.
        kwargs.pop("pin_memory", None)
        return torch.empty(*args, **kwargs)

    def register(pointer, size, flags):
        registrations.append((pointer, size, flags))
        return SimpleNamespace(value=0 if registration == "success" else 1)

    api = SimpleNamespace(
        cudaHostRegister=register,
        cudaHostUnregister=lambda pointer: None,
    )
    ns = {
        "weakref": weakref,
        "torch": SimpleNamespace(
            Tensor=torch.Tensor,
            uint8=torch.uint8,
            empty=empty,
            cuda=SimpleNamespace(
                is_available=lambda: registration != "unavailable",
                cudart=lambda: api,
            ),
        ),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), ns)
    with torch.device(default_device):
        pinned_buffer = cast(Callable[[int], torch.Tensor], ns["_pinned_uint8_buffer"])
        out = pinned_buffer(257)
    assert out.device.type == "cpu"
    assert out.dtype == torch.uint8
    assert out.numel() == 257
    if registration == "success":
        assert registrations == [(out.data_ptr(), 257, 0)]
        assert len(allocations) == 1
    else:
        assert allocations[-1]["pin_memory"] is True
