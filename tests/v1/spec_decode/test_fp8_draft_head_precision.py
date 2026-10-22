# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU scale/dispatch regression; GPU numerical candidate coverage is separate."""

import importlib.util
from pathlib import Path

import torch


def test_fp32_scales_and_gemm_output_contract(monkeypatch):
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm/model_executor/layers/fp8_draft_head.py"
    )
    spec = importlib.util.spec_from_file_location("head_precision", path)
    assert spec is not None and spec.loader is not None
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    weight = torch.tensor([[1.125, -0.75], [0.375, 1.75]], dtype=torch.bfloat16)
    hidden = torch.tensor([[0.625, -1.125]], dtype=torch.bfloat16)
    head = module.quantize_draft_head(weight)
    expected_scale = weight.float().abs().amax(1).reshape(1, -1) / 448.0
    torch.testing.assert_close(head.row_scale, expected_scale, rtol=0, atol=0)
    assert head.row_scale.dtype == head.unit_scale.dtype == torch.float32
    calls = []

    def scaled_mm(a, b, *, scale_a, scale_b, out_dtype):
        calls.append(out_dtype)
        assert a.dtype == b.dtype == torch.float8_e4m3fn
        assert scale_a.dtype == scale_b.dtype == torch.float32
        assert out_dtype == torch.float32
        return a.float() @ b.float()

    monkeypatch.setattr(torch, "_scaled_mm", scaled_mm)
    actual = module.fp8_draft_head_logits(hidden, head)
    h = hidden.float()
    act_max = h.abs().amax(-1, keepdim=True).clamp(min=1e-6)
    quantized_h = (h * (448.0 / act_max)).to(torch.float8_e4m3fn).float()
    expected = (quantized_h @ head.weight_fp8.float().t()) * expected_scale
    expected = expected * (act_max / 448.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == [torch.float32]
    assert actual.dtype == torch.float32
