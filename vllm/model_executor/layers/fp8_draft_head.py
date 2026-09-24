# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rowwise FP8 quantization for a speculative-decoding draft LM head."""

from typing import NamedTuple

import torch

_FP8_MAX = 448.0


class Fp8DraftHead(NamedTuple):
    """Rowwise-quantized copy of a vocab-sharded LM head."""

    weight_fp8: torch.Tensor
    row_scale: torch.Tensor
    unit_scale: torch.Tensor


def fp8_draft_head_supported(device: torch.device | None = None) -> bool:
    """Return whether ``torch._scaled_mm`` has FP8 tensor-core support."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return (major, minor) >= (8, 9)


def quantize_draft_head(weight: torch.Tensor) -> Fp8DraftHead:
    """Quantize a ``[local_vocab, hidden]`` head rowwise to FP8 E4M3."""
    with torch.no_grad():
        w = weight.detach()
        row_max = w.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-6)
        weight_fp8 = (w.float() * (_FP8_MAX / row_max)).to(torch.float8_e4m3fn)
        row_scale = (row_max / _FP8_MAX).to(w.dtype).reshape(1, -1)
        unit_scale = torch.ones(1, dtype=torch.float32, device=w.device)
    return Fp8DraftHead(weight_fp8, row_scale, unit_scale)


def fp8_draft_head_logits(
    hidden_states: torch.Tensor,
    head: Fp8DraftHead,
) -> torch.Tensor:
    """Compute local draft logits with dynamic-activation rowwise FP8 GEMM."""
    act_max = hidden_states.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    act_fp8 = (hidden_states * (_FP8_MAX / act_max)).to(torch.float8_e4m3fn)
    logits = torch._scaled_mm(
        act_fp8,
        head.weight_fp8.t(),
        scale_a=head.unit_scale,
        scale_b=head.unit_scale,
        out_dtype=hidden_states.dtype,
    )
    logits = logits * head.row_scale
    logits = logits * (act_max / _FP8_MAX).to(hidden_states.dtype)
    return logits
