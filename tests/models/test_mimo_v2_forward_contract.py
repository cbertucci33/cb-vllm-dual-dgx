# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for MiMo's forward shape boundary, without model weights."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models import mimo_v2


@pytest.mark.parametrize("shape", [(4,), (1, 4), (3, 4)])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_moe_preserves_supported_input_shape(monkeypatch, shape, sequence_parallel):
    seen = []

    def gate(hidden):
        seen.append(hidden.shape)
        return hidden.float(), None

    if sequence_parallel:
        monkeypatch.setattr(mimo_v2, "sequence_parallel_chunk", lambda x: x)
        monkeypatch.setattr(
            mimo_v2, "tensor_model_parallel_all_gather", lambda x, dim: x
        )
    model = SimpleNamespace(
        is_sequence_parallel=sequence_parallel,
        gate_dtype=torch.bfloat16,
        gate=gate,
        experts=lambda hidden_states, router_logits: hidden_states * 3,
    )
    hidden = torch.arange(torch.tensor(shape).prod().item()).reshape(shape).float()
    result = mimo_v2.MiMoV2MoE.forward(model, hidden)
    assert result.shape == hidden.shape
    torch.testing.assert_close(result, hidden * 3)
    assert seen == [torch.Size((1 if len(shape) == 1 else shape[0], 4))]
