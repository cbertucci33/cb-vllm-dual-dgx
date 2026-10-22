# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM MoE flattens arbitrary leading dims while preserving SP semantics."""

import pytest
import torch
from torch import nn

from vllm.models.glm5next.common import model as glm_model


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, hidden_states, router_logits):
        self.input_shape = hidden_states.shape
        assert router_logits is hidden_states
        return hidden_states + 1


@pytest.mark.parametrize("shape", [(8, 4), (1, 8, 4), (2, 4, 4)])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_moe_forward_preserves_input_shape_and_sequence_parallel(
    monkeypatch, shape, sequence_parallel
):
    module = object.__new__(glm_model.Glm5NextMoE)
    nn.Module.__init__(module)
    module.is_sequence_parallel = sequence_parallel
    module.experts = _Experts()
    calls = []

    def chunk(x):
        calls.append(("chunk", x.shape))
        return x[1::2]

    def all_gather(x, dim):
        calls.append(("gather", x.shape, dim))
        return torch.cat((x, x), dim=dim)

    monkeypatch.setattr(glm_model, "sequence_parallel_chunk", chunk)
    monkeypatch.setattr(glm_model, "tensor_model_parallel_all_gather", all_gather)
    hidden = torch.randn(*shape)
    output = module(hidden)

    assert output.shape == hidden.shape
    num_tokens = hidden.numel() // shape[-1]
    expected_tokens = (num_tokens + 1) // 2 if sequence_parallel else num_tokens
    assert module.experts.input_shape == (expected_tokens, shape[-1])
    assert [call[0] for call in calls] == (
        ["chunk", "gather"] if sequence_parallel else []
    )


def test_moe_forward_keeps_already_sequence_parallel_input_flat(monkeypatch):
    module = object.__new__(glm_model.Glm5NextMoE)
    nn.Module.__init__(module)
    module.is_sequence_parallel = True
    module.experts = _Experts()
    hidden = torch.randn(1, 5, 4)

    output = module(hidden, already_sequence_parallel=True)

    assert module.experts.input_shape == (5, 4)
    assert output.shape == hidden.shape
