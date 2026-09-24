# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.models.glm5next.nvidia.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    set_eagle3_aux_hidden_state_layers,
)


def test_dflash_configures_glm5_multimodal_target_aux_layers():
    inner = object.__new__(Glm5NextModel)
    torch.nn.Module.__init__(inner)

    language_model = object.__new__(Glm5NextForCausalLM)
    torch.nn.Module.__init__(language_model)
    language_model.model = inner

    target = object.__new__(Glm5NextForConditionalGeneration)
    torch.nn.Module.__init__(target)
    target.language_model = language_model

    spec_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                dflash_config={"target_layer_ids": [5, 14, 24, 33, 42]}
            )
        )
    )

    assert supports_eagle3(target)
    set_eagle3_aux_hidden_state_layers(target, spec_config)
    assert inner.aux_hidden_state_layers == (6, 15, 25, 34, 43)


def test_glm5_aux_capture_requires_explicit_dflash_layers():
    target = object.__new__(Glm5NextForCausalLM)
    torch.nn.Module.__init__(target)

    try:
        target.get_eagle3_default_aux_hidden_state_layers()
    except NotImplementedError as exc:
        assert "explicit target_layer_ids" in str(exc)
    else:
        raise AssertionError("GLM5 must not invent generic EAGLE3 capture layers")


def test_glm5_forward_returns_requested_completed_layer_outputs():
    class AddOneLayer(torch.nn.Module):
        def __init__(self, layer_idx: int):
            super().__init__()
            self.layer_idx = layer_idx

        def forward(self, positions, hidden_states, residual, post, comb):
            return hidden_states + 1, residual, post, comb

    model = object.__new__(Glm5NextModel)
    torch.nn.Module.__init__(model)
    model.is_sequence_parallel = False
    model._active_layers = torch.nn.ModuleList(
        [AddOneLayer(0), AddOneLayer(1), AddOneLayer(2)]
    )
    model.norm = torch.nn.Identity()
    model.aux_hidden_state_layers = (1, 3)

    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    inputs = torch.zeros(2, 4)
    positions = torch.arange(2)
    with patch(
        "vllm.models.glm5next.nvidia.model.get_pp_group", return_value=pp_group
    ):
        output, aux_hidden_states = model(
            input_ids=None,
            positions=positions,
            intermediate_tensors=None,
            inputs_embeds=inputs,
        )

    torch.testing.assert_close(output, inputs + 3)
    assert len(aux_hidden_states) == 2
    torch.testing.assert_close(aux_hidden_states[0], inputs + 1)
    torch.testing.assert_close(aux_hidden_states[1], inputs + 3)
