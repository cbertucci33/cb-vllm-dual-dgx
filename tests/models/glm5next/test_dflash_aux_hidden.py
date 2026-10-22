# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.models.glm5next.common.model import (
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


def test_glm5_multimodal_wrapper_preserves_aux_hidden_tuple():
    final_hidden = torch.randn(2, 4)
    aux_hidden = [torch.randn(2, 4), torch.randn(2, 4)]
    seen = {}

    class Inner(torch.nn.Module):
        def forward(self, **kwargs):
            seen.update(kwargs)
            return final_hidden, aux_hidden

    target = object.__new__(Glm5NextForConditionalGeneration)
    torch.nn.Module.__init__(target)
    target.language_model = SimpleNamespace(model=Inner())
    input_ids = torch.tensor([1, 2])
    positions = torch.arange(2)

    output = target(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=None,
        inputs_embeds=None,
    )

    assert output[0] is final_hidden
    assert output[1] is aux_hidden
    assert seen == {
        "input_ids": input_ids,
        "positions": positions,
        "intermediate_tensors": None,
        "inputs_embeds": None,
    }


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
    model.start_layer = 0
    model.end_layer = 3
    model.is_sequence_parallel = False
    model._active_layers = torch.nn.ModuleList(
        [AddOneLayer(0), AddOneLayer(1), AddOneLayer(2)]
    )
    model.norm = torch.nn.Identity()
    model.aux_hidden_state_layers = (1, 3)

    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    inputs = torch.zeros(2, 4)
    positions = torch.arange(2)
    with patch("vllm.models.glm5next.common.model.get_pp_group", return_value=pp_group):
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


def test_glm5_completed_mhc_capture_does_not_mutate_deferred_state():
    model = object.__new__(Glm5NextModel)
    torch.nn.Module.__init__(model)

    hidden_states = torch.tensor([[1.0, 2.0]])
    residual = torch.tensor([[3.0, 4.0]])
    post = torch.tensor([[5.0, 6.0]])
    comb = torch.tensor([[7.0, 8.0]])
    snapshots = tuple(t.clone() for t in (hidden_states, residual, post, comb))

    class Layer:
        n = 2

        @staticmethod
        def hc_post(hidden, layer_residual, layer_post, layer_comb):
            return hidden + layer_residual + layer_post + layer_comb

    with patch(
        "vllm.models.glm5next.common.model.hc_contract",
        side_effect=lambda value, _n: value,
    ):
        model.config = SimpleNamespace(hc_mult=2)
        model._aux_post_op = Layer.hc_post
        captured = model._aux_hidden_state(hidden_states, residual, post, comb)

    torch.testing.assert_close(captured, torch.tensor([[16.0, 20.0]]))
    for actual, expected in zip(
        (hidden_states, residual, post, comb), snapshots, strict=True
    ):
        torch.testing.assert_close(actual, expected)


def test_glm5_sequence_parallel_preserves_completed_aux_and_final_outputs():
    class AddOneLayer(torch.nn.Module):
        def __init__(self, layer_idx: int):
            super().__init__()
            self.layer_idx = layer_idx

        def forward(self, positions, hidden_states, residual, post, comb):
            return hidden_states + 1, residual, post, comb

    model = object.__new__(Glm5NextModel)
    torch.nn.Module.__init__(model)
    model.start_layer = 0
    model.end_layer = 3
    model.is_sequence_parallel = True
    model._active_layers = torch.nn.ModuleList(
        [AddOneLayer(0), AddOneLayer(1), AddOneLayer(2)]
    )
    model.norm = torch.nn.Identity()
    model.aux_hidden_state_layers = (1, 3)

    gathered: list[torch.Tensor] = []

    def gather(value: torch.Tensor) -> torch.Tensor:
        gathered.append(value.clone())
        return torch.cat((value, torch.full_like(value[:1], -100)), dim=0)

    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    inputs = torch.zeros(2, 4)
    positions = torch.arange(2)
    with (
        patch("vllm.models.glm5next.common.model.get_pp_group", return_value=pp_group),
        patch("vllm.models.glm5next.common.model.sp_shard", side_effect=lambda x: x),
        patch("vllm.models.glm5next.common.model.sp_all_gather", side_effect=gather),
    ):
        output, aux_hidden_states = model(
            input_ids=None,
            positions=positions,
            intermediate_tensors=None,
            inputs_embeds=inputs,
        )

    assert len(gathered) == 3
    torch.testing.assert_close(gathered[0], inputs + 1)
    torch.testing.assert_close(gathered[1], inputs + 3)
    torch.testing.assert_close(gathered[2], inputs + 3)
    torch.testing.assert_close(output, inputs + 3)
    torch.testing.assert_close(aux_hidden_states[0], inputs + 1)
    torch.testing.assert_close(aux_hidden_states[1], inputs + 3)


def test_dflash_glm_projection_accepts_exact_20480_width_only():
    class Projection(torch.nn.Module):
        input_size = 20480

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value[:, :4096]

    owner = SimpleNamespace(
        model=SimpleNamespace(use_aux_hidden_state=True, fc=Projection())
    )
    hidden_states = torch.zeros(2, 20480)
    projected = DFlashQwen3ForCausalLM.combine_hidden_states(owner, hidden_states)
    assert projected.shape == (2, 4096)

    try:
        DFlashQwen3ForCausalLM.combine_hidden_states(owner, hidden_states[:, :-1])
    except ValueError as exc:
        assert "expects 20480" in str(exc)
        assert "received 20479" in str(exc)
    else:
        raise AssertionError("DFlash must reject an incomplete GLM aux-state tuple")
