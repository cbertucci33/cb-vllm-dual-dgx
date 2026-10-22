# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 weight and scale pairs must survive split GLM weight-loader calls."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.glm5next.common.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
    _try_load_fp8_attn_proj,
    _try_load_fp8_indexer_wk,
)

pytestmark = pytest.mark.cpu_test

HIDDEN = 256
BLOCK = 128
Q_LORA_RANK = 256
KV_LORA_RANK = 128
ROPE_DIM = 64
LAYER = 0
ATTN = f"layers.{LAYER}.self_attn"
FUSED = f"{ATTN}.fused_qkv_a_proj.weight"


def _weight(rows: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, HIDDEN, generator=generator) * 0.1).to(
        torch.float8_e4m3fn
    )


def _scale(rows: int, value: float) -> torch.Tensor:
    return torch.full(((rows + BLOCK - 1) // BLOCK, 2), value)


def _make_model() -> tuple[Glm5NextModel, list[int]]:
    model = object.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        is_moe=False,
        is_linear_attn=False,
        n_routed_experts=None,
        layer_types=None,
        mla_use_nope=True,
        qk_rope_head_dim=ROPE_DIM,
        num_hidden_layers=1,
        num_nextn_predict_layers=0,
    )
    model.layers = nn.ModuleList([nn.Module()])
    attention = nn.Module()
    fused = nn.Module()
    fused.weight = nn.Parameter(
        torch.zeros(
            Q_LORA_RANK + KV_LORA_RANK + ROPE_DIM,
            HIDDEN,
            dtype=torch.bfloat16,
        ),
        requires_grad=False,
    )
    shards: list[int] = []

    def weight_loader(param, loaded, shard_id):
        shards.append(shard_id)
        start = 0 if shard_id == 0 else Q_LORA_RANK
        param.data[start : start + loaded.shape[0]].copy_(loaded)

    fused.weight.weight_loader = weight_loader
    attention.fused_qkv_a_proj = fused
    model.layers[0].self_attn = attention
    return model, shards


def _weights() -> dict[str, torch.Tensor]:
    return {
        f"{ATTN}.q_a_proj.weight": _weight(Q_LORA_RANK, 1),
        f"{ATTN}.q_a_proj.weight_scale_inv": _scale(Q_LORA_RANK, 0.5),
        f"{ATTN}.kv_a_proj_with_mqa.weight": _weight(KV_LORA_RANK, 2),
        f"{ATTN}.kv_a_proj_with_mqa.weight_scale_inv": _scale(KV_LORA_RANK, 0.25),
    }


def _assert_pair_loaded(model: Glm5NextModel, shards: list[int]) -> None:
    param = dict(model.named_parameters())[FUSED]
    assert sorted(shards) == [0, 1]
    assert torch.count_nonzero(param[:Q_LORA_RANK])
    assert torch.count_nonzero(param[Q_LORA_RANK : Q_LORA_RANK + KV_LORA_RANK])
    assert torch.count_nonzero(param[Q_LORA_RANK + KV_LORA_RANK :]) == 0


def test_fp8_weight_scale_pairs_survive_separate_load_calls() -> None:
    model, shards = _make_model()
    weights = _weights()
    weight_items = [(k, v) for k, v in weights.items() if k.endswith(".weight")]
    scale_items = [(k, v) for k, v in weights.items() if "weight_scale_inv" in k]

    first = model.load_weights(iter(weight_items))
    second = model.load_weights(iter(scale_items))

    assert FUSED not in first
    assert FUSED in second
    _assert_pair_loaded(model, shards)


def test_multimodal_autoweightsloader_preserves_split_fp8_pairs() -> None:
    text_model, shards = _make_model()
    language_model = object.__new__(Glm5NextForCausalLM)
    nn.Module.__init__(language_model)
    language_model.config = SimpleNamespace(tie_word_embeddings=False)
    language_model.model = text_model

    model = object.__new__(Glm5NextForConditionalGeneration)
    nn.Module.__init__(model)
    model.language_model = language_model
    model.visual = nn.Module()
    model.visual.patch_embed = nn.Module()
    model.visual.patch_embed.proj = nn.Module()
    model.visual.patch_embed.proj.weight = nn.Parameter(
        torch.zeros(1), requires_grad=False
    )

    weights = _weights()
    stream = [
        (
            f"model.language_model.{ATTN}.q_a_proj.weight",
            weights[f"{ATTN}.q_a_proj.weight"],
        ),
        (
            "model.visual.patch_embed.proj.weight",
            torch.tensor([1.0]),
        ),
        (
            f"model.language_model.{ATTN}.q_a_proj.weight_scale_inv",
            weights[f"{ATTN}.q_a_proj.weight_scale_inv"],
        ),
        (
            f"model.language_model.{ATTN}.kv_a_proj_with_mqa.weight",
            weights[f"{ATTN}.kv_a_proj_with_mqa.weight"],
        ),
        (
            f"model.language_model.{ATTN}.kv_a_proj_with_mqa.weight_scale_inv",
            weights[f"{ATTN}.kv_a_proj_with_mqa.weight_scale_inv"],
        ),
    ]

    loaded = model.load_weights(iter(stream))

    assert "visual.patch_embed.proj.weight" in loaded
    assert torch.equal(model.visual.patch_embed.proj.weight, torch.tensor([1.0]))
    assert f"language_model.model.{FUSED}" in loaded
    _assert_pair_loaded(text_model, shards)


def test_fp8_helpers_do_not_buffer_absent_targets() -> None:
    # On a PP rank without these parameters, the fast paths must defer to the
    # regular loader's PP-missing guard rather than retaining unused tensors.
    weights = _weights()
    buffer: dict[str, torch.Tensor] = {}
    loaded_params: set[str] = set()
    params_dict: dict[str, torch.Tensor] = {}

    for name, tensor in weights.items():
        assert not _try_load_fp8_attn_proj(
            name, tensor, buffer, params_dict, loaded_params, ROPE_DIM
        )
    assert buffer == {}

    indexer_weights = {
        "layers.0.self_attn.indexer.wk.weight": _weight(128, 3),
        "layers.0.self_attn.indexer.wk.weight_scale_inv": _scale(128, 0.5),
    }
    for name, tensor in indexer_weights.items():
        assert not _try_load_fp8_indexer_wk(
            name, tensor, buffer, params_dict, loaded_params
        )
    assert buffer == {}
