# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU context-copy lifecycle checks, not native GEMM/graph qualification."""

from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from vllm.model_executor.model_loader.derived_weights import refresh_derived_weights
from vllm.model_executor.models.dflash_policy import DFlashExecutionPolicy
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model


def make_model(bias=False):
    model = DFlashQwen3Model.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.config = NS(hidden_size=64)
    model.hidden_norm = nn.LayerNorm(64, dtype=torch.bfloat16)
    model.dflash_execution_policy = DFlashExecutionPolicy(True, True)
    model.layers = nn.ModuleList()
    for i in range(2):
        layer = nn.Module()
        a = layer.self_attn = nn.Module()
        a.q_size, a.kv_size, a.head_dim, a.num_kv_heads = 8, 2, 2, 1
        a.qkv_proj = nn.Module()
        a.qkv_proj.register_parameter(
            "weight",
            nn.Parameter(
                (torch.arange(12 * 64).reshape(12, 64) % 7 - 3).to(torch.float8_e4m3fn),
                requires_grad=False,
            ),
        )
        a.qkv_proj.register_parameter(
            "weight_scale",
            nn.Parameter(
                torch.full((12, 2), 126 + i, dtype=torch.uint8), requires_grad=False
            ),
        )
        a.qkv_proj.register_parameter(
            "bias",
            nn.Parameter(torch.arange(12).to(torch.bfloat16), requires_grad=False)
            if bias
            else None,
        )
        a.k_norm = nn.LayerNorm(2, dtype=torch.bfloat16)
        a.q_norm = NS(variance_epsilon=1e-5)
        a.rotary_emb = nn.Module()
        a.rotary_emb.head_size = 2
        a.rotary_emb.is_neox_style = True
        a.rotary_emb.register_buffer("cos_sin_cache", torch.ones(4, 2))
        a.attn = nn.Module()
        model.layers.append(layer)
    model.requires_grad_(False)
    return model


def pack_like_b12x(proj):
    # The actual pinned packer retains padded row-major values + [1,N,K/32]
    # compact scales. CPU fixture covers their consumer, not CUDA packing.
    weight = proj.weight.detach()
    scale = proj.weight_scale.detach()
    n, k = weight.shape
    padded_k = (k + 127) // 128 * 128
    values = weight.new_zeros((n, padded_k))
    values[:, :k] = weight
    rows = torch.full((1, n, padded_k // 32), 127, dtype=torch.uint8)
    rows[:, :, : k // 32] = scale
    proj.b12x_mxfp8_packed_weight = NS(
        weight=NS(values=values, scale_rows=rows.view(torch.float8_e8m0fnu)),
        in_features=k,
        padded_in_features=padded_k,
        out_features=n,
    )
    proj.weight = nn.Parameter(weight.new_empty(0), requires_grad=False)
    proj.weight_scale = nn.Parameter(scale.new_empty(0), requires_grad=False)


@pytest.mark.parametrize("bias", [False, True])
def test_rebuild_preserves_graph_buffers_and_new_values(bias):
    model = make_model(bias)
    with torch.inference_mode():
        model._build_fused_kv_buffers()
    names = ["_fused_kv_weight", "_k_norm_weights"]
    if bias:
        names.append("_fused_kv_bias")
    saved = {n: (getattr(model, n), getattr(model, n).clone()) for n in names}
    for layer in model.layers:
        a = layer.self_attn
        a.qkv_proj.weight_scale.add_(1)
        a.k_norm.weight.mul_(2)
        if bias:
            a.qkv_proj.bias.mul_(2)
    model._build_fused_kv_buffers()
    for name, (old, value) in saved.items():
        assert getattr(model, name) is old
        torch.testing.assert_close(old, value * 2, rtol=0, atol=0)


@pytest.mark.parametrize("bias", [False, True])
def test_packed_weights_refresh_after_reload(bias):
    model = make_model(bias)
    model._build_fused_kv_buffers()
    captured = model._fused_kv_weight
    original = captured.clone()
    for layer in model.layers:
        proj = layer.self_attn.qkv_proj
        pack_like_b12x(proj)
        rows = proj.b12x_mxfp8_packed_weight.weight.scale_rows.view(torch.uint8)
        rows.add_(1)
        layer.self_attn.k_norm.weight.fill_(3)
    refresh_derived_weights(model)
    assert model._fused_kv_weight is captured
    torch.testing.assert_close(captured, original * 2, rtol=0, atol=0)
    assert torch.equal(model._k_norm_weights, torch.full((2, 2), 3.0).bfloat16())


def test_cold_build_from_processed_weights():
    model = make_model()
    expected = torch.cat(
        [model._glm_context_kv_rows(x.self_attn) for x in model.layers]
    )
    for layer in model.layers:
        pack_like_b12x(layer.self_attn.qkv_proj)
    model._build_fused_kv_buffers()
    torch.testing.assert_close(model._fused_kv_weight, expected, rtol=0, atol=0)


def test_context_buffers_are_nonpersistent_module_owned():
    model = make_model(True)
    model._build_fused_kv_buffers()
    names = {"_fused_kv_weight", "_fused_kv_bias", "_k_norm_weights"}
    assert names <= model._non_persistent_buffers_set
    assert not names & model.state_dict().keys()
    model.to("meta")
    assert all(getattr(model, n).is_meta for n in names)


def test_hidden_norm_reads_current_parameter(monkeypatch):
    from vllm.model_executor.models import qwen3_dflash

    model = make_model()
    model._build_fused_kv_buffers()
    model.hidden_norm.weight = nn.Parameter(torch.full((64,), 3.0).bfloat16())
    seen = []

    def norm(output, source, weight, eps):
        seen.append(weight)
        output.copy_(source)

    monkeypatch.setattr(qwen3_dflash.ops, "rms_norm", norm)
    model._project_context_kv(torch.ones(1, 64).bfloat16(), 1, 2, 1, 2)
    assert len(seen) == 1 and seen[0] is model.hidden_norm.weight


def test_non_glm_refresh_does_not_rebuild(monkeypatch):
    model = make_model()
    model.dflash_execution_policy = DFlashExecutionPolicy()
    model._num_attn_layers = 2

    def forbidden():
        raise AssertionError("GLM lifecycle must not select a different draft policy")

    monkeypatch.setattr(model, "_build_fused_kv_buffers", forbidden)
    refresh_derived_weights(model)


@pytest.mark.parametrize("bias", [False, True])
def test_real_layerwise_reload_preserves_context_copies(bias):
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    class PackMethod(QuantizeMethodBase):
        def create_weights(self, layer, *args, **kwargs):
            pass

        def apply(self, layer, *args, **kwargs):
            raise AssertionError("No native GEMM in CPU reload check")

        def process_weights_after_loading(self, layer):
            pack_like_b12x(layer)

    model = make_model(bias)
    for layer in model.layers:
        layer.self_attn.qkv_proj.quant_method = PackMethod()
    checkpoint = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    record_metadata_for_reloading(model)
    for layer in model.layers:
        pack_like_b12x(layer.self_attn.qkv_proj)
    with torch.inference_mode():
        model._build_fused_kv_buffers()
    captured = model._fused_kv_weight
    expected = captured.clone()
    norm = model._k_norm_weights
    for cycle in (1, 2):
        initialize_layerwise_reload(model)
        for name, source in checkpoint.items():
            target = model.get_parameter(name)
            if name.endswith("weight_scale"):
                value = source + cycle
            elif ".k_norm.weight" in name:
                value = source * (cycle + 1)
            else:
                value = source
            getattr(target, "weight_loader", default_weight_loader)(target, value)
        # The real DFlash model's load_weights ends by rebuilding before the
        # generic finalizer restores parent-module buffer registrations.
        model._build_fused_kv_buffers()
        finalize_layerwise_reload(model, NS(dtype=torch.bfloat16))
        assert model._fused_kv_weight is captured
        assert model._k_norm_weights is norm
        torch.testing.assert_close(captured, expected * (2**cycle), rtol=0, atol=0)
        assert torch.equal(norm, torch.full_like(norm, cycle + 1))


def test_derived_hook_on_other_inherited_draft_has_no_policy():
    model = make_model()
    del model.dflash_execution_policy
    refresh_derived_weights(model)
