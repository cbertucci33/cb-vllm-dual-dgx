# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for GLM indexer head weights; native sparse scoring is separate."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.models.glm5next.test_kda_weight_refresh import _runner_reload
from vllm.model_executor.model_loader.derived_weights import refresh_derived_weights
from vllm.model_executor.model_loader.reload import record_metadata_for_reloading
from vllm.model_executor.model_loader.weight_cache.ipc_loader import IpcModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.glm5next.common import attention


class Projection(nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.weight = nn.Parameter(
            torch.arange(rows * 4, dtype=torch.bfloat16).reshape(rows, 4) / 128,
            requires_grad=False,
        )
        self.weight.weight_loader = default_weight_loader

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight), None


class TinyIndexer(nn.Module):
    def __init__(self):
        super().__init__()
        layer = attention.Indexer.__new__(attention.Indexer)
        nn.Module.__init__(layer)
        self.layer = layer
        layer.head_dim = 128
        layer.n_head = 2
        layer.rope_dim = 0
        layer.quant_block_size = 128
        layer.scale_fmt = "ue8m0"
        layer.softmax_scale = 128**-0.5
        layer.wq_b = Projection(256)
        layer.wk_weights_proj = Projection(130)
        layer.k_norm = nn.Identity()
        layer.register_buffer("_wp_fp32", None, persistent=False)
        record_metadata_for_reloading(self)

    def load_weights(self, weights):
        names = set()
        for name, value in weights:
            param = self.get_parameter(name)
            getattr(param, "weight_loader", default_weight_loader)(param, value)
            names.add(name)
        return names


def observe_head_weights(model, monkeypatch):
    """Run the real forward through the FP32 gate GEMM; stop before native scoring."""
    observed = []

    class Observed(Exception):
        pass

    def scale(weights, *_):
        observed.append(weights)
        raise Observed

    monkeypatch.setattr(attention, "_fused_indexer_weight_scale", scale)
    monkeypatch.setattr(
        attention,
        "fwht128_quant_fp8",
        lambda x: (torch.zeros_like(x), torch.ones(x.shape[0], 1)),
    )
    hidden = torch.tensor([[1, 2, 3, 4]], dtype=torch.bfloat16)
    with pytest.raises(Observed):
        model.layer(hidden, hidden, None, None)
    return observed[0], hidden


@pytest.mark.parametrize("entry", ["checkpoint", "kernel", "ipc_copy", "ipc_zero_copy"])
@torch.inference_mode()
def test_indexer_reload_keeps_current_fp32_gate_and_graph_storage(entry, monkeypatch):
    model = TinyIndexer()
    observe_head_weights(model, monkeypatch)
    captured = model.layer._wp_fp32
    pointer = captured.data_ptr()
    names = set(model.state_dict())
    for delta in (2, 5):
        weights = [
            (name, value.detach().clone() + delta)
            for name, value in model.named_parameters()
        ]
        if entry.startswith("ipc_"):
            loader = IpcModelLoader.__new__(IpcModelLoader)
            loader.mode = entry.removeprefix("ipc_")
            state = SimpleNamespace(
                entries={
                    name: SimpleNamespace(kind="param", rebuild=lambda _, x=x: x)
                    for name, x in weights
                },
                aliases={},
            )
            loader._apply_entries(model, state, 0)
            refresh_derived_weights(model)
        else:
            _runner_reload(model, iter(weights), entry == "checkpoint")
        actual, hidden = observe_head_weights(model, monkeypatch)
        current = model.layer.wk_weights_proj.weight[128:].T.float()
        expected = hidden.float() @ current
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert model.layer._wp_fp32 is captured
        assert captured.data_ptr() == pointer
        assert captured.dtype == torch.float32
        assert captured.is_contiguous()
        assert set(model.state_dict()) == names
        assert not torch.cuda.is_initialized()


def test_refresh_inference_indexer_storage_outside_inference_mode(monkeypatch):
    model = TinyIndexer()
    with torch.inference_mode():
        observe_head_weights(model, monkeypatch)
    captured = model.layer._wp_fp32
    assert captured.is_inference()
    model.layer.wk_weights_proj.weight.add_(7)
    refresh_derived_weights(model)
    expected = model.layer.wk_weights_proj.weight[128:].T.float()
    assert model.layer._wp_fp32 is captured
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)


def test_cold_postload_owns_derived_fp32_gate_without_checkpoint_entry():
    model = TinyIndexer()
    refresh_derived_weights(model)
    value = model.layer._wp_fp32
    assert value is not None
    assert value.dtype == torch.float32
    assert value.is_contiguous()
    assert dict(model.named_buffers())["layer._wp_fp32"] is value
    assert "layer._wp_fp32" not in model.state_dict()
    torch.testing.assert_close(
        value, model.layer.wk_weights_proj.weight[128:].T.float(), rtol=0, atol=0
    )
