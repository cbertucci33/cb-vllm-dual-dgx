# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for derived KDA weights across supported load/reload entries."""

import ast
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from vllm.model_executor.model_loader import utils as loader_utils
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.sharded_state_loader import ShardedStateLoader
from vllm.model_executor.model_loader.weight_cache.ipc_loader import IpcModelLoader
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.glm5next.common import kda
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

ROOT = Path(__file__).resolve().parents[3]


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        layer = kda.Glm5NextLinearAttention.__new__(kda.Glm5NextLinearAttention)
        nn.Module.__init__(layer)
        self.layer = layer
        layer.prefix = "layer"
        layer.kda_safe_gate = True
        layer.kda_lower_bound = -5.0
        layer._conv_state_dim_first = True
        layer.kv_cache = (torch.zeros(2, 6, 2), torch.zeros(2, 1, 2, 2))
        # Same non-persistent ownership as the repaired constructor.
        layer.register_buffer("_merged_conv_weight", None, persistent=False)
        for i, name in enumerate(("q_conv1d", "k_conv1d", "v_conv1d")):
            conv = nn.Module()
            conv.weight = nn.Parameter(
                torch.arange(6, dtype=torch.float32).view(2, 1, 3) + i * 10,
                requires_grad=False,
            )
            conv.weight.weight_loader = default_weight_loader
            conv.bias = None
            setattr(layer, name, conv)
        record_metadata_for_reloading(self)

    def load_weights(self, weights):
        loaded = set()
        for name, value in weights:
            param = self.get_parameter(name)
            getattr(param, "weight_loader", default_weight_loader)(param, value)
            loaded.add(name)
        return loaded


def _observe_conv_weights(model, monkeypatch):
    """Execute the real forward until its native-convolution call."""
    fields = dict(
        num_actual_tokens=1,
        num_spec_decodes=0,
        num_prefills=1,
        num_decodes=0,
        has_initial_state=None,
        non_spec_query_start_loc=None,
        non_spec_state_indices_tensor=None,
        spec_sequence_masks=None,
        spec_query_start_loc=None,
        spec_state_indices_tensor=None,
        spec_token_indx=None,
        non_spec_token_indx=None,
        num_accepted_tokens=None,
    )
    meta = GDNAttentionMetadata.__new__(GDNAttentionMetadata)
    meta.__dict__.update(fields)
    monkeypatch.setattr(
        kda,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer": meta}),
    )
    observed = []

    class Observed(Exception):
        pass

    def conv(_input, weight, *_args, **_kwargs):
        observed.append(weight)
        raise Observed

    monkeypatch.setattr(kda, "causal_conv1d_fn", conv)
    with pytest.raises(Observed):
        model.layer._forward(
            torch.zeros(1, 6),
            torch.zeros(1, 1, 1, 2),
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 1, 2),
        )
    return observed[0]


def _runner_reload(model, weights, checkpoint):
    """Run the production runner method without constructing a GPU runner."""
    source = ROOT / "vllm/v1/worker/gpu_model_runner.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "reload_weights"
    )
    fn.decorator_list = []
    namespace: dict[str, Any] = {
        "time": time,
        "initialize_layerwise_reload": initialize_layerwise_reload,
        "finalize_layerwise_reload": finalize_layerwise_reload,
        "_get_parameter_for_reload": lambda mod, name: mod.get_parameter(name),
        "logger": SimpleNamespace(
            info_once=lambda *_: None,
            warning_once=lambda *_: None,
            warning=lambda *_: None,
        ),
    }
    # Absent on the defect revision; new caller uses this actual dispatcher.
    if hasattr(loader_utils, "refresh_derived_weights"):
        namespace["refresh_derived_weights"] = loader_utils.refresh_derived_weights
    import __future__

    exec(
        compile(
            ast.Module(body=[fn], type_ignores=[]),
            str(source),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    runner = SimpleNamespace(
        get_model=lambda: model,
        get_draft_model=lambda: None,
        lora_config=None,
        model_config=SimpleNamespace(quantization=None, dtype=torch.float32),
        reset_lora_state=lambda: None,
        reset_encoder_cache=lambda: None,
        reset_mm_cache=lambda: None,
    )
    namespace["reload_weights"](
        runner, weights_iterator=weights, is_checkpoint_format=checkpoint
    )


@pytest.mark.parametrize(
    "entry", ["checkpoint", "kernel", "sharded", "ipc", "postload", "default"]
)
@torch.inference_mode()
def test_reloaded_kda_convolution_is_current_and_address_stable(entry, monkeypatch):
    model = TinyModel()
    captured = _observe_conv_weights(model, monkeypatch)
    pointer = captured.data_ptr()
    original_names = set(model.state_dict())
    for delta in (100, 200):
        weights = [
            (name, p.detach().clone() + delta) for name, p in model.named_parameters()
        ]
        if entry in ("checkpoint", "kernel"):
            _runner_reload(model, iter(weights), entry == "checkpoint")
        elif entry == "default":
            loader = DefaultModelLoader.__new__(DefaultModelLoader)
            loader.counter_before_loading_weights = time.perf_counter()
            loader.enable_weights_track = True
            loader._init_ep_weight_filter = lambda _: None
            loader._init_mm_encoder_only_weight_filter = lambda *_: None
            loader.get_all_weights = lambda *_, values=weights: iter(values)
            loader.load_weights(model, SimpleNamespace(quantization=None))
        elif entry == "sharded":
            import vllm.distributed
            from vllm.model_executor.model_loader import sharded_state_loader

            monkeypatch.setattr(
                vllm.distributed, "get_tensor_model_parallel_rank", lambda: 0
            )
            monkeypatch.setattr(sharded_state_loader.glob, "glob", lambda _: ["part"])
            loader = ShardedStateLoader.__new__(ShardedStateLoader)
            loader.pattern = "rank-{rank}-part-{part}"
            loader.iterate_over_files = lambda _, values=weights: iter(values)
            loader.load_weights(
                model, SimpleNamespace(model="local", model_weights=None)
            )
        elif entry == "ipc":
            monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
            entries = {
                name: SimpleNamespace(rebuild=lambda _idx, x=value: x)
                for name, value in weights
            }
            loader = IpcModelLoader.__new__(IpcModelLoader)
            loader._fetch_entries = lambda _, values=entries: SimpleNamespace(
                entries=values
            )
            loader.load_weights(model, SimpleNamespace())
        else:
            model.load_weights(weights)
            loader_utils.process_weights_after_loading(
                model,
                SimpleNamespace(
                    dtype=torch.float32,
                    quantization=None,
                    word_embeddings_untied_by_checkpoint=False,
                ),
                torch.device("cpu"),
            )
        expected = torch.cat([value[:, 0] for _, value in weights])
        actual = _observe_conv_weights(model, monkeypatch)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual is captured
        assert actual.data_ptr() == pointer
        assert set(model.state_dict()) == original_names
        assert not torch.cuda.is_initialized()


@torch.inference_mode()
def test_ordinary_modules_are_unchanged_by_refresh():
    refresh = getattr(loader_utils, "refresh_derived_weights", None)
    assert refresh is not None
    model = nn.Linear(3, 2)
    identities = {name: p for name, p in model.named_parameters()}
    before = {name: p.clone() for name, p in model.named_parameters()}
    refresh(model)
    for name, p in model.named_parameters():
        assert identities[name] is p
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)


def test_refresh_inference_buffer_outside_inference_mode(monkeypatch):
    model = TinyModel()
    with torch.inference_mode():
        captured = _observe_conv_weights(model, monkeypatch)
    assert captured.is_inference()
    for param in model.parameters():
        param.add_(17)
    expected = torch.cat(
        [
            conv.weight[:, 0]
            for conv in (
                model.layer.q_conv1d,
                model.layer.k_conv1d,
                model.layer.v_conv1d,
            )
        ]
    )
    loader_utils.refresh_derived_weights(model)
    assert model.layer._merged_conv_weight is captured
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["copy", "zero_copy"])
def test_ipc_install_preserves_declared_buffer_persistence(mode):
    model = TinyModel()
    weights = torch.arange(18, dtype=torch.float32).view(6, 3)
    loader = IpcModelLoader.__new__(IpcModelLoader)
    loader.mode = mode
    state = SimpleNamespace(
        entries={
            "layer._merged_conv_weight": SimpleNamespace(
                kind="buffer", rebuild=lambda _: weights
            )
        },
        aliases={},
    )
    loader._apply_entries(model, state, 0)
    assert "_merged_conv_weight" in model.layer._non_persistent_buffers_set
    assert "layer._merged_conv_weight" not in model.state_dict()
    torch.testing.assert_close(model.layer._merged_conv_weight, weights)
    if mode == "copy":
        assert model.layer._merged_conv_weight.data_ptr() != weights.data_ptr()
    else:
        assert model.layer._merged_conv_weight is weights
