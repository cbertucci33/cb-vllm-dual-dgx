# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU lifecycle contracts; native FP8 GEMM and CUDA graphs remain separate."""

import __future__

import ast
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.model_loader.derived_weights import refresh_derived_weights
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import qwen3_dflash2 as draft_module

ROOT = Path(__file__).resolve().parents[3]


class Target(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.lm_head.weight.requires_grad_(False)
        self.lm_head.weight.weight_loader = default_weight_loader
        self.lm_head.weight.copy_(torch.arange(1, 17).reshape(4, 4))
        record_metadata_for_reloading(self)

    def load_weights(self, weights):
        names = set()
        for name, value in weights:
            param = self.get_parameter(name)
            getattr(param, "weight_loader", default_weight_loader)(param, value)
            names.add(name)
        return names


def make_draft(target, monkeypatch, enabled=True):
    monkeypatch.setattr(draft_module.envs, "VLLM_DFLASH_FP8_DRAFT_HEAD", enabled)
    # Exercise real quantization on CPU; this does not qualify CUDA support.
    monkeypatch.setattr(draft_module, "fp8_draft_head_supported", lambda _: True)
    draft = draft_module.DFlash2Qwen3ForCausalLM.__new__(
        draft_module.DFlash2Qwen3ForCausalLM
    )
    nn.Module.__init__(draft)
    draft.lm_head = target.lm_head
    # Match the production constructor's non-persistent buffer deoperatortions.
    for name in ("_fp8_head_weight", "_fp8_head_row_scale", "_fp8_head_unit_scale"):
        draft.register_buffer(name, None, persistent=False)
    draft.maybe_init_fp8_draft_head()
    return draft


def method(path, cls, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cl = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in cl.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    exec(
        compile(
            ast.Module(body=[fn], type_ignores=[]),
            str(ROOT / path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[name]


def reload_target(target, draft, value, checkpoint):
    fn = method(
        "vllm/v1/worker/gpu_model_runner.py",
        "GPUModelRunner",
        "reload_weights",
        {
            "time": time,
            "logger": SimpleNamespace(
                info_once=lambda *_: None,
                warning_once=lambda *_: None,
                warning=lambda *_: None,
            ),
            "initialize_layerwise_reload": initialize_layerwise_reload,
            "finalize_layerwise_reload": finalize_layerwise_reload,
            "_get_parameter_for_reload": lambda m, n: m.get_parameter(n),
            "refresh_derived_weights": refresh_derived_weights,
        },
    )
    runner = SimpleNamespace(
        get_model=lambda: target,
        get_draft_model=lambda: draft,
        model_config=SimpleNamespace(quantization=None, dtype=torch.bfloat16),
        lora_config=None,
        reset_lora_state=lambda: None,
        reset_encoder_cache=lambda: None,
        reset_mm_cache=lambda: None,
    )
    fn(
        runner,
        weights_iterator=iter([("lm_head.weight", value)]),
        is_checkpoint_format=checkpoint,
    )


def assert_current(draft, captured):
    current = draft._fp8_draft_head
    expected = draft_module.quantize_draft_head(draft.lm_head.weight)
    assert current is not None
    for got, want, old in zip(current, expected, captured, strict=True):
        assert torch.equal(got.float(), want.float())
        assert got is old
        assert got.data_ptr() == old.data_ptr()


@pytest.mark.parametrize("checkpoint", [False, True])
@torch.inference_mode()
def test_target_reload_refreshes_shared_draft_and_captured_addresses(
    monkeypatch, checkpoint
):
    target = Target()
    draft = make_draft(target, monkeypatch)
    captured = draft._fp8_draft_head
    for multiplier in (3, 5):
        value = target.lm_head.weight.detach().clone() * multiplier
        reload_target(target, draft, value, checkpoint)
        assert torch.equal(target.lm_head.weight, value)
        assert_current(draft, captured)


@pytest.mark.parametrize("is_draft_update", [False, True])
@torch.inference_mode()
def test_worker_transfer_completion_refreshes_shared_copies(
    monkeypatch, is_draft_update
):
    target = Target()
    draft = make_draft(target, monkeypatch)
    captured = draft._fp8_draft_head
    events = []

    def finish():
        events.append("finish")
        target.lm_head.weight.mul_(3)

    engine = SimpleNamespace(
        finish_weight_update=finish,
        reset_weight_update_target=lambda: events.append("reset"),
    )
    worker = SimpleNamespace(
        _check_weight_transfer_engine=lambda: None,
        weight_transfer_engine=engine,
        _weight_update_active=True,
        _weight_update_is_draft=is_draft_update,
        vllm_config=None,
        get_model=lambda: target,
        get_draft_model=lambda: draft,
        model_runner=SimpleNamespace(reset_lora_state=lambda: None),
    )
    fn = method(
        "vllm/v1/worker/gpu_worker.py",
        "Worker",
        "finish_weight_update",
        {
            "set_current_vllm_config": lambda _: nullcontext(),
            "refresh_derived_weights": refresh_derived_weights,
        },
    )
    fn(worker)
    assert_current(draft, captured)
    assert events == ["finish", "reset"]
    assert not worker._weight_update_active


def test_refresh_owns_inference_mode(monkeypatch):
    with torch.inference_mode():
        target = Target()
        draft = make_draft(target, monkeypatch)
        captured = draft._fp8_draft_head
        target.lm_head.weight.mul_(3)
    refresh_derived_weights(draft)
    assert_current(draft, captured)


def test_head_buffers_follow_module_device_ownership_without_checkpoint_keys(
    monkeypatch,
):
    target = Target()
    draft = make_draft(target, monkeypatch)
    assert set(draft.state_dict()) == {"lm_head.weight"}
    assert len(dict(draft.named_buffers())) == 3
    old = draft._fp8_draft_head
    draft.to(device="meta")
    assert all(t.device.type == "meta" for t in draft._fp8_draft_head)
    assert all(a is not b for a, b in zip(old, draft._fp8_draft_head, strict=True))


def test_disabled_fp8_head_is_not_created_by_refresh(monkeypatch):
    target = Target()
    draft = make_draft(target, monkeypatch, enabled=False)
    refresh_derived_weights(draft)
    assert draft._fp8_draft_head is None
    assert not list(draft.named_buffers())


@torch.inference_mode()
def test_non_fp8_draft_model_unchanged(monkeypatch):
    target = Target()
    draft = nn.Module()
    draft.lm_head = target.lm_head
    value = target.lm_head.weight.detach().clone() * 3
    reload_target(target, draft, value, checkpoint=False)
    assert torch.equal(draft.lm_head.weight, value)
    assert not list(draft.named_buffers())
