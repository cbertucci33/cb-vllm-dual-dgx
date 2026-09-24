# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU contract tests for rank-sliced EXL3 through Sparkinfer Trellis.

Rank-sliced checkpoints always use the planned Sparkinfer PR49 API. Decode
and prefill use distinct plans when their capacities differ. The legacy
ExLlamaV3 routed-expert path is intentionally unavailable to these tests.
"""

import os
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.quantization.exl3 as exl3_module
from vllm.model_executor.layers.quantization.exl3 import Exl3MoEMethod

HIDDEN = 128
INTERMEDIATE = 128
EXPERTS = 8
TOPK = 4
MAX_BATCHED = 256


class _FakePlan:
    def __init__(self, caps):
        self.caps = caps

    def scratch_specs(self):
        return (
            SimpleNamespace(
                shape=(64,),
                dtype=torch.uint8,
                device=self.caps["device"],
            ),
        )


class _FakeTrellisApi:
    def __init__(self):
        self.planned = []
        self.bound = []

    def Caps(self, **kwargs):
        return kwargs

    def plan(self, caps):
        plan = _FakePlan(caps)
        self.planned.append(caps)
        return plan

    def bind(self, plan, *, scratch, a, weights, topk_weights, topk_ids):
        del scratch, weights, topk_weights, topk_ids
        self.bound.append((plan, int(a.shape[0])))
        return SimpleNamespace(plan=plan, m=int(a.shape[0]))

    def run(self, *, binding):
        return torch.zeros((binding.m, HIDDEN), dtype=torch.float32)


def _make_layer():
    return SimpleNamespace(
        exl3_max_num_batched_tokens=MAX_BATCHED,
        exl3_hidden_size=HIDDEN,
        exl3_intermediate_size_per_partition=INTERMEDIATE,
        local_num_experts=EXPERTS,
        exl3_trellis_tile_config=(64, 128, 64, 128),
        exl3_trellis_weights=object(),
    )


def _make_method():
    method = object.__new__(Exl3MoEMethod)
    method.quant_config = SimpleNamespace(
        bits=3.0,
        rank_sliced_metadata={"tp": 4},
    )
    return method


class _Harness:
    def __init__(self, env=None):
        self._env = dict(env or {})
        self._saved_env = {}
        self._saved_capturing = None
        self.api = _FakeTrellisApi()

    def __enter__(self):
        for name, value in self._env.items():
            self._saved_env[name] = os.environ.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._saved_loader = exl3_module._load_sparkinfer_trellis
        self._saved_exl3_loader = exl3_module._load_exl3_ext
        exl3_module._load_sparkinfer_trellis = lambda: self.api

        def fail_if_called():
            raise AssertionError("rank-sliced EXL3 must not load exllamav3_ext")

        exl3_module._load_exl3_ext = fail_if_called
        self._saved_capturing = torch.cuda.is_current_stream_capturing
        torch.cuda.is_current_stream_capturing = lambda: False
        exl3_module._RANK_SLICED_RUNTIMES.clear()
        return self

    def __exit__(self, *exc):
        exl3_module._load_sparkinfer_trellis = self._saved_loader
        exl3_module._load_exl3_ext = self._saved_exl3_loader
        torch.cuda.is_current_stream_capturing = self._saved_capturing
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        exl3_module._RANK_SLICED_RUNTIMES.clear()
        return False


def _apply(method, layer, m):
    x = torch.zeros((m, HIDDEN), dtype=torch.bfloat16)
    weights = torch.zeros((m, TOPK), dtype=torch.float32)
    ids = torch.zeros((m, TOPK), dtype=torch.int64)
    return method._apply_rank_sliced(layer, x, weights, ids)


def test_rank_sliced_dispatch_uses_trellis_for_decode_and_prefill():
    with _Harness() as h:
        method = _make_method()
        layer = _make_layer()

        for m in (1, 2, 16, 32, 200):
            out = _apply(method, layer, m)
            assert out.dtype == torch.bfloat16
            assert out.shape == (m, HIDDEN)

        assert [
            (caps["max_tokens"], caps["block_size_m"]) for caps in h.api.planned
        ] == [(32, 8), (MAX_BATCHED, 64)]
        assert [
            (plan.caps["max_tokens"], m) for plan, m in h.api.bound
        ] == [(32, 1), (32, 2), (32, 16), (32, 32), (MAX_BATCHED, 200)]


def test_rank_sliced_dispatch_rejects_unplanned_capacity():
    with _Harness() as h:
        method = _make_method()
        layer = _make_layer()
        _apply(method, layer, 16)
        with pytest.raises(
            ValueError,
            match=rf"m={MAX_BATCHED + 1}, capacity={MAX_BATCHED}",
        ):
            _apply(method, layer, MAX_BATCHED + 1)
        assert len(h.api.planned) == 2


def test_rank_sliced_prefill_block_size_is_configurable():
    with _Harness(env={"VLLM_EXL3_PREFILL_BLOCK_M": "48"}) as h:
        _apply(_make_method(), _make_layer(), 40)
        assert h.api.planned[-1]["block_size_m"] == 48


def test_rank_sliced_rejects_a_parity_window_override():
    with _Harness(env={"VLLM_EXL3_TRELLIS_MIN_M": "4"}) as h:
        with pytest.raises(ValueError, match="must be 1"):
            _apply(_make_method(), _make_layer(), 16)
        assert not h.api.planned
