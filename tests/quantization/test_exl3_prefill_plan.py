# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Planning and native numerical contracts for rank-sliced EXL3 through Trellis.

Rank-sliced checkpoints always use the planned Sparkinfer PR49 API. Decode
and prefill use distinct plans when their capacities differ. The legacy
ExLlamaV3 routed-expert path is intentionally unavailable to these tests.
"""

import os
import struct
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.quantization.exl3 as exl3_module
from vllm.model_executor.layers.fused_moe import MoEActivation, RoutedExperts
from vllm.model_executor.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod
from vllm.platforms import current_platform

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
        assert [(plan.caps["max_tokens"], m) for plan, m in h.api.bound] == [
            (32, 1),
            (32, 2),
            (32, 16),
            (32, 32),
            (MAX_BATCHED, 200),
        ]


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


# MCG decoding of an all-zero packed word: (0 * MCG & MASK) ^ 0x3B603B60
# gives two identical fp16 numbers which the decoder adds in fp16. Every
# trellis history/window is zero, so every decoded matrix entry is this value.
_ZERO_MCG_WEIGHT = 2 * struct.unpack("<e", struct.pack("<H", 0x3B60))[0]


def _zero_mcg_full_rotation_reference(x, up_sign, limit):
    """Independent dense oracle for the real zero-word full-rotation weights.

    Keep inverse rotation/projection scales before logical SwiGLU clamping,
    and down-input rotation after it. Match the native fp16 GEMM boundaries,
    without rounding the reconstructed logical gate/up before activation.
    """
    rows = torch.arange(128, device=x.device, dtype=torch.int64)
    parity = torch.zeros((128, 128), device=x.device, dtype=torch.int64)
    for bit in range(7):
        parity ^= ((rows[:, None] & rows[None, :]) >> bit) & 1
    hadamard = (1 - 2 * parity).float() / 128**0.5
    weight = torch.full(
        (128, 128), _ZERO_MCG_WEIGHT, dtype=torch.float32, device=x.device
    )
    rotated = (x.half().float() @ hadamard).half()
    projected = (rotated.float() @ weight).half()
    gate = projected.float() @ hadamard
    up = gate * up_sign
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    activated = gate * gate.double().sigmoid().float() * up
    down_input = (activated @ hadamard).half()
    down = (down_input.float() @ weight).half()
    return (down.float() @ hadamard).to(x.dtype)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="NVIDIA CUDA required")
@pytest.mark.parametrize("limit", [10.0, None], ids=["glm-limit10", "no-limit"])
@pytest.mark.parametrize("up_sign", [1.0, -1.0], ids=["positive-up", "negative-up"])
@pytest.mark.parametrize("num_rows", [1, 32, 33])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_rank_sliced_real_rotated_swiglu_limit_and_graph_replay(
    monkeypatch, limit, up_sign, num_rows, ids_dtype
):
    """G01 regression: real planned native math, expected to fail until G01 repair.

    This is not the fake planning harness above: real packed quantized weights
    flow through RoutedExperts.forward_modular, Exl3.apply, Caps, plan, bind and
    run. The owning limit is supplied by an actual GLM text config. Construct
    only the small routed layer, not a distributed/full-model engine.
    """
    from transformers import Glm5NextTextConfig

    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("full-rotation Trellis requires SM120/SM121")
    for name in (
        "VLLM_EXL3_TRELLIS_MIN_M",
        "VLLM_EXL3_TRELLIS_MAX_M",
        "VLLM_EXL3_TRELLIS_BLOCK_M",
        "VLLM_EXL3_PREFILL_BLOCK_M",
    ):
        monkeypatch.delenv(name, raising=False)
    api = exl3_module._load_sparkinfer_trellis()
    device = torch.device("cuda", torch.accelerator.current_device_index())
    model_config = Glm5NextTextConfig(swiglu_limit=limit)
    config = Exl3Config(bits=3.0)
    config.rank_sliced_metadata = {"tp": 2}
    method = object.__new__(Exl3MoEMethod)
    method.quant_config = config
    method.moe_kernel = None
    w13 = torch.zeros((2, 1, 8, 8, 48), device=device, dtype=torch.int16)
    w2 = torch.zeros((1, 8, 8, 48), device=device, dtype=torch.int16)
    edge = torch.ones((1, 128), device=device, dtype=torch.float16)
    rotations = torch.ones((1, 384), device=device, dtype=torch.float16)
    rotations[:, 128:256] = up_sign
    weights = api.prepare_weights(
        w13,
        w2,
        gate_suh=edge,
        up_suh=edge,
        intermediate_rotations=rotations,
        down_svh=edge,
        codebook="mcg",
        mcg=0xCBAC1FED,
        tile_config=(64, 128, 64, 128),
    )
    layers = []
    for is_draft in (False, True):
        layer = object.__new__(RoutedExperts)
        torch.nn.Module.__init__(layer)
        layer.quant_method = method
        layer.activation = MoEActivation.SILU
        layer._expert_map = None
        layer.apply_router_weight_on_input = False
        layer.swiglu_limit = model_config.swiglu_limit
        layer.exl3_is_draft = is_draft
        layer.exl3_max_num_batched_tokens = 64
        layer.exl3_hidden_size = layer.exl3_intermediate_size_per_partition = 128
        layer.local_num_experts = 1
        layer.exl3_trellis_tile_config = (64, 128, 64, 128)
        layer.exl3_trellis_weights = weights
        layers.append(layer)
    x = torch.zeros((num_rows, 128), device=device, dtype=torch.bfloat16)
    # Include positive gate overflow, either up overflow sign, negative gates
    # and in-range values. The first row overflows even at decode m=1.
    amplitudes = torch.tensor([20.0, -20.0, 5.0, -5.0], device=device)
    x[:, 0] = amplitudes[torch.arange(num_rows, device=device) % 4] / (
        128 * _ZERO_MCG_WEIGHT
    )
    ids = torch.zeros((num_rows, 1), device=device, dtype=ids_dtype)
    routing = torch.ones((num_rows, 1), device=device, dtype=torch.float32)
    expected = _zero_mcg_full_rotation_reference(x, up_sign, limit)
    try:
        runtimes = [method._rank_sliced_runtime(layer, x, ids) for layer in layers]
        assert runtimes[0] is not runtimes[1]
        for arena in ("trellis_scratch", "prefill_scratch"):
            assert runtimes[0][arena].data_ptr() != runtimes[1][arena].data_ptr()
        for layer, runtime in zip(layers, runtimes):
            assert runtime["trellis_plan"].caps.max_tokens == 32
            assert runtime["prefill_plan"].caps.max_tokens == 64
            assert runtime["trellis_plan"].fused_launch.full_rotation
            assert runtime["prefill_plan"].fused_launch.full_rotation
            actual = layer.forward_modular(x, routing, ids, None, None)
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = layer.forward_modular(x, routing, ids, None, None)
            for _ in range(2):
                graph.replay()
                torch.testing.assert_close(captured, expected, rtol=2e-2, atol=2e-3)
    finally:
        method.release_runtime_resources()


@pytest.mark.parametrize("rank_sliced", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64])
def test_exl3_router_id_dtype_avoids_rank_sliced_cast(rank_sliced, input_dtype):
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

    method = _make_method()
    if not rank_sliced:
        method.quant_config.rank_sliced_metadata = None
    expected_dtype = torch.int32 if rank_sliced else torch.int64
    native_ids = torch.tensor([[0, 1, 255, 287]], dtype=input_dtype)
    # Exercise the conversion used by BaseRouter._select_experts.
    routed_ids = BaseRouter._convert_indices_dtype(
        None, native_ids, method.topk_indices_dtype
    )
    assert routed_ids.dtype == expected_dtype
    assert torch.equal(routed_ids.to(torch.int64), native_ids.to(torch.int64))
    if input_dtype == expected_dtype:
        assert routed_ids is native_ids
        assert routed_ids.data_ptr() == native_ids.data_ptr()
