# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for target-specific projection and input preparation.

Run kernel contracts with TRITON_INTERPRET=1 and no CUDA device. They execute
production Triton bodies, not compiled GPU kernels or CUDA graph replay.
"""

import copy
import os
from types import SimpleNamespace as NS
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.models.dflash_policy import resolve_dflash_execution_policy
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    DFlashSpeculator,
    pad_dflash_buffers,
    prepare_dflash_inputs,
)


def config(target="glm5_next", arch="Glm5NextForConditionalGeneration", algo="MXFP8"):
    return NS(
        model_config=NS(hf_config=NS(model_type=target, architectures=[arch])),
        speculative_config=NS(
            draft_model_config=NS(
                hf_config=NS(
                    model_type="qwen3",
                    architectures=["DFlash2DraftModel"],
                    quantization_config={
                        "quant_method": "modelopt",
                        "quantization": {
                            "quant_algo": algo,
                        },
                    },
                )
            )
        ),
    )


def test_target_identity_separates_same_qwen_drafter():
    glm = resolve_dflash_execution_policy(config())
    assert glm.fused_mxfp8_context_kv and glm.parallel_padding
    assert glm is resolve_dflash_execution_policy(copy.deepcopy(config()))
    for target, arch, algo in [
        ("mimo_v2", "MiMoV2ForCausalLM", "MXFP8"),
        ("glm5_next", "UnknownModel", "MXFP8"),
        ("glm5_next", "Glm5NextForConditionalGeneration", "NVFP4"),
        ("qwen3", "DFlash2DraftModel", "MXFP8"),
    ]:
        policy = resolve_dflash_execution_policy(config(target, arch, algo))
        assert not policy.fused_mxfp8_context_kv and not policy.parallel_padding


def make_model(target="glm5_next", bias=False):
    model = object.__new__(DFlashQwen3Model)
    torch.nn.Module.__init__(model)
    model.config = NS(hidden_size=64)
    model.hidden_norm = NS(weight=torch.ones(64, dtype=torch.bfloat16))
    model.dflash_execution_policy = resolve_dflash_execution_policy(
        config(
            target,
            "Glm5NextForConditionalGeneration"
            if target == "glm5_next"
            else "MiMoV2ForCausalLM",
        )
    )
    layers = []
    for i in range(2):
        weight = (torch.arange(12 * 64).reshape(12, 64) % 7 - 3).float()
        weight = weight.to(torch.float8_e4m3fn)
        scales = torch.full((12, 2), 126 + i, dtype=torch.uint8)
        proj = NS(
            weight=weight,
            weight_scale=scales,
            quant_method=object(),
            bias=torch.arange(12).to(torch.bfloat16) if bias else None,
        )
        layers.append(
            NS(
                qkv_proj=proj,
                q_size=8,
                kv_size=2,
                k_norm=NS(weight=torch.ones(2, dtype=torch.bfloat16)),
            )
        )
    return model, layers


@pytest.mark.parametrize("num_ctx", [1, 8, 37])
@pytest.mark.parametrize("bias", [False, True])
def test_glm_context_projection_uses_only_dequantized_kv(num_ctx, bias):
    model, layers = make_model(bias=bias)
    model._build_context_kv_buffers(layers, bias)
    assert model._fused_kv_weight.shape == (8, 64)
    x = (torch.arange(num_ctx * 64).reshape(num_ctx, 64) % 5 - 2).to(torch.bfloat16)

    # Only the unchanged RMSNorm custom op is substituted; projection is production.
    def norm(out, source, weight, eps):
        out.copy_(F.rms_norm(source, (64,), weight, eps))

    model._rms_norm_eps = 1e-5
    # Emulate postload packing clearing original tensors. The retained fused copy
    # must survive and must not call either full quantized QKV projection.
    for layer in layers:
        layer.qkv_proj.weight = torch.empty(0)
        layer.qkv_proj.weight_scale = torch.empty(0)
    with patch("vllm.model_executor.models.qwen3_dflash.ops.rms_norm", norm):
        k, v = model._project_context_kv(x, num_ctx, 2, 1, 2)
    rows = (torch.arange(12 * 64).reshape(12, 64) % 7 - 3).float()[8:]
    weights = torch.cat([rows * 0.5, rows]).to(torch.bfloat16)
    biases = torch.arange(12)[8:].repeat(2).to(torch.bfloat16) if bias else None
    reference = F.linear(
        F.rms_norm(x, (64,), model.hidden_norm.weight, 1e-5), weights, biases
    )
    reference = reference.view(num_ctx, 2, 2, 1, 2).permute(2, 1, 0, 3, 4)
    torch.testing.assert_close(k, reference[0], rtol=0, atol=0)
    torch.testing.assert_close(v, reference[1], rtol=0, atol=0)
    assert k.is_contiguous() and v.is_contiguous()


@pytest.mark.parametrize(
    "invalid", ["scale_shape", "scale_dtype", "weight_shape", "compute_dtype"]
)
def test_glm_invalid_unpacked_format_fails_explicitly(invalid):
    model, layers = make_model()
    proj = layers[0].qkv_proj
    if invalid == "scale_shape":
        proj.weight_scale = torch.empty(12, 1, dtype=torch.uint8)
    elif invalid == "scale_dtype":
        proj.weight_scale = proj.weight_scale.float()
    elif invalid == "weight_shape":
        proj.weight = proj.weight[:10]
    else:
        model.hidden_norm.weight = model.hidden_norm.weight.float()
    with pytest.raises(ValueError, match="unpacked TP-local ModelOpt"):
        model._build_context_kv_buffers(layers, False)


def test_mimo_keeps_quantized_projection_dispatch():
    model, layers = make_model("mimo_v2")
    calls = []

    class Projection:
        quant_method = object()
        bias = None

        def __init__(self, index):
            self.index = index

        def __call__(self, x):
            calls.append((self.index, x.shape[0]))
            return torch.full((x.shape[0], 12), self.index + 1, dtype=x.dtype), None

    for i, layer in enumerate(layers):
        layer.qkv_proj = Projection(i)
    model._build_context_kv_buffers(layers, False)
    assert model._fused_kv_weight is None
    model._rms_norm_eps = 1e-5
    with patch(
        "vllm.model_executor.models.qwen3_dflash.ops.rms_norm",
        lambda out, x, *_: out.copy_(x),
    ):
        k, v = model._project_context_kv(
            torch.ones(8, 64, dtype=torch.bfloat16), 8, 2, 1, 2
        )
    assert calls == [(0, 8), (1, 8)]
    assert torch.equal(k[0], torch.ones_like(k[0]))
    assert torch.equal(v[1], torch.full_like(v[1], 2))


def inputs(num_reqs, max_reqs=4, max_tokens=2057):
    def full(n, dtype=torch.int64):
        return torch.full((n,), -777, dtype=dtype)

    lengths = np.arange(2, num_reqs + 2, dtype=np.int32)
    starts = np.concatenate(([0], lengths.cumsum())).astype(np.int32)
    batch = NS(
        num_reqs=num_reqs,
        num_scheduled_tokens=lengths,
        positions=torch.cat([torch.arange(10, 10 + int(n)) for n in lengths]),
        query_start_loc=torch.tensor(starts, dtype=torch.int32),
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32),
    )
    buffers = NS(
        input_ids=full(max_tokens, torch.int32),
        positions=full(max_tokens),
        query_start_loc=full(max_reqs + 1, torch.int32),
        seq_lens=full(max_reqs, torch.int32),
    )
    return [
        buffers,
        full(max_tokens),
        full(max_tokens),
        full(max_tokens),
        full(max_reqs * 3),
        full(max_reqs * 3),
        full(max_reqs * 3, torch.int32),
        torch.zeros(max_reqs),
        full(max_reqs),
        batch,
        torch.ones(num_reqs, dtype=torch.int32),
        torch.arange(num_reqs, dtype=torch.int32) % 2,
        torch.arange(max_reqs, dtype=torch.int64) + 20,
        torch.arange(max_reqs, dtype=torch.int32) + 40,
        torch.arange(max_reqs).float(),
        torch.arange(max_reqs, dtype=torch.int64),
        torch.arange(1, max_reqs * 16 + 1, dtype=torch.int32).view(max_reqs, 16),
        4,
        0,
        1,
        1,
        99,
        4,
        3,
        max_reqs,
        max_tokens,
        128,
        False,
        False,
        True,
    ]


@pytest.mark.parametrize("num_reqs", [1, 3, 4])
def test_parallel_padding_matches_serial_production_kernel(num_reqs):
    serial = inputs(num_reqs)
    parallel = copy.deepcopy(serial)
    serial[-2] = False
    parallel[-2] = True
    prepare_dflash_inputs(*serial)
    prepare_dflash_inputs(*parallel)
    for left, right in zip(serial[:9], parallel[:9]):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        else:
            for name in vars(left):
                assert torch.equal(getattr(left, name), getattr(right, name)), name
    # Check explicit external contract as well as equivalent old/new execution.
    assert torch.all(parallel[1][num_reqs * 4 :] == -1)
    assert torch.all(parallel[6][num_reqs * 3 :] == -1)
    assert torch.all(parallel[0].query_start_loc[num_reqs:] == num_reqs * 4)
    assert torch.all(parallel[0].seq_lens[num_reqs:] == 0)


@pytest.mark.parametrize("max_reqs,max_tokens", [(4, 16), (400, 2057)])
def test_zero_request_padding_warmup_initializes_all_tails(max_reqs, max_tokens):
    args = inputs(1, max_reqs, max_tokens)
    pad_dflash_buffers(
        args[0], args[1], args[4], args[5], args[6], 0, 4, 3, max_reqs, max_tokens
    )
    assert torch.all(args[0].query_start_loc == 0)
    assert torch.all(args[0].seq_lens == 0)
    assert torch.all(args[1] == -1)
    assert torch.all(args[4] == 0)
    assert torch.all(args[5] == 0)
    assert torch.all(args[6] == -1)


def test_capture_warms_all_group_slots_before_graph_manager():
    args = inputs(1, 4, 16)
    query_slots = torch.full((2, 16), -777, dtype=torch.int64)
    observed = []

    graph_context = lambda num_reqs, attn_metadata: None

    def capture(*_args, **_kwargs):
        assert _kwargs["prepare_forward"] is graph_context
        assert torch.all(query_slots == -1)
        assert torch.all(args[4] == 0)
        assert torch.all(args[5] == 0)
        assert torch.all(args[6] == -1)
        observed.append(True)

    spec = NS(
        _speculator_name="DFlash2",
        sample_indices=args[4],
        sample_pos=args[5],
        sample_idx_mapping=args[6],
        _context_slot_mappings=torch.full((2, 16), 77),
        query_cudagraph_manager=NS(capture=capture),
        dflash_execution_policy=resolve_dflash_execution_policy(config()),
        draft_kv_cache_group_ids=[0, 1],
        input_buffers=args[0],
        block_tables=NS(slot_mappings=query_slots),
        num_query_per_req=4,
        num_speculative_steps=3,
        max_num_reqs=4,
        max_num_tokens=16,
        max_query_tokens=16,
        _generate_draft=None,
        _prepare_graph_forward=graph_context,
        attn_groups=[],
        kv_cache_config=None,
        max_model_len=128,
        _group_causal=[False, False],
    )
    DFlashSpeculator.capture(spec)
    assert observed == [True]
    assert torch.all(spec._context_slot_mappings == -1)


def test_cpu_contracts_do_not_initialize_cuda():
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "arch",
    [
        "MiMoV2ForCausalLM",
        "MiMoV2FlashForCausalLM",
        "MiMoV2OmniForCausalLM",
    ],
)
def test_mimo_standard_dflash_padding_does_not_enable_mxfp8(arch):
    cfg = config("mimo_v2", arch, "NONE")
    cfg.speculative_config.draft_model_config.hf_config.architectures = [
        "DFlashDraftModel"
    ]
    cfg.speculative_config.draft_model_config.hf_config.quantization_config = {}
    before = copy.deepcopy(cfg)
    policy = resolve_dflash_execution_policy(cfg)
    assert policy.parallel_padding
    assert not policy.fused_mxfp8_context_kv
    assert cfg == before
    assert policy is resolve_dflash_execution_policy(copy.deepcopy(cfg))
    assert resolve_dflash_execution_policy(config()).fused_mxfp8_context_kv


@pytest.mark.parametrize("k", range(1, 8))
@pytest.mark.parametrize("private_ring", [False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_mimo_padding_preserves_slots_across_batch_changes(k, private_ring, graph):
    """Execute real Triton bodies on CPU, or native CUDA and graph replay.

    GPU execution is separate acceptance; TRITON_INTERPRET never proves graph
    replay. All layouts keep an anchor plus K sampled mask positions.
    """
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        synthesize_draft_ring_block_tables,
    )

    interpreted = os.environ.get("TRITON_INTERPRET") == "1"
    if interpreted and graph:
        pytest.skip("CUDA graph replay requires separate native execution")
    if not interpreted and not torch.cuda.is_available():
        pytest.skip("native padding regression requires CUDA")
    device = torch.device("cpu" if interpreted else "cuda")

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, NS):
            return NS(**{key: move(item) for key, item in vars(value).items()})
        return value

    outputs = None
    for num_reqs in [3, 1, 4, 2]:
        args = inputs(num_reqs)
        args[4] = torch.full((4 * k,), -777, dtype=torch.int64)
        args[5] = args[4].clone()
        args[6] = torch.full((4 * k,), -777, dtype=torch.int32)
        args[22], args[23] = k + 1, k
        # Noncontiguous request owners plus mixed prefill/verification.
        args[9].idx_mapping = torch.tensor([3, 1, 0, 2][:num_reqs], dtype=torch.int32)
        args[10][0] = 0
        args = [move(arg) for arg in args]
        if outputs is not None:
            args[:9] = outputs
        for group in range(2):
            table = args[16]
            if private_ring:
                lengths = []
                for request in range(num_reqs):
                    end = int(args[9].query_start_loc[request + 1])
                    lengths.append(int(args[9].positions[end - 1]) + 1)
                synthesize_draft_ring_block_tables(
                    table,
                    args[9].idx_mapping,
                    torch.tensor(lengths, dtype=torch.int32, device=device),
                    4,
                    1 + group * 100,
                    4,
                    k + 1,
                )
            else:
                table.copy_(
                    torch.arange(
                        1 + group * 100,
                        65 + group * 100,
                        dtype=torch.int32,
                        device=device,
                    ).view(4, 16)
                )
                # A null/evicted block must never become a writable cache slot.
                table[0, 3] = 0
            serial = copy.deepcopy(args)
            serial[-2] = False
            args[-2], args[-1] = True, group == 0
            prepare_dflash_inputs(*serial)
            prepare_dflash_inputs(*args)
            if graph:
                captured = torch.cuda.CUDAGraph()
                with torch.cuda.graph(captured):
                    prepare_dflash_inputs(*args)
                captured.replay()
            for left, right in zip(serial[:9], args[:9]):
                if isinstance(left, torch.Tensor):
                    assert torch.equal(left, right)
                else:
                    for name in vars(left):
                        assert torch.equal(getattr(left, name), getattr(right, name))
            assert torch.all(args[1][num_reqs * (k + 1) :] == -1)
            assert torch.all(args[6][num_reqs * k :] == -1)
            assert torch.all(args[4][num_reqs * k :] == 0)
            assert torch.all(args[0].seq_lens[num_reqs:] == 0)
            for request in range(num_reqs):
                end = int(args[9].query_start_loc[request + 1])
                valid = end - int(args[11][request])
                last = int(args[9].positions[valid - 1])
                owner = int(args[9].idx_mapping[request])
                base = request * (k + 1)
                assert args[0].input_ids[base].item() == (
                    args[13][owner] if args[10][request] == 0 else args[12][owner]
                )
                assert args[0].input_ids[base + 1 : base + k + 1].tolist() == [99] * k
                assert args[4][request * k : (request + 1) * k].tolist() == list(
                    range(base + 1, base + k + 1)
                )
                assert args[5][request * k : (request + 1) * k].tolist() == list(
                    range(last + 2, last + k + 2)
                )
                assert args[6][request * k : (request + 1) * k].tolist() == [owner] * k
                for offset in range(k + 1):
                    position = last + 1 + offset
                    block_id = int(table[request, position // 4])
                    expected = block_id * 4 + position % 4 if block_id else -1
                    assert args[1][base + offset].item() == expected
                assert torch.all(args[3][valid:end] == -1)
        outputs = args[:9]


@pytest.mark.parametrize("k", range(1, 8))
@pytest.mark.parametrize("parallel", [False, True])
def test_dflash_query_footprint_and_group_local_writes(k, parallel):
    """Context capacity stays large; only query-visible slots and owners change."""
    previous = None
    for num_reqs in (3, 1, 4, 2):
        args = inputs(num_reqs)
        args[4] = torch.full((4 * k,), -777, dtype=torch.int64)
        args[5] = args[4].clone()
        args[6] = torch.full((4 * k,), -777, dtype=torch.int32)
        args[22], args[23] = k + 1, k
        spec = object.__new__(DFlashSpeculator)
        spec.max_num_tokens = 2057
        spec.max_num_reqs = 4
        spec.num_query_per_req = k + 1
        args[25] = spec.max_query_tokens
        assert args[25] == 4 * (k + 1)
        assert spec.max_num_tokens == 2057
        if previous is not None:
            args[:9] = previous
        args[-2] = parallel
        prepare_dflash_inputs(*args)
        shared = copy.deepcopy([args[i] for i in (0, 2, 4, 5, 6, 7, 8)])
        first_query, first_context = args[1].clone(), args[3].clone()
        # A second group gets distinct physical pages. Common fields must not
        # be re-derived from changed inputs or redundantly overwritten.
        args[16] += 100
        args[12].fill_(12345)
        args[14].fill_(42)
        args[15].fill_(43)
        args[-2], args[-1] = parallel, False
        prepare_dflash_inputs(*args)
        for index, expected in zip((0, 2, 4, 5, 6, 7, 8), shared):
            if isinstance(expected, NS):
                for name, tensor in vars(expected).items():
                    assert torch.equal(getattr(args[index], name), tensor)
            else:
                assert torch.equal(args[index], expected)
        active = num_reqs * (k + 1)
        assert torch.equal(args[1][:active], first_query[:active] + 400)
        assert torch.all(args[1][active : args[25]] == -1)
        assert torch.all(args[1][args[25] :] == -777)
        context_end = int(args[9].query_start_loc[-1])
        resident = first_context[:context_end] >= 0
        assert torch.equal(
            args[3][:context_end][resident],
            first_context[:context_end][resident] + 400,
        )
        assert torch.equal(
            args[3][:context_end][~resident], first_context[:context_end][~resident]
        )
        # Storage beyond this turn's target rows is not consumed by
        # _precompute_context_kv(start, end); graph-only rows are padded there.
        assert torch.equal(args[3][context_end:], first_context[context_end:])
        previous = args[:9]


@pytest.mark.parametrize("k", range(1, 8))
@pytest.mark.parametrize("parallel", [False, True])
def test_dflash_800k_boundary_never_aliases_last_cache_block(k, parallel):
    """Large logical positions require only a small block table, not a KV pool."""
    args = inputs(1)
    args[4] = torch.full((4 * k,), -777, dtype=torch.int64)
    args[5] = args[4].clone()
    args[6] = torch.full((4 * k,), -777, dtype=torch.int32)
    args[22], args[23] = k + 1, k
    args[9].positions = torch.tensor([799996, 799997], dtype=torch.int64)
    args[16] = torch.zeros((4, 6250), dtype=torch.int32)
    args[16][:, -1] = 31
    args[17] = 128
    args[26] = 800000
    args[-2] = parallel
    prepare_dflash_inputs(*args)
    assert args[1][:2].tolist() == [4094, 4095]
    assert torch.all(args[1][2 : k + 1] == -1)
    assert args[0].seq_lens[0] == 800000


@pytest.mark.parametrize("target_type", ["glm5_next", "mimo_v2"])
def test_dflash_loader_does_not_remap_target_quantization(monkeypatch, target_type):
    """Draft initialization must not replace the target's packed-module map."""
    from dataclasses import dataclass

    from vllm.config import AttentionConfig, LoadConfig
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
    from vllm.v1.worker.gpu.spec_decode.dflash import utils

    @dataclass
    class Config:
        model_config: object
        quant_config: object
        speculative_config: object
        attention_config: object
        cache_config: object
        load_config: object

    class Quant:
        def __init__(self, mapping):
            self.packed_modules_mapping = mapping
            self.mapper_calls = []

        def apply_vllm_mapper(self, mapper):
            self.mapper_calls.append(mapper)

    target_map = {"target_fusion": ["left", "right"]}
    target_quant = Quant(target_map)
    draft_quant = Quant({}) if target_type == "glm5_next" else None
    target_config = NS(hf_config=NS(model_type=target_type))
    draft_config = NS(hf_config=NS(num_hidden_layers=5))
    config = Config(
        model_config=target_config,
        quant_config=target_quant,
        speculative_config=NS(
            draft_model_config=draft_config,
            attention_backend=None,
            kv_cache_dtype=None,
        ),
        attention_config=AttentionConfig(),
        cache_config=object(),
        load_config=LoadConfig(),
    )
    seen = {}

    class Captured(Exception):
        pass

    def capture_model(*, vllm_config, model_config):
        seen["config"] = vllm_config
        assert model_config is draft_config
        if vllm_config.quant_config is not None:
            configure_quant_config(vllm_config.quant_config, DFlashQwen3ForCausalLM)
        raise Captured

    monkeypatch.setattr(utils, "get_draft_load_config", lambda _: config.load_config)
    monkeypatch.setattr(utils, "get_pp_safe_draft_load_config", lambda value: value)
    monkeypatch.setattr(
        utils, "get_draft_quant_config", lambda _: draft_quant, raising=False
    )
    monkeypatch.setattr(utils, "get_model", capture_model)
    with pytest.raises(Captured):
        utils.load_dflash_model(object(), config)

    assert target_quant.packed_modules_mapping is target_map
    assert not target_quant.mapper_calls
    assert seen["config"].quant_config is draft_quant
    assert seen["config"].model_config is target_config
    assert config.quant_config is target_quant
    if draft_quant is not None:
        assert draft_quant.packed_modules_mapping == (
            DFlashQwen3ForCausalLM.packed_modules_mapping
        )


@pytest.mark.parametrize("entry", ["load", "metadata", "legacy"])
@pytest.mark.parametrize("target_type", ["glm5_next", "mimo_v2"])
@pytest.mark.parametrize("explicit", [False, True])
def test_dflash_attention_backend_does_not_inherit_target_group_overrides(
    monkeypatch, entry, target_type, explicit
):
    """Run the actual config producers and selector, stopping before CUDA/load."""
    from dataclasses import dataclass

    from vllm.config import (
        AttentionConfig,
        CacheConfig,
        LoadConfig,
        set_current_vllm_config,
    )
    from vllm.v1.attention import selector
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as Backend
    from vllm.v1.spec_decode.dflash import DFlashProposer
    from vllm.v1.worker.gpu.spec_decode.dflash import utils

    @dataclass
    class Config:
        model_config: object
        quant_config: object
        speculative_config: object
        attention_config: object
        cache_config: object
        load_config: object
        parallel_config: object
        kv_transfer_config: object = None

    target_backend = (
        Backend.FLASHINFER_MLA_SPARSE_SM120
        if target_type == "glm5_next"
        else Backend.TRITON_ATTN_DIFFKV
    )
    group_backend = (
        Backend.FLASH_ATTN if target_type == "glm5_next" else Backend.TRITON_ATTN_DIFFKV
    )
    target_attention = AttentionConfig(
        backend=target_backend,
        backend_per_kind={
            "full_attention": group_backend,
            "sliding_window": group_backend,
        },
    )
    target_groups = target_attention.backend_per_kind
    draft_backend = Backend.TRITON_ATTN if explicit else None
    draft_config = NS(hf_config=NS(num_hidden_layers=5))
    target = Config(
        model_config=NS(
            hf_config=NS(model_type=target_type),
            model_arch_config=NS(is_mm_prefix_lm=False),
            rswa_window=None,
        ),
        quant_config=None,
        speculative_config=NS(
            draft_model_config=draft_config,
            attention_backend=draft_backend,
            kv_cache_dtype="fp8_e4m3",
            moe_backend=None,
            enable_adaptive_verification=False,
        ),
        attention_config=target_attention,
        cache_config=CacheConfig(),
        load_config=LoadConfig(),
        parallel_config=NS(
            prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
    )
    if entry == "load":
        captured = {}

        class Captured(Exception):
            pass

        def capture_model(*, vllm_config, model_config):
            assert model_config is draft_config
            captured["config"] = vllm_config
            raise Captured

        monkeypatch.setattr(
            utils, "get_draft_load_config", lambda _: target.load_config
        )
        monkeypatch.setattr(utils, "get_pp_safe_draft_load_config", lambda x: x)
        monkeypatch.setattr(utils, "get_draft_quant_config", lambda _: None)
        monkeypatch.setattr(utils, "get_model", capture_model)
        with pytest.raises(Captured):
            utils.load_dflash_model(object(), target)
        draft = captured["config"]
    elif entry == "metadata":
        speculator = object.__new__(DFlashSpeculator)
        speculator.vllm_config = target
        speculator.speculative_config = target.speculative_config
        speculator.requires_non_causal = True
        draft = speculator.attn_vllm_config
    else:
        proposer = object.__new__(DFlashProposer)
        proposer.vllm_config = target
        proposer.speculative_config = target.speculative_config
        proposer.dflash_causal = False
        draft = proposer._create_draft_vllm_config()

    # The real selector applies per-kind overrides before platform selection.
    # Capture that boundary without importing or invoking a CUDA backend.
    monkeypatch.setattr(
        selector, "_cached_get_attn_backend", lambda *, backend, **_: backend
    )
    with set_current_vllm_config(draft):
        for sliding in (False, True):
            selected = selector.get_attn_backend(
                head_size=128,
                dtype=torch.bfloat16,
                kv_cache_dtype="fp8_e4m3",
                has_sliding_window=sliding,
            )
            assert selected is draft_backend

    assert draft.attention_config.use_non_causal is True
    assert draft.cache_config.cache_dtype == "fp8_e4m3"
    assert draft.attention_config.backend_per_kind == {}
    assert draft.attention_config is not target_attention
    assert target.attention_config is target_attention
    target_attention = cast(AttentionConfig, target.attention_config)
    assert target_attention.backend is target_backend
    assert target_attention.backend_per_kind is target_groups
    assert target_groups == {
        "full_attention": group_backend,
        "sliding_window": group_backend,
    }
    assert target_attention.use_non_causal is False
