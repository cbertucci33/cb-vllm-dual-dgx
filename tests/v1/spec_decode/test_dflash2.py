# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from vllm.model_executor.models.qwen3_dflash import (
    _add_global_draft_layer_exclusions,
)
from vllm.model_executor.models.qwen3_dflash2 import (
    CandidateSelector,
    DFlash2Qwen3ForCausalLM,
    DFlashGroupedConv,
    _grouped_conv,
    _score_edges,
)
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
from vllm.v1.worker.utils import AttentionGroup


def test_dflash_attention_config_uses_draft_cache_dtype():
    from vllm.config import VllmConfig

    target_config = VllmConfig()
    speculator = object.__new__(DFlashSpeculator)
    speculator.vllm_config = target_config
    speculator.speculative_config = SimpleNamespace(
        attention_backend="FLASHINFER",
        kv_cache_dtype="fp8_e4m3",
    )
    speculator.requires_non_causal = True

    config = speculator.attn_vllm_config

    assert config.cache_config.cache_dtype == "fp8_e4m3"
    assert config.attention_config.backend is not None
    assert config.attention_config.backend.name == "FLASHINFER"
    assert config.attention_config.use_non_causal is True
    assert config.cache_config is not target_config.cache_config


def test_attention_groups_receive_target_and_draft_configs(monkeypatch):
    from vllm.v1.worker.gpu import attn_utils

    target_config = SimpleNamespace(parallel_config=object())
    draft_config = SimpleNamespace(parallel_config=object())
    captured = {}

    class FakeSpec:
        has_layer_views = True

    class FakeBackend:
        def __init__(self, name):
            self.name = name

        def full_cls_name(self):
            return self.name

    layers = {
        "target": SimpleNamespace(
            get_attn_backend=lambda: FakeBackend("target-backend"), num_heads=1
        ),
        "draft": SimpleNamespace(
            get_attn_backend=lambda: FakeBackend("draft-backend"), num_heads=1
        ),
    }

    class FakeGroup:
        def __init__(self, backend, layer_names, kv_cache_spec, kv_cache_group_id):
            self.layer_names = list(layer_names)
            self.metadata_builders = []

        def create_metadata_builders(self, vllm_config, **kwargs):
            captured[self.layer_names[0]] = vllm_config

    monkeypatch.setattr(attn_utils, "AttentionGroup", FakeGroup)
    monkeypatch.setattr(
        attn_utils,
        "get_layers_from_vllm_config",
        lambda config, layer_type, names: {name: layers[name] for name in names},
    )
    monkeypatch.setattr(attn_utils, "get_shared_kv_cache_layers", lambda config: {})
    monkeypatch.setattr(
        attn_utils, "add_kv_sharing_layers_to_kv_cache_groups", lambda *args: None
    )
    monkeypatch.setattr(
        attn_utils, "get_kv_sharing_fast_prefill_eligible_layers", lambda *args: set()
    )
    monkeypatch.setattr(attn_utils, "prepare_kernel_block_sizes", lambda *args: [16])
    monkeypatch.setattr(attn_utils, "get_num_ubatches", lambda config: 1)
    monkeypatch.setattr(attn_utils, "get_attn_cg_support", lambda *args: "support")

    cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["target", "draft"], kv_cache_spec=FakeSpec())
        ]
    )
    attn_utils.init_attn_backend(
        cache_config,
        target_config,
        device=None,
        draft_layer_names={"draft"},
        draft_vllm_config=draft_config,
    )

    assert captured == {"target": target_config, "draft": draft_config}


@pytest.mark.parametrize("block_size", [5, 8])
def test_grouped_conv_matches_reference(block_size: int):
    torch.manual_seed(0)
    batch, taps, num_groups, group_size = 3, 3, 4, 2
    hidden = torch.randn(batch * block_size, num_groups * group_size)
    delta = torch.randn(batch * block_size, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = _grouped_conv(
        hidden, delta, base, block_size, num_groups, group_size, taps
    )
    hidden_blocks = hidden.view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base = base.view(taps, num_groups, group_size)
    delta = delta.view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base[tap] + delta[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(actual, expected.flatten(0, 1).flatten(-2))


def test_dflash2_projection_layers_receive_draft_quant_config(monkeypatch):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.models import qwen3_dflash2

    captured = []

    class FakeReplicatedLinear(torch.nn.Module):
        def __init__(self, *args, quant_config=None, **kwargs):
            super().__init__()
            captured.append(quant_config)

    monkeypatch.setattr(qwen3_dflash2, "ReplicatedLinear", FakeReplicatedLinear)
    quant_config = object()
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(config), torch.device("meta"):
        DFlashGroupedConv(
            hidden_size=16,
            taps=3,
            group_size=4,
            block_size=8,
            params_dtype=torch.bfloat16,
            prefix="conv",
            quant_config=quant_config,
        )
        CandidateSelector(
            hidden_size=16,
            vocab_size=32,
            rank=8,
            top_k=4,
            params_dtype=torch.bfloat16,
            prefix="selector",
            quant_config=quant_config,
        )

    assert captured == [quant_config, quant_config]


def test_dflash2_rejects_speculation_wider_than_trained_block(monkeypatch):
    from torch import nn

    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3DecoderLayer
    from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3DecoderLayer

    monkeypatch.setattr(
        DFlashQwen3DecoderLayer,
        "__init__",
        lambda self, *args, **kwargs: nn.Module.__init__(self),
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=8),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    config = SimpleNamespace(
        hidden_size=4096,
        dflash_config={
            "block_size": 8,
            "conv_kernel_size": 2,
            "conv_group_size": 16,
        },
    )

    with pytest.raises(ValueError, match="trained with block_size 8"):
        DFlash2Qwen3DecoderLayer(
            vllm_config,
            config=config,
            layer_idx=0,
            prefix="model.layers.0",
        )


def test_dflash2_rejects_target_draft_vocabulary_mismatch(monkeypatch):
    from torch import nn

    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM

    def init_base(self, **kwargs):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(
            vocab_size=154879,
            dflash_config={},
        )

    monkeypatch.setattr(DFlashQwen3ForCausalLM, "__init__", init_base)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(get_vocab_size=lambda: 154880)
    )

    with pytest.raises(
        ValueError,
        match=r"draft vocab \(154879\).+target vocab \(154880\)",
    ):
        DFlash2Qwen3ForCausalLM(vllm_config=vllm_config)


def test_candidate_selector_owns_exact_mxfp8_weight_and_scale(monkeypatch):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.quantization import modelopt
    from vllm.model_executor.models.utils import AutoWeightsLoader

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(
        modelopt,
        "select_linear_kernel",
        lambda *args, **kwargs: SimpleNamespace(input_quant_key=lambda: None),
    )
    quant_config = modelopt.ModelOptMxFp8Config(
        is_checkpoint_mxfp8_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(config):
        selector = CandidateSelector(
            hidden_size=32,
            vocab_size=64,
            rank=16,
            top_k=4,
            params_dtype=torch.bfloat16,
            prefix="candidate_selector",
            quant_config=quant_config,
        )

    projection = selector.hidden_projection
    assert projection.weight.shape == (16, 32)
    assert projection.weight.dtype == torch.float8_e4m3fn
    assert projection.weight_scale.shape == (16, 1)
    assert projection.weight_scale.dtype == torch.uint8
    assert set(dict(projection.named_parameters())) == {"weight", "weight_scale"}

    weight = torch.zeros_like(projection.weight)
    scale = torch.full_like(projection.weight_scale, 127)
    loaded = AutoWeightsLoader(selector).load_weights(
        [
            ("hidden_projection.weight", weight),
            ("hidden_projection.weight_scale", scale),
        ]
    )
    assert loaded == {"hidden_projection.weight", "hidden_projection.weight_scale"}
    torch.testing.assert_close(projection.weight, weight)
    torch.testing.assert_close(projection.weight_scale, scale)


def test_candidate_topk_accepts_precomputed_local_logits():
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    processor = object.__new__(LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.scale = 2.0
    processor.soft_cap = 2.0

    shard_indices = SimpleNamespace(
        num_org_vocab_padding=1,
        org_vocab_start_index=10,
    )
    lm_head = SimpleNamespace(tp_size=1, shard_indices=shard_indices)
    local_logits = torch.tensor([[1.0, 3.0, 2.0, 100.0]])

    ids, values = processor.get_top_k_tokens(
        lm_head,
        hidden_states=torch.empty(1, 0),
        k=2,
        local_logits=local_logits,
    )

    assert ids.tolist() == [[11, 12]]
    expected = torch.tanh(torch.tensor([[3.0, 2.0]]) / 2.0) * 2.0 * 2.0
    torch.testing.assert_close(values, expected)


def test_candidate_topk_local_logits_preserves_logprob_branch():
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    processor = object.__new__(LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.scale = 1.0
    processor.soft_cap = None
    processor.head_dtype = None
    logits = torch.tensor([[1.0, 3.0, 2.0, -5.0]])

    class QuantMethod:
        def apply(self, lm_head, hidden_states, bias=None):
            return logits.clone()

    lm_head = SimpleNamespace(
        tp_size=1,
        shard_indices=SimpleNamespace(
            num_org_vocab_padding=1,
            org_vocab_start_index=0,
        ),
        quant_method=QuantMethod(),
    )
    expected = processor.get_top_k_tokens(
        lm_head,
        hidden_states=torch.empty(1, 1),
        k=2,
        return_log_probs=True,
    )
    actual = processor.get_top_k_tokens(
        lm_head,
        hidden_states=torch.empty(1, 1),
        k=2,
        return_log_probs=True,
        local_logits=logits.clone(),
    )
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_candidate_top16_merges_tp2_shards_and_masks_padding(monkeypatch):
    from vllm.model_executor.layers import logits_processor

    processor = object.__new__(logits_processor.LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.scale = 1.0
    processor.soft_cap = None

    rank0 = torch.arange(20, dtype=torch.float32).unsqueeze(0)
    rank0[:, -2:] = 1000
    rank1_values = torch.arange(100, 116, dtype=torch.float32).unsqueeze(0)
    rank1_ids = torch.arange(120, 136, dtype=torch.int64).unsqueeze(0)
    gathered = [
        torch.cat((torch.arange(17, 1, -1).float().unsqueeze(0), rank1_values), dim=-1),
        torch.cat((torch.arange(17, 1, -1).long().unsqueeze(0), rank1_ids), dim=-1),
    ]
    monkeypatch.setattr(
        logits_processor,
        "tensor_model_parallel_all_gather",
        lambda value, dim=-1: gathered.pop(0),
    )
    lm_head = SimpleNamespace(
        tp_size=2,
        shard_indices=SimpleNamespace(
            num_org_vocab_padding=2,
            org_vocab_start_index=0,
        ),
    )

    ids, values = processor.get_top_k_tokens(
        lm_head,
        hidden_states=torch.empty(1, 0),
        k=16,
        local_logits=rank0,
    )

    assert ids.tolist() == [list(range(135, 119, -1))]
    assert values.tolist() == [list(range(115, 99, -1))]
    assert torch.isneginf(rank0[:, -2:]).all()


def test_fp8_copy_is_initialized_from_post_share_target_head(monkeypatch):
    from vllm.model_executor.models import qwen3_dflash, qwen3_dflash2
    from vllm.v1.worker.gpu.spec_decode.dflash import utils

    discarded_head = torch.nn.Linear(4, 8, bias=False)
    target_head = torch.nn.Linear(4, 8, bias=False)

    class Draft(torch.nn.Module):
        maybe_init_fp8_draft_head = DFlash2Qwen3ForCausalLM.maybe_init_fp8_draft_head

        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace()
            self.config = SimpleNamespace(has_own_lm_head=False)
            self.lm_head = discarded_head
            self._fp8_draft_head = None

    draft = Draft()
    target = SimpleNamespace(
        get_language_model=lambda: SimpleNamespace(
            model=SimpleNamespace(), lm_head=target_head
        )
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=SimpleNamespace()),
            attention_backend=None,
            kv_cache_dtype=None,
        ),
        attention_config=SimpleNamespace(),
        cache_config=SimpleNamespace(),
    )
    marker = object()
    seen: list[torch.Tensor] = []
    monkeypatch.setattr(qwen3_dflash, "dflash_has_any_non_causal", lambda cfg: False)
    monkeypatch.setattr(utils, "replace", lambda value, **kwargs: value)
    monkeypatch.setattr(utils, "get_draft_load_config", lambda cfg: None)
    monkeypatch.setattr(utils, "get_draft_quant_config", lambda cfg: None)
    monkeypatch.setattr(utils, "get_pp_safe_draft_load_config", lambda cfg: cfg)
    monkeypatch.setattr(utils, "get_model", lambda **kwargs: draft)
    monkeypatch.setattr(utils, "maybe_share_target_embed", lambda *args: None)
    monkeypatch.setattr(utils, "_should_share", lambda *args: True)
    monkeypatch.setattr(qwen3_dflash2.envs, "VLLM_DFLASH_FP8_DRAFT_HEAD", True)
    monkeypatch.setattr(qwen3_dflash2, "fp8_draft_head_supported", lambda device: True)

    def quantize(weight):
        seen.append(weight)
        return marker

    monkeypatch.setattr(qwen3_dflash2, "quantize_draft_head", quantize)

    loaded = utils.load_dflash_model(target, config)

    assert loaded is draft
    assert draft.lm_head is target_head
    assert seen == [target_head.weight]
    assert draft._fp8_draft_head is marker


@pytest.mark.parametrize(
    ("enabled", "supported"),
    [(False, True), (True, False)],
)
def test_fp8_draft_head_falls_back_when_disabled_or_unsupported(
    monkeypatch, enabled, supported
):
    from vllm.model_executor.models import qwen3_dflash2

    class Draft(torch.nn.Module):
        maybe_init_fp8_draft_head = DFlash2Qwen3ForCausalLM.maybe_init_fp8_draft_head

        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(4, 8, bias=False)
            self._fp8_draft_head = None

    draft = Draft()
    monkeypatch.setattr(qwen3_dflash2.envs, "VLLM_DFLASH_FP8_DRAFT_HEAD", enabled)
    monkeypatch.setattr(
        qwen3_dflash2, "fp8_draft_head_supported", lambda device: supported
    )
    monkeypatch.setattr(
        qwen3_dflash2,
        "quantize_draft_head",
        lambda weight: pytest.fail("fallback must not quantize the draft head"),
    )

    draft.maybe_init_fp8_draft_head()

    assert draft._fp8_draft_head is None


def test_dflash2_rejects_draft_token_id_remapping():
    model = torch.nn.Module.__new__(DFlash2Qwen3ForCausalLM)
    torch.nn.Module.__init__(model)
    model.draft_id_to_target_id = torch.arange(4)
    model.lm_head = SimpleNamespace()
    model.model = SimpleNamespace(candidate_selector=SimpleNamespace(top_k=2))
    model._fp8_head_weight = None
    model._fp8_head_row_scale = None
    model._fp8_head_unit_scale = None
    model.candidate_logits_processor = SimpleNamespace(
        get_top_k_tokens=lambda *args, **kwargs: pytest.fail(
            "candidate selection must not run with token-ID remapping"
        )
    )

    with pytest.raises(AssertionError, match="does not support"):
        model.compute_candidates(torch.randn(1, 4))


def test_exact_dflash2_candidate_geometry_uses_fp8_local_logits(monkeypatch):
    from vllm.model_executor.models import qwen3_dflash2

    model = torch.nn.Module.__new__(DFlash2Qwen3ForCausalLM)
    torch.nn.Module.__init__(model)
    model.draft_id_to_target_id = None
    model.lm_head = SimpleNamespace()
    model.model = SimpleNamespace(candidate_selector=SimpleNamespace(top_k=16))
    model._fp8_head_weight = torch.empty((8, 4), dtype=torch.float8_e4m3fn)
    model._fp8_head_row_scale = torch.ones(8)
    model._fp8_head_unit_scale = torch.ones(())

    local_logits = torch.randn(14, 32)
    seen: dict[str, object] = {}

    class Processor:
        @staticmethod
        def get_top_k_tokens(head, hidden_states, k, *, local_logits=None):
            seen.update(
                head=head,
                hidden_states=hidden_states,
                k=k,
                local_logits=local_logits,
            )
            return torch.zeros(14, 16, dtype=torch.int64), torch.zeros(14, 16)

    model.candidate_logits_processor = Processor()
    monkeypatch.setattr(
        qwen3_dflash2,
        "fp8_draft_head_logits",
        lambda hidden_states, head: local_logits,
    )
    hidden_states = torch.randn(2 * 7, 4096)

    ids, unary = model.compute_candidates(hidden_states)

    assert ids.shape == (14, 16)
    assert unary.shape == (14, 16)
    assert seen == {
        "head": model.lm_head,
        "hidden_states": hidden_states,
        "k": 16,
        "local_logits": local_logits,
    }


def test_private_ring_manager_pages_map_to_kernel_pages(monkeypatch):
    from vllm.v1.kv_cache_interface import (
        DFlashSWASpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )
    from vllm.v1.worker.gpu.spec_decode import speculator as speculator_module

    monkeypatch.setattr(
        speculator_module.DraftModelSpeculator,
        "set_attn",
        lambda *args, **kwargs: None,
    )
    spec = DFlashSWASpec(
        block_size=64,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=2048,
        private_ring=True,
    )
    manager_pages = 1639
    cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[
            KVCacheTensor(
                size=manager_pages * spec.page_size_bytes,
                layers=["draft"],
                layer_stride=manager_pages * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
                offset=0,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["draft"], spec)],
    )
    group = SimpleNamespace(
        get_metadata_builder=lambda: SimpleNamespace(
            aot_schedule=False,
            kv_cache_spec=spec,
        )
    )
    block_tables = SimpleNamespace()
    owner = object.__new__(DFlashSpeculator)
    owner.attn_groups = [[cast(AttentionGroup, group)]]
    owner.block_tables = block_tables
    owner.max_num_reqs = 3
    owner.max_num_tokens = 32
    owner.device = torch.device("cpu")
    owner.model = SimpleNamespace()
    owner.requires_non_causal = False
    owner.draft_ring_base = {}
    owner.draft_ring_size = {}

    owner.set_attn(None, cache_config, block_tables, None, [])

    assert owner.draft_ring_base == {0: 1}
    assert owner.draft_ring_size == {0: 546}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fp8_draft_head_logits_track_bf16_reference():
    from vllm.model_executor.layers.fp8_draft_head import (
        fp8_draft_head_logits,
        quantize_draft_head,
    )

    torch.manual_seed(0)
    hidden = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    actual = fp8_draft_head_logits(hidden, quantize_draft_head(weight))
    expected = hidden @ weight.t()

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected.float(), atol=1.0, rtol=0.08)


def test_exact_dflash2_cpu_grouped_conv_geometry_matches_reference():
    torch.manual_seed(2)
    batch, block_size, num_groups, group_size, taps = 2, 8, 160, 16, 2
    rows = batch * block_size
    hidden = torch.randn(rows, num_groups * group_size)
    delta = torch.randn(rows, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = _grouped_conv(
        hidden, delta, base, block_size, num_groups, group_size, taps
    )
    hidden_blocks = hidden.view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base_blocks = base.view(taps, num_groups, group_size)
    delta_blocks = delta.view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base_blocks[tap] + delta_blocks[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(actual, expected.flatten(0, 1).flatten(-2))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "batch,num_groups,group_size,block_size,taps",
    [
        (7, 13, 11, 5, 1),
        (7, 13, 11, 5, 3),
        (7, 13, 11, 8, 2),
        (16, 64, 16, 8, 2),
        (16, 160, 16, 8, 2),
    ],
)
def test_grouped_conv_triton_matches_reference(
    dtype: torch.dtype,
    batch: int,
    num_groups: int,
    group_size: int,
    block_size: int,
    taps: int,
):
    torch.manual_seed(0)
    rows = batch * block_size
    hidden = torch.randn(rows, num_groups * group_size, device="cuda", dtype=dtype)
    base = torch.randn(taps, num_groups * group_size, device="cuda", dtype=dtype)
    projected = torch.randn(rows, 2, taps, num_groups, device="cuda", dtype=dtype)
    delta = projected[:, 1]

    actual = _grouped_conv(
        hidden, delta, base, block_size, num_groups, group_size, taps
    )

    hidden_blocks = hidden.float().view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base_blocks = base.float().view(taps, num_groups, group_size)
    delta_blocks = delta.float().view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base_blocks[tap] + delta_blocks[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(
        actual,
        expected.flatten(0, 1).flatten(-2).to(dtype),
        rtol=1e-2 if dtype is torch.bfloat16 else 1e-5,
        atol=1e-2 if dtype is torch.bfloat16 else 1e-5,
    )


def test_draft_quant_exclusions_include_global_layer_indices():
    quant_config = SimpleNamespace(
        exclude_modules=[
            "layers.0.mlp_conv*",
            "*layers.4.self_attn.q_proj",
            "layers.88.already_global",
            "lilicorr.layers.0.mlp.0",
        ]
    )

    _add_global_draft_layer_exclusions(quant_config, 88, 5)

    assert "layers.88.mlp_conv*" in quant_config.exclude_modules
    assert "*layers.92.self_attn.q_proj" in quant_config.exclude_modules
    assert quant_config.exclude_modules.count("layers.88.already_global") == 1
    assert "lilicorr.layers.88.mlp.0" not in quant_config.exclude_modules


def test_selector_edges_match_sequential_reference():
    torch.manual_seed(1)
    batch, steps, top_k, rank = 2, 4, 3, 5
    vocab = 17
    predecessors = torch.randn(vocab, rank)
    successors = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k)
    hidden = torch.randn(batch, steps, rank)
    anchors = torch.randint(vocab, (batch,))

    actual = _score_edges(
        predecessors,
        successors,
        candidate_ids,
        unary,
        hidden,
        anchors,
        top_k,
    )
    expected = torch.empty_like(actual)
    for step in range(steps):
        pred = (
            anchors[:, None].expand(-1, top_k)
            if step == 0
            else candidate_ids[:, step - 1]
        )
        expected[:, step] = unary[:, step, None] + torch.einsum(
            "bpr,bcr->bpc",
            predecessors[pred] * hidden[:, step, None],
            successors[candidate_ids[:, step]],
        )

    torch.testing.assert_close(actual, expected)


def _stub_base(monkeypatch, draft_logits):
    """A DFlashSpeculator.__init__ that allocates only what the base class would.

    The real base class fills draft_logits from draft_logits_spec, so callers
    pass a tensor already in that state.
    """

    def init_base(self, _vllm_config, device):
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"selector_top_k": 3})
        )
        self.max_num_reqs = 2
        self.num_query_per_req = 5
        self.num_speculative_steps = 4
        self.vocab_size = 17
        self.draft_tokens = torch.empty((2, 4), dtype=torch.int64, device=device)
        self.draft_logits = draft_logits

    monkeypatch.setattr(DFlashSpeculator, "__init__", init_base)


def test_selector_leaves_greedy_drafting_without_proposal_logits(monkeypatch):
    """Greedy is the default, and it caches no proposal distribution.

    The base class allocates draft_logits only for "probabilistic"; verification
    reads `draft_logits is None` to decide whether a distribution is on offer, so
    allocating one here would claim a proposal the walk never sampled from.
    """
    _stub_base(monkeypatch, None)
    speculator = DFlash2Speculator(None, torch.device("cpu"))

    assert speculator.draft_logits is None


def test_selector_asks_for_fp32_proposal_logits():
    """The spec the base class allocates from: fp32, filled -inf.

    Not the head dtype -- rounding selector scores to bf16 moves the argmax of a
    candidate row often enough that the walk and the rejection sampler checking it
    would no longer read the same distribution.
    """
    dtype, fill = DFlash2Speculator.draft_logits_spec(None, None)

    assert dtype is torch.float32
    assert fill == float("-inf")


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("variant", ["dflash2", "lilicorr", "lilicorr_plain"])
def test_candidate_model_decoder_layer_cls(monkeypatch, variant):
    from types import SimpleNamespace

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.models.lilicorr import LiLiCorr
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3DecoderLayer
    from vllm.model_executor.models.qwen3_dflash2 import (
        DFlash2Qwen3DecoderLayer,
        DFlash2Qwen3Model,
    )

    # 1. Mock get_current_vllm_config and TP groups
    mock_current_vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            user_specified_block_size=False,
            kv_cache_dtype_skip_layers=[],
            cache_dtype="auto",
            sliding_window=None,
            enable_prefix_caching=False,
        ),
        kv_transfer_config=None,
        speculative_config=None,
        attention_config=SimpleNamespace(
            use_non_causal=False,
            backend=None,
            backend_per_kind={},
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(
            compile_custom_ops=False,
            custom_ops="all",
            enabled_custom_ops=set(),
            static_forward_context={},
            mode=0,  # CompilationMode.NONE is 0
        ),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_mm_prefix_lm=False,
            rswa_window=None,
        ),
        kernel_config=SimpleNamespace(
            linear_backend="auto",
        ),
    )
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform,
        "get_attn_backend_cls",
        lambda *args, **kwargs: (
            "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"
        ),
    )

    class MockGroup:
        rank_in_group = 0
        world_size = 1

    monkeypatch.setattr(
        "vllm.distributed.parallel_state._TP",
        MockGroup(),
    )

    # 2. Mock vllm_config
    hf_config = SimpleNamespace(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_parameters={},
        intermediate_size=512,
        hidden_act="silu",
        dflash_config={
            "selector_rank": 4,
            "selector_top_k": 3,
            "conv_kernel_size": 0 if variant == "lilicorr_plain" else 3,
            "conv_group_size": 0 if variant == "lilicorr_plain" else 2,
            "block_size": 5,
            "lilicorr_candidate_topk": 4,
            "lilicorr_hidden_size": 8,
            "lilicorr_num_layers": 2,
            "lilicorr_num_heads": 2,
            "lilicorr_mlp_ratio": 2.0,
            "lilicorr_factor_dim": 4,
            "lilicorr_vector_eps": 1e-6,
            "lilicorr_logit_scale": 3.0,
            "use_aux_hidden_state": False,
        },
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=hf_config,
                quantization=None,
            ),
            num_speculative_tokens=4,
            enable_adaptive_verification=False,
        ),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            is_mm_prefix_lm=False,
        ),
        load_config=SimpleNamespace(
            quantization=None,
            quantization_param_path=None,
        ),
    )
    mock_current_vllm_config.speculative_config = vllm_config.speculative_config
    vllm_config.compilation_config = mock_current_vllm_config.compilation_config

    # 3. Instantiate the model under meta device to avoid parameter allocation issues
    with set_current_vllm_config(mock_current_vllm_config), torch.device("meta"):
        model_cls = DFlash2Qwen3Model if variant == "dflash2" else LiLiCorr
        model = model_cls(vllm_config=vllm_config)

    # 4. Assert that the layers are DFlash2Qwen3DecoderLayer (the subclass)
    assert len(model.layers) == 2
    expected = (
        DFlashQwen3DecoderLayer
        if variant == "lilicorr_plain"
        else DFlash2Qwen3DecoderLayer
    )
    assert type(model.layers[0]) is expected


def test_conv_projections_use_draft_quant_config(monkeypatch):
    from torch import nn

    from vllm.distributed import parallel_state
    from vllm.model_executor.layers.quantization import modelopt
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3DecoderLayer
    from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3DecoderLayer
    from vllm.model_executor.models.utils import AutoWeightsLoader

    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(
        DFlashQwen3DecoderLayer,
        "__init__",
        lambda self, *a, **kw: nn.Module.__init__(self),
    )
    # Exercise the real quantized parameter allocation without selecting a GPU kernel.
    monkeypatch.setattr(
        modelopt,
        "select_linear_kernel",
        lambda *a, **kw: SimpleNamespace(input_quant_key=lambda: None),
    )
    quant_config = modelopt.ModelOptNvFp4Config(
        quant_method="W4A16_NVFP4", is_checkpoint_nvfp4_serialized=True
    )
    layer = DFlash2Qwen3DecoderLayer(
        SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=7),
            model_config=SimpleNamespace(dtype=torch.bfloat16),
        ),
        config=SimpleNamespace(
            hidden_size=16,
            dflash_config={"conv_kernel_size": 2, "conv_group_size": 2},
        ),
        layer_idx=0,
        prefix="model.layers.0",
        quant_config=quant_config,
    )
    for name in ("attention_conv", "mlp_conv"):
        module = getattr(layer, name)
        projection = module.kernel_projection
        assert projection.weight.dtype == torch.uint8
        assert projection.weight.shape == (32, 8)
        weight = torch.ones_like(projection.weight)
        AutoWeightsLoader(module).load_weights([("kernel_projection.weight", weight)])
        torch.testing.assert_close(projection.weight, weight)
        assert module.base_kernel.dtype == torch.bfloat16


def test_context_kv_uses_quantized_projection_fallback(monkeypatch):
    from torch import nn

    from vllm.model_executor.models import qwen3_dflash
    from vllm.model_executor.models.dflash_policy import DFlashExecutionPolicy

    class Projection(nn.Module):
        def __init__(self, packed_weight):
            super().__init__()
            self.register_buffer("packed_weight", packed_weight)
            self.quant_method = object()
            self.calls = 0

        def forward(self, hidden_states):
            self.calls += 1
            return torch.nn.functional.linear(hidden_states, self.packed_weight), None

    monkeypatch.setattr(
        qwen3_dflash.ops,
        "rms_norm",
        lambda output, hidden_states, weight, eps: output.copy_(hidden_states),
    )
    context_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    projections = [
        Projection(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [1.0, -1.0],
                ]
            )
        ),
        Projection(
            torch.tensor(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                    [-1.0, 0.0],
                    [0.0, -1.0],
                ]
            )
        ),
    ]
    model = qwen3_dflash.DFlashQwen3Model.__new__(qwen3_dflash.DFlashQwen3Model)
    nn.Module.__init__(model)
    model.dflash_execution_policy = DFlashExecutionPolicy()
    model.hidden_norm = SimpleNamespace(weight=nn.Parameter(torch.ones(2)))
    model._rms_norm_eps = 1e-6
    layers_attn = [
        SimpleNamespace(
            qkv_proj=projection,
            q_size=2,
            k_norm=SimpleNamespace(weight=nn.Parameter(torch.ones(2))),
        )
        for projection in projections
    ]
    qwen3_dflash.DFlashQwen3Model._build_context_kv_buffers(
        model,
        layers_attn,
        has_bias=False,
    )
    assert model._fused_kv_weight is None
    assert all(not hasattr(projection, "weight") for projection in projections)

    actual_k, actual_v = qwen3_dflash.DFlashQwen3Model._project_context_kv(
        model,
        context_states,
        num_ctx=2,
        num_layers=2,
        num_kv_heads=1,
        head_dim=2,
    )

    expected_k = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [6.0, 8.0]]]
    ).unsqueeze(2)
    expected_v = torch.tensor(
        [[[3.0, -1.0], [7.0, -1.0]], [[-1.0, -2.0], [-3.0, -4.0]]]
    ).unsqueeze(2)
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)
    assert [projection.calls for projection in projections] == [1, 1]


@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("use_aux", [False, True])
@pytest.mark.parametrize("own_weights", [False, True])
def test_dflash_load_weights_releases_consumed_inputs(with_mask, use_aux, own_weights):
    """Stream inputs without retaining the checkpoint through fused-buffer setup."""
    import gc
    import weakref

    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM

    draft = torch.nn.Module()
    draft.supports_eagle = True
    draft.has_own_lm_head = False
    draft.has_own_embed_tokens = False
    draft.model = torch.nn.Module()
    draft.model.use_aux_hidden_state = use_aux
    draft.model.has_separate_mask_embedding = False
    draft.model.fc = torch.nn.Linear(2, 2, bias=False)
    draft.model.embed_tokens = torch.nn.Embedding(2, 2)
    draft.lm_head = torch.nn.Linear(2, 2, bias=False)
    draft.register_parameter(
        "draft_id_to_target_id", torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    )
    draft.model.register_parameter(
        "mask_embedding", torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    )
    for index in range(8):
        draft.model.register_parameter(
            f"weight{index}", torch.nn.Parameter(torch.zeros(2), requires_grad=False)
        )

    embed_before = draft.model.embed_tokens.weight.detach().clone()
    head_before = draft.lm_head.weight.detach().clone()
    refs = []
    weight_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    early_release = []

    def weights():
        specs: list[tuple[str, tuple[int, ...], int]] = [
            (f"weight{i}", (2,), i + 1) for i in range(8)
        ]
        specs += [
            ("fc.weight", (2, 2), 9),
            ("embed_tokens.weight", (2, 2), 10),
            ("lm_head.weight", (2, 2), 11),
            ("d2t", (2,), 12),
            ("t2d", (2,), 13),
        ]
        for name, shape, value in specs:
            if not own_weights and name in {"embed_tokens.weight", "lm_head.weight"}:
                continue
            if name == "weight7":
                gc.collect()
                early_release.append(weight_refs[0]() is None)
            tensor = torch.full(shape, float(value))
            refs.append(weakref.ref(tensor))
            weight_refs.append(weakref.ref(tensor))
            yield name, tensor
            del tensor

    def read_mask():
        if not with_mask:
            return None
        tensor = torch.full((2,), 14.0)
        refs.append(weakref.ref(tensor))
        return tensor

    def build_fused():
        gc.collect()
        assert early_release == [True], "loader accumulated the checkpoint"
        assert all(ref() is None for ref in refs), "inputs retained at fused setup"
        for index in range(8):
            torch.testing.assert_close(
                getattr(draft.model, f"weight{index}"), torch.full((2,), index + 1.0)
            )
        if use_aux:
            torch.testing.assert_close(draft.model.fc.weight, torch.full((2, 2), 9.0))
        torch.testing.assert_close(
            draft.model.embed_tokens.weight,
            torch.full((2, 2), 10.0) if own_weights else embed_before,
        )
        torch.testing.assert_close(
            draft.lm_head.weight,
            torch.full((2, 2), 11.0) if own_weights else head_before,
        )
        torch.testing.assert_close(draft.draft_id_to_target_id, torch.full((2,), 12.0))
        torch.testing.assert_close(
            draft.model.mask_embedding, torch.full((2,), 14.0 if with_mask else 0.0)
        )
        assert draft.has_own_lm_head == own_weights
        assert draft.has_own_embed_tokens == own_weights
        assert draft.model.has_separate_mask_embedding == with_mask

    draft._read_mask_embedding = read_mask
    draft.model._build_fused_kv_buffers = build_fused
    DFlashQwen3ForCausalLM.load_weights(draft, weights())


def test_mrv1_private_ring_slots_survive_unscheduled_steps_and_release_on_preemption():
    from vllm.v1.spec_decode.dflash import DFlashProposer

    owner = object.__new__(DFlashProposer)
    owner._ring_block_table = torch.zeros(3, 8, dtype=torch.int32)
    owner.max_batch_size = 3
    owner._ring_request_slots = {}

    def update(ids, finished=(), preempted=()):
        owner.update_request_state(
            SimpleNamespace(
                num_scheduled_tokens=dict.fromkeys(ids, 1),
                finished_req_ids=set(finished),
                preempted_req_ids=set(preempted),
            )
        )

    update(["a", "b"])
    assert owner._ring_request_slots == {"a": 0, "b": 1}
    update(["b"])
    assert owner._ring_request_slots == {"a": 0, "b": 1}
    update(["b", "a"])
    assert owner._ring_request_slots == {"a": 0, "b": 1}
    update(["b", "c"], preempted=["a"])
    assert owner._ring_request_slots == {"b": 1, "c": 0}
    update(["c", "a"], finished=["b"])
    assert owner._ring_request_slots == {"c": 0, "a": 1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mrv1_private_ring_table_tracks_request_identity_and_rejected_tokens():
    from vllm.v1.spec_decode.dflash import DFlashProposer

    owner = object.__new__(DFlashProposer)
    owner.device = torch.device("cuda")
    owner._ring_block_table = torch.zeros(3, 8, dtype=torch.int32, device="cuda")
    owner._ring_base, owner._ring_size = 1, 3
    owner.block_size, owner.num_speculative_tokens = 4, 2
    owner._ring_request_slots = {"a": 0, "b": 1}
    owner._target_runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["b", "a"])
    )
    from vllm.v1.attention.backend import CommonAttentionMetadata

    cad = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([16, 10], dtype=torch.int32, device="cuda"),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=16,
        block_table_tensor=torch.zeros(2, 8, dtype=torch.int32, device="cuda"),
        slot_mapping=torch.zeros(2, dtype=torch.int64, device="cuda"),
    )
    mapped = owner._prepare_ring_block_table(
        cad, torch.tensor([2, 1], dtype=torch.int32, device="cuda")
    )
    assert mapped.block_table_tensor.tolist() == [
        [4, 5, 6, 4, 5, 0, 0, 0],
        [1, 2, 3, 0, 0, 0, 0, 0],
    ]
    assert mapped.block_table_tensor.data_ptr() == owner._ring_block_table.data_ptr()
    assert cad.block_table_tensor.count_nonzero().item() == 0


@pytest.mark.parametrize("private_ring", [False, True])
def test_mrv1_private_ring_initialization_uses_allocated_manager_pages(
    monkeypatch, private_ring
):
    from vllm.v1.kv_cache_interface import (
        DFlashSWASpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )
    from vllm.v1.spec_decode.dflash import DFlashProposer
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    spec = DFlashSWASpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        sliding_window=8,
        private_ring=private_ring,
    )
    owner = object.__new__(DFlashProposer)
    owner.device = torch.device("cpu")
    owner._target_runner = object()
    owner._ring_block_table = None
    owner.max_batch_size = 3
    owner.max_model_len = 32
    owner.num_speculative_tokens = 2
    owner.parallel_drafting_token_id = 99
    owner.kv_cache_gid = 0
    owner.block_size = 4
    monkeypatch.setattr(
        SpecDecodeBaseProposer, "initialize_attn_backend", lambda *args: None
    )
    config = KVCacheConfig(
        num_blocks=100,
        kv_cache_groups=[KVCacheGroupSpec(["draft"], spec)],
        kv_cache_tensors=[
            KVCacheTensor(
                size=10 * spec.page_size_bytes,
                layers=["draft"],
                layer_stride=10 * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
                num_blocks=10,
            )
        ],
    )
    owner.initialize_attn_backend(config)
    if private_ring:
        assert owner._ring_size == 3
        assert owner._ring_block_table is not None
        assert owner._ring_block_table.shape == (3, 9)
    else:
        assert owner._ring_block_table is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mrv1_private_ring_first_pass_maps_context_and_query_slots():
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.spec_decode.dflash import DFlashProposer

    owner = object.__new__(DFlashProposer)
    owner.device = torch.device("cuda")
    owner._ring_block_table = torch.zeros(3, 8, dtype=torch.int32, device="cuda")
    owner._ring_base, owner._ring_size = 1, 3
    owner.block_size, owner.num_speculative_tokens = 4, 2
    owner.parallel_drafting_token_id = 99
    owner.dflash_causal = True
    owner._ring_request_slots = {"a": 0, "b": 1}
    owner._target_runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["b", "a"])
    )
    owner._context_positions_buffer = torch.zeros(4, dtype=torch.int64, device="cuda")
    owner._context_slot_mapping_buffer = torch.zeros(
        4, dtype=torch.int64, device="cuda"
    )
    owner._slot_mapping_buffer = torch.zeros(6, dtype=torch.int64, device="cuda")
    owner.positions = torch.zeros(6, dtype=torch.int64, device="cuda")
    owner.input_ids = torch.zeros(6, dtype=torch.int32, device="cuda")
    owner.arange = torch.arange(9, dtype=torch.int32, device="cuda")
    owner.token_arange_np = torch.arange(9, dtype=torch.int32).numpy()
    cad = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda"),
        query_start_loc_cpu=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([16, 10], dtype=torch.int32, device="cuda"),
        num_reqs=2,
        num_actual_tokens=4,
        max_query_len=2,
        max_seq_len=16,
        block_table_tensor=torch.zeros(2, 8, dtype=torch.int32, device="cuda"),
        slot_mapping=torch.zeros(4, dtype=torch.int64, device="cuda"),
    )
    count, sample_indices, prepared = owner.set_inputs_first_pass(
        target_token_ids=torch.arange(4, dtype=torch.int32, device="cuda"),
        next_token_ids=torch.tensor([101, 202], dtype=torch.int32, device="cuda"),
        target_positions=torch.tensor([14, 15, 8, 9], dtype=torch.int64, device="cuda"),
        target_hidden_states=torch.zeros(4, 1, device="cuda"),
        token_indices_to_sample=None,
        cad=cad,
        num_rejected_tokens_gpu=torch.tensor([1, 1], dtype=torch.int32, device="cuda"),
    )
    assert count == 6
    assert owner.input_ids.tolist() == [101, 99, 99, 202, 99, 99]
    assert owner.positions.tolist() == [15, 16, 17, 9, 10, 11]
    assert owner._context_slot_mapping_buffer.tolist() == [18, 19, 12, 13]
    assert prepared.slot_mapping.tolist() == [19, 20, 21, 13, 14, 15]
    assert sample_indices.tolist() == [1, 2, 4, 5]
    assert prepared.seq_lens.tolist() == [18, 12]
    assert prepared.block_table_tensor.data_ptr() == owner._ring_block_table.data_ptr()


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("soft_cap", [None, 2.0, 4.0])
@pytest.mark.parametrize("precomputed", [False, True])
@pytest.mark.parametrize("return_log_probs", [False, True])
def test_candidate_topk_matches_full_logit_transforms(
    scale, soft_cap, precomputed, return_log_probs
):
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    processor = object.__new__(LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.scale = scale
    processor.soft_cap = soft_cap
    processor.head_dtype = None
    processor.logits_as_input = True
    logits = torch.tensor(
        [[-1.5, 0.75, 1.5, 3.0, 100.0], [2.25, -0.5, 1.25, 0.0, 100.0]]
    )

    class QuantMethod:
        def apply(self, lm_head, hidden_states, bias=None):
            return logits.clone()

    lm_head = SimpleNamespace(
        tp_size=1,
        shard_indices=SimpleNamespace(num_org_vocab_padding=1, org_vocab_start_index=0),
        quant_method=QuantMethod(),
    )
    # Compare distinct production paths, not a copy of the compact formula.
    full = processor.forward(lm_head, logits[:, :-1].clone())
    assert full is not None
    if return_log_probs:
        full = full.log_softmax(dim=-1)
    expected_values, expected_ids = full.topk(3, dim=-1)
    ids, values = processor.get_top_k_tokens(
        lm_head,
        torch.empty(2, 1),
        3,
        return_log_probs=return_log_probs,
        local_logits=logits.clone() if precomputed else None,
    )
    torch.testing.assert_close(ids, expected_ids)
    torch.testing.assert_close(values, expected_values)
