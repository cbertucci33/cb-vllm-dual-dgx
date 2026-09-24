# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.qwen3_dflash2 import (
    CandidateSelector,
    DFlashGroupedConv,
    _grouped_conv,
    _score_edges,
)
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator


def test_dflash_attention_config_uses_draft_cache_dtype():
    import copy

    from vllm.config import VllmConfig, replace

    target_config = VllmConfig()
    target_config.cache_config.kv_cache_layout = "LBHNC"
    draft_config = copy.copy(target_config)
    draft_config.attention_config = replace(
        target_config.attention_config,
        backend="FLASHINFER",
        use_non_causal=True,
    )
    draft_config.cache_config = replace(
        target_config.cache_config, cache_dtype="fp8_e4m3"
    )
    assert draft_config.cache_config.kv_cache_layout is None

    speculator = object.__new__(DFlashSpeculator)
    speculator.vllm_config = target_config
    speculator._draft_vllm_config = draft_config

    config = speculator.attn_vllm_config

    assert config is draft_config
    assert config.cache_config.cache_dtype == "fp8_e4m3"
    assert config.attention_config.backend.name == "FLASHINFER"
    assert config.attention_config.use_non_causal is True
    assert config.cache_config.kv_cache_layout == "LBHNC"
    assert config.cache_config is not target_config.cache_config


def test_dflash_load_retains_exact_draft_config(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as module

    target_config = object()
    draft_config = object()
    draft_model = object()
    speculator = object.__new__(DFlashSpeculator)
    speculator.vllm_config = target_config
    monkeypatch.setattr(
        module,
        "load_dflash_model",
        lambda target_model, vllm_config: (draft_model, draft_config),
    )

    assert speculator.load_draft_model(object(), set()) is draft_model
    assert speculator._draft_vllm_config is draft_config


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_ring_synthesis_covers_context_and_draft_queries():
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        synthesize_draft_ring_block_tables,
    )

    ring_size = 4
    block_table = torch.tensor(
        [[0, 0, 17, 23, 9, 41, 0], [5, 6, 7, 0, 0, 0, 0]],
        dtype=torch.int32,
        device="cuda",
    )
    idx_mapping = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([20, 12], dtype=torch.int32, device="cuda")
    synthesize_draft_ring_block_tables(
        block_table,
        idx_mapping,
        seq_lens,
        block_size=4,
        ring_size=ring_size,
        num_query_per_req=5,
    )

    base0, base1 = 1 + 3 * ring_size, 1 + ring_size
    expected = torch.tensor(
        [
            [base0, base0 + 1, base0 + 2, base0 + 3, base0, base0 + 1, base0 + 2],
            [base1, base1 + 1, base1 + 2, base1 + 3, base1, 0, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    assert torch.equal(block_table, expected)


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


def test_dflash2_projection_layers_receive_draft_quant_config(
    monkeypatch, default_vllm_config
):
    """MXFP8 scale tensors need quant-aware projection owners."""
    from vllm.model_executor.models import qwen3_dflash2

    captured = []

    class FakeReplicatedLinear(torch.nn.Module):
        def __init__(self, *args, quant_config=None, **kwargs):
            super().__init__()
            captured.append(quant_config)

    monkeypatch.setattr(qwen3_dflash2, "ReplicatedLinear", FakeReplicatedLinear)
    from vllm.config import set_current_vllm_config

    quant_config = object()
    with set_current_vllm_config(default_vllm_config), torch.device("meta"):
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


def test_mxfp8_context_kv_rows_are_dequantized_for_plain_linear():
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

    model = object.__new__(DFlashQwen3Model)
    torch.nn.Module.__init__(model)
    model.hidden_norm = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)

    source = torch.linspace(-3, 3, 6 * 32, dtype=torch.bfloat16).view(6, 32)
    weight = source.to(torch.float8_e4m3fn)
    scales = torch.full((6, 1), 127, dtype=torch.uint8)
    attn = SimpleNamespace(
        q_size=2,
        qkv_proj=SimpleNamespace(weight=weight, weight_scale=scales),
    )

    rows = model._kv_projection_rows(attn)

    assert rows.dtype is torch.bfloat16
    torch.testing.assert_close(rows, weight[2:].to(torch.bfloat16))


def test_candidate_topk_accepts_precomputed_local_logits():
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    processor = object.__new__(LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.scale = 2.0
    processor.soft_cap = 2.0

    shard_indices = SimpleNamespace(
        num_org_vocab_padding=1,
        num_org_elements=3,
        num_org_elements_padded=4,
        num_added_vocab_padding=1,
        num_added_elements=1,
        num_added_elements_padded=2,
        added_vocab_start_index=100,
        org_vocab_start_index=10,
    )
    lm_head = SimpleNamespace(tp_size=1, shard_indices=shard_indices)
    local_logits = torch.tensor([[1.0, 3.0, 2.0, 100.0, 4.0, 99.0]])

    ids, values = processor.get_top_k_tokens(
        lm_head,
        hidden_states=torch.empty(1, 0),
        k=2,
        local_logits=local_logits,
    )

    assert ids.tolist() == [[100, 11]]
    expected = torch.tanh(torch.tensor([[4.0, 3.0]]) / 2.0) * 4.0
    torch.testing.assert_close(values, expected)


def test_selector_walk_uses_disjoint_draft_gumbel_positions(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode import utils
    from vllm.v1.worker.gpu.spec_decode.dflash2 import speculator as module

    captured = {}

    class FakeKernel:
        def __getitem__(self, grid):
            captured["grid"] = grid

            def launch(*args, **kwargs):
                captured.update(kwargs)

            return launch

    monkeypatch.setattr(module, "_selector_walk_kernel", FakeKernel())
    speculator = object.__new__(DFlash2Speculator)
    speculator.selector_top_k = 4
    speculator.num_speculative_steps = 3
    speculator.draft_logits = None
    speculator.use_fp64_gumbel = False
    speculator.sample_pos = torch.empty(3, dtype=torch.int64)
    speculator.sample_idx_mapping = torch.empty(3, dtype=torch.int64)
    speculator.temperature = torch.empty(1)
    speculator.seeds = torch.empty(1, dtype=torch.int64)
    speculator.draft_tokens = torch.empty((1, 3), dtype=torch.int64)
    speculator._selector_scores = torch.empty((1, 3, 4))

    speculator._sample_path(torch.empty((1, 3, 4)), torch.empty((1, 3, 4, 4)), 1)

    assert captured["POS_OFFSET"] == utils.DRAFT_GUMBEL_POS_OFFSET == 1 << 30


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

    torch.testing.assert_close(actual, expected, atol=1.0, rtol=0.08)


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
def test_dflash2_model_decoder_layer_cls(monkeypatch):
    from types import SimpleNamespace

    from vllm.config import set_current_vllm_config
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
            "conv_kernel_size": 3,
            "conv_group_size": 2,
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
        model = DFlash2Qwen3Model(vllm_config=vllm_config)

    # 4. Assert that the layers are DFlash2Qwen3DecoderLayer (the subclass)
    assert len(model.layers) == 2
    assert isinstance(model.layers[0], DFlash2Qwen3DecoderLayer)
