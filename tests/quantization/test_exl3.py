# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.quantization.exl3 as exl3_module
import vllm.model_executor.parameter as parameter_module
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.exl3 import (
    Exl3Config,
    Exl3LinearMethod,
    Exl3MoEMethod,
    Exl3MoEParameter,
    Exl3Parameter,
)
from vllm.model_executor.models.glm4_moe_mtp import Glm4MoeMTP
from vllm.models.glm5next.common.mtp import Glm5NextMTP


@pytest.fixture(autouse=True)
def parameter_tp_env(monkeypatch):
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )


def _rank_sliced_metadata(**overrides):
    metadata = {
        "format": "exl3-trellis",
        "bits": 3.0,
        "codebook": "mcg",
        "experts_per_layer": 256,
        "moe_layers": [3, 77],
        "tensor_schema": (
            "model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}"
        ),
        "tp": 4,
    }
    metadata.update(overrides)
    return metadata


def _dense_storage():
    def entry(prefix):
        return {
            "quant_format": "exl3",
            "stored_tensors": {
                f"{prefix}.trellis": {},
                f"{prefix}.suh": {},
                f"{prefix}.svh": {},
                f"{prefix}.mcg": {},
            },
        }

    return {
        "lm_head": entry("lm_head"),
        "model.layers.0.self_attn.o_proj": entry("model.layers.0.self_attn.o_proj"),
    }


def test_rank_sliced_checkpoint_selects_exl3_override():
    hf_config = SimpleNamespace(hybrid_tr3_tail=_rank_sliced_metadata())

    assert get_quantization_config("exl3") is Exl3Config
    assert (
        Exl3Config.override_quantization_method(
            {"quant_method": "modelopt"}, None, hf_config
        )
        == "exl3"
    )
    assert (
        Exl3Config.override_quantization_method(
            {"quant_method": "modelopt"}, "fp8", hf_config
        )
        is None
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"codebook": "mul1"}, "MCG codebook"),
        ({"moe_layers": [77, 3]}, "moe_layers"),
        ({"tensor_schema": "unsupported"}, "tensor schema"),
    ],
)
def test_rank_sliced_metadata_fails_closed(overrides, message):
    config = Exl3Config()
    hf_config = SimpleNamespace(hybrid_tr3_tail=_rank_sliced_metadata(**overrides))

    with pytest.raises(ValueError, match=message):
        config.maybe_update_config("unused", hf_config)


def test_rank_sliced_metadata_admits_only_declared_moe_layers(tmp_path):
    config = Exl3Config(tensor_storage=_dense_storage())
    config.maybe_update_config(
        str(tmp_path),
        SimpleNamespace(hybrid_tr3_tail=_rank_sliced_metadata()),
    )

    assert config._moe_prefix_is_exl3("model.layers.3.mlp.experts")
    assert config._moe_prefix_is_exl3("model.layers.77.mlp.experts")
    assert not config._moe_prefix_is_exl3("model.layers.2.mlp.experts")
    assert not config._moe_prefix_is_exl3("model.layers.78.mlp.experts")
    assert (
        config.codebook_for_prefix("model.layers.10.mlp.experts.0.gate_proj") == "mcg"
    )
    assert config.codebook_for_prefix("model.layers.0.self_attn.o_proj") == "mcg"


def test_rank_sliced_checkpoint_hydrates_dense_exl3_metadata(monkeypatch):
    storage = _dense_storage()
    monkeypatch.setattr(
        exl3_module,
        "get_hf_file_to_dict",
        lambda filename, *args, **kwargs: (
            {"weight_map": {}}
            if filename == "model.safetensors.index.json"
            else {"tensor_storage": storage}
        ),
    )
    config = Exl3Config()

    config.maybe_update_config(
        "checkpoint",
        SimpleNamespace(
            hybrid_tr3_tail=_rank_sliced_metadata(),
            tie_word_embeddings=False,
        ),
    )

    assert config.tensor_storage == storage
    assert config._linear_prefix_is_exl3("language_model.lm_head")


@pytest.mark.parametrize("quant_file", [None, {}, {"tensor_storage": {}}])
def test_expert_only_rank_sliced_metadata_does_not_require_dense_quant_file(
    monkeypatch, quant_file
):
    monkeypatch.setattr(
        exl3_module,
        "get_hf_file_to_dict",
        lambda filename, *a, **kw: (
            {"weight_map": {}}
            if filename == "model.safetensors.index.json"
            else quant_file
        ),
    )
    config = Exl3Config(tensor_storage={})

    config.maybe_update_config(
        "checkpoint",
        SimpleNamespace(hybrid_tr3_tail=_rank_sliced_metadata()),
    )

    assert config.rank_sliced_metadata == _rank_sliced_metadata()
    assert config.tensor_storage == {}


def test_dense_exl3_quantization_does_not_force_eager_execution(monkeypatch):
    config = Exl3Config(tensor_storage=_dense_storage())
    config.rank_sliced_metadata = _rank_sliced_metadata()
    monkeypatch.setattr(
        exl3_module,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False)),
    )
    layer = object.__new__(exl3_module.LinearBase)
    torch.nn.Module.__init__(layer)
    assert isinstance(
        config.get_quant_method(layer, "model.layers.0.self_attn.o_proj"),
        Exl3LinearMethod,
    )


def test_dense_exl3_rejects_unwarmed_cuda_execution():
    from torch._subclasses.fake_tensor import FakeTensorMode

    method = Exl3LinearMethod(Exl3Config())
    layer = torch.nn.Module()
    layer.exl3_autotune_warmed = False
    with FakeTensorMode():
        x = torch.empty((16, 128), device="cuda:0", dtype=torch.float16)
        with pytest.raises(RuntimeError, match="finish autotune warmup"):
            method.apply(layer, x)


def test_rank_sliced_weight_name_keeps_only_local_tp_rank(monkeypatch, tmp_path):
    config = Exl3Config(tensor_storage=_dense_storage())
    config.maybe_update_config(
        str(tmp_path),
        SimpleNamespace(hybrid_tr3_tail=_rank_sliced_metadata()),
    )
    monkeypatch.setattr(exl3_module, "get_tensor_model_parallel_rank", lambda: 2)
    prefix = "model.layers.3.mlp.experts.17.gate_proj"

    assert (
        config.normalize_rank_sliced_weight_name(f"{prefix}.rank2.trellis")
        == f"{prefix}.trellis"
    )
    assert config.normalize_rank_sliced_weight_name(f"{prefix}.rank1.trellis") is None
    assert (
        config.normalize_rank_sliced_weight_name("model.embed_tokens.weight")
        == "model.embed_tokens.weight"
    )


def test_rank_sliced_dense_parameter_copies_mapped_weight():
    source = torch.arange(8, dtype=torch.float16)
    param = Exl3Parameter(weight_loader=lambda *_: None)
    param.exl3_copy_on_load = True

    param.load_exl3_weight(source)

    assert param.exl3_tensors[None].data_ptr() != source.data_ptr()
    assert torch.equal(param.exl3_tensors[None], source)


def test_asymmetric_qkv_v_trims_padding_before_tp_shard_selection():
    layer = object.__new__(QKVParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.v_head_size = 64
    layer.head_size = 128
    layer.total_num_kv_heads = 2
    layer.num_kv_heads = 1
    layer.num_kv_head_replicas = 1
    layer.exl3_tp_rank = 1
    layer.svh = SimpleNamespace(
        exl3_tensors={"v": torch.empty(2 * 128, dtype=torch.float16)}
    )

    assert Exl3LinearMethod._is_padded_qkv_value_shard(layer, "v")
    padded = torch.arange(2 * 128, dtype=torch.float16).reshape(1, -1)

    trimmed = Exl3LinearMethod._trim_and_shard_padded_qkv_value(layer, padded)

    assert tuple(trimmed.shape) == (1, 64)
    expected = torch.arange(128, 192, dtype=torch.float16).view(1, -1)
    torch.testing.assert_close(trimmed, expected)


def test_rank_sliced_parameter_preallocates_projection_major_slab(monkeypatch):
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    param = Exl3MoEParameter(
        weight_loader=lambda *args, **kwargs: None,
        num_experts=3,
        shard_ids=("w1", "w3"),
        preallocate=True,
    )
    w1 = torch.arange(8, dtype=torch.int16).reshape(2, 2, 2)
    w3 = w1 + 20

    param.load_exl3_weight(w1, expert_id=1, shard_id="w1")
    param.load_exl3_weight(w3, expert_id=2, shard_id="w3")

    assert param.exl3_backing is not None
    assert tuple(param.exl3_backing.shape) == (2, 3, 2, 2, 2)
    assert (
        param.exl3_tensors[(1, "w1")].data_ptr() == param.exl3_backing[0, 1].data_ptr()
    )
    assert (
        param.exl3_tensors[(2, "w3")].data_ptr() == param.exl3_backing[1, 2].data_ptr()
    )
    torch.testing.assert_close(param.exl3_tensors[(1, "w1")], w1)
    torch.testing.assert_close(param.exl3_tensors[(2, "w3")], w3)


def test_moe_scalar_marker_does_not_retain_source_storage():
    source = torch.tensor([0xCBAC1FED - (1 << 32)], dtype=torch.int32)
    param = Exl3MoEParameter(weight_loader=lambda *_: None)

    param.load_exl3_weight(source, expert_id=0, shard_id="mcg")

    stored = param.exl3_tensors[(0, "mcg")]
    assert stored.data_ptr() != source.data_ptr()
    assert torch.equal(stored, source)


def _make_rank_sliced_mtp_loader(mtp_class):
    model = object.__new__(mtp_class)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_hidden_layers=10,
        num_nextn_predict_layers=1,
        n_routed_experts=1,
        n_shared_experts=0,
        mla_use_nope=False,
        qk_rope_head_dim=0,
    )
    model.quant_config = SimpleNamespace(
        normalize_rank_sliced_weight_name=lambda name: (
            None if ".rank1." in name else name.replace(".rank0.", ".")
        )
    )
    model.num_redundant_experts = 0
    if mtp_class is Glm4MoeMTP:
        model.is_fused_shared_expert_enabled = False

    model.model = torch.nn.Module()
    model.model.mtp_start_layer_idx = 10
    model.model.num_mtp_layers = 1
    model.model.layers = torch.nn.ModuleDict()
    layer = torch.nn.Module()
    layer.mtp_block = torch.nn.Module()
    layer.mtp_block.mlp = torch.nn.Module()
    layer.mtp_block.mlp.experts = torch.nn.Module()
    layer.mtp_block.mlp.experts.routed_experts = torch.nn.Module()
    param = Exl3MoEParameter(
        weight_loader=exl3_module._exl3_moe_weight_loader,
        num_experts=1,
        shard_ids=("w1",),
    )
    layer.mtp_block.mlp.experts.routed_experts.register_parameter("w13_trellis", param)
    model.model.layers["10"] = layer
    return model, param


@pytest.mark.parametrize("mtp_class", [Glm5NextMTP, Glm4MoeMTP])
def test_rank_sliced_mtp_loaders_send_only_local_normalized_expert_name(mtp_class):
    model, param = _make_rank_sliced_mtp_loader(mtp_class)
    local = torch.full((2, 2, 48), 3, dtype=torch.int16)
    remote = torch.full((2, 2, 48), 9, dtype=torch.int16)
    expert_loader = param.weight_loader
    loader_names = []

    def record_expert_loader(*args, **kwargs):
        loader_names.append(args[2])
        return expert_loader(*args, **kwargs)

    param.weight_loader = record_expert_loader
    prefix = "model.layers.10.mlp.experts.0.gate_proj"

    loaded = model.load_weights(
        [
            (f"{prefix}.rank0.trellis", local),
            (f"{prefix}.rank1.trellis", remote),
        ]
    )

    expected_name = "model.layers.10.mtp_block.mlp.experts.routed_experts.w13_trellis"
    assert loader_names == [expected_name]
    assert ".rank" not in loader_names[0]
    assert loaded == {expected_name}
    assert torch.equal(param.exl3_tensors[(0, "w1")], local)


def test_rank_sliced_weights_use_the_planned_sparkinfer_contract(monkeypatch):
    experts = 2
    hidden = intermediate = 128
    bits = 3
    slabs = {
        "w13_trellis": torch.zeros(
            (2, experts, hidden // 16, intermediate // 16, 16 * bits),
            dtype=torch.int16,
        ),
        "w2_trellis": torch.zeros(
            (experts, intermediate // 16, hidden // 16, 16 * bits),
            dtype=torch.int16,
        ),
        "w13_suh": torch.ones((2, experts, hidden), dtype=torch.float16),
        "w13_svh": torch.ones((2, experts, intermediate), dtype=torch.float16),
        "w2_suh": torch.ones((experts, intermediate), dtype=torch.float16),
        "w2_svh": torch.ones((experts, hidden), dtype=torch.float16),
    }

    class FakeTrellis:
        def __init__(self):
            self.args = None
            self.kwargs = None

        def prepare_weights(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            return object()

    api = FakeTrellis()
    monkeypatch.setattr(exl3_module, "_load_sparkinfer_trellis", lambda: api)
    method = object.__new__(Exl3MoEMethod)
    method.quant_config = SimpleNamespace(bits=float(bits))
    monkeypatch.setattr(
        method, "_rank_sliced_backing", lambda _layer, name: slabs[name]
    )
    marker = torch.tensor(0xCBAC1FED - (1 << 32), dtype=torch.int32)
    layer = SimpleNamespace(
        local_num_experts=experts,
        exl3_hidden_size=hidden,
        exl3_intermediate_size_per_partition=intermediate,
        w13_mcg=SimpleNamespace(exl3_tensors={(0, "w1"): marker}),
    )

    method._prepare_rank_sliced_weights(layer)

    assert api.args == (slabs["w13_trellis"], slabs["w2_trellis"])
    assert api.kwargs is not None
    assert set(api.kwargs) == {
        "gate_suh",
        "up_suh",
        "intermediate_rotations",
        "down_svh",
        "codebook",
        "mcg",
        "tile_config",
    }
    assert api.kwargs["gate_suh"].data_ptr() == slabs["w13_suh"][0].data_ptr()
    assert api.kwargs["up_suh"].data_ptr() == slabs["w13_suh"][1].data_ptr()
    assert api.kwargs["down_svh"].data_ptr() == slabs["w2_svh"].data_ptr()
    assert api.kwargs["mcg"] is marker
    assert api.kwargs["codebook"] == "mcg"
    assert api.kwargs["tile_config"] == (64, 128, 64, 128)


def test_rank_sliced_runtime_scope_is_per_owning_model():
    """Target and rank-sliced MTP draft layers must not share a cached runtime.

    The rank-sliced runtime cache stores mutable Trellis and prefill scratch.
    A target MoE layer and an MTP draft layer of the same
    model have identical shapes, topk and planner settings, so a shape-only key
    would hand the draft the target's scratch and break the target/draft
    isolation their independently captured CUDA graphs depend on.
    """
    target_config = SimpleNamespace()
    draft_config = SimpleNamespace()

    target_scope = exl3_module._runtime_scope_id(target_config)
    draft_scope = exl3_module._runtime_scope_id(draft_config)

    # Distinct owning configs must never collide...
    assert target_scope != draft_scope
    # ...and the scope must be stable, so every layer of one model keeps sharing
    # a single runtime (the prefill arena is ~1 GiB; per-layer runtimes would not
    # fit on a 75+ layer model).
    assert exl3_module._runtime_scope_id(target_config) == target_scope
    assert exl3_module._runtime_scope_id(draft_config) == draft_scope


def test_rank_sliced_runtime_key_differs_across_models_with_same_shape():
    """Two same-shape layers owned by different models get different cache keys."""

    def _key(quant_config):
        # Mirrors the scope-prefixed key built in Exl3MoEMethod._rank_sliced_runtime
        # for two layers whose shape/planner components are byte-for-byte equal.
        return (
            exl3_module._runtime_scope_id(quant_config),
            0,  # device index
            torch.bfloat16,
            5120,  # hidden size
            768,  # intermediate size per partition
            64,  # local experts
            8,  # topk
            3072,  # max batched tokens
        )

    target_config = SimpleNamespace()
    draft_config = SimpleNamespace()

    target_key = _key(target_config)
    draft_key = _key(draft_config)

    assert target_key != draft_key
    # Everything except the leading scope is identical, proving the scope is the
    # only thing preventing the collision.
    assert target_key[1:] == draft_key[1:]
    # Same owner -> same key, so target layers still share one runtime.
    assert _key(target_config) == target_key


def test_rank_sliced_window_defaults_to_min_capturable_m(monkeypatch) -> None:
    """Every rank-sliced layer must cover small rows without an env workaround.

    Regression test for the boot failure reported in vLLM #183: with the Trellis
    window left at its historical default of 4, CUDA-graph capture of an EXL3
    rank-sliced MTP draft reaches the eager parity path at m=1,2,3 and the engine
    cannot start:

        RuntimeError: EXL3 eager parity path entered during CUDA graph capture
        (m=3); capture sizes must lie inside the Trellis window [4, 32]

    Target profiling can also produce m=2, so role-dependent defaults merely
    move the same failure from the draft to the target. The backend declares one
    capability floor and uses it for both roles.
    """
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization import exl3 as exl3_mod

    monkeypatch.delenv("VLLM_EXL3_TRELLIS_MIN_M", raising=False)

    # The GLM-5.2 MTP head is named exactly like a target layer, so the role
    # comes from the exl3_is_draft stamp applied by load_eagle_model -- name
    # inspection alone cannot classify it.
    draft = SimpleNamespace(
        layer_name="model.layers.78.mlp.experts", exl3_is_draft=True
    )
    target = SimpleNamespace(layer_name="model.layers.30.mlp.experts")

    assert exl3_mod._is_draft_layer(draft)
    assert not exl3_mod._is_draft_layer(target)
    # Unstamped draft with a distinctive prefix still classifies via fallback.
    assert exl3_mod._is_draft_layer(
        SimpleNamespace(layer_name="model.layers.0.mtp.mlp.experts")
    )
    # A stamp always wins over the name, in both directions.
    assert not exl3_mod._is_draft_layer(
        SimpleNamespace(layer_name="model.layers.0.mtp.experts", exl3_is_draft=False)
    )

    def resolved():
        return exl3_mod._positive_env_int(
            "VLLM_EXL3_TRELLIS_MIN_M", exl3_mod._DEFAULT_TRELLIS_MIN_M
        )

    assert resolved() == exl3_mod._DEFAULT_TRELLIS_MIN_M
    assert resolved() == exl3_mod.MIN_CAPTURABLE_TRELLIS_M == 1

    # Rank-sliced execution has no eager parity fallback. A higher floor must
    # therefore be rejected before runtime planning.
    monkeypatch.setenv("VLLM_EXL3_TRELLIS_MIN_M", "4")
    assert resolved() == 4


def test_draft_role_stamp_wins_over_name() -> None:
    """The exl3_is_draft stamp set in create_weights is authoritative.

    Forward/plan/capture time has no current vllm config, so the role cannot be
    inferred there; create_weights stamps it from runner_type while the
    construction context is live. Stamped values must win over any name
    heuristic in both directions.
    """
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization import exl3 as exl3_mod

    # GLM-5.2 MTP head: named like a target, stamped draft.
    assert exl3_mod._is_draft_layer(
        SimpleNamespace(layer_name="model.layers.78.mlp.experts", exl3_is_draft=True)
    )
    # Target stamped False keeps its role even with a suspicious name.
    assert not exl3_mod._is_draft_layer(
        SimpleNamespace(layer_name="model.layers.0.mtp.experts", exl3_is_draft=False)
    )
    # Unstamped layers fall back to the name heuristic.
    assert exl3_mod._is_draft_layer(
        SimpleNamespace(layer_name="model.layers.0.mtp.mlp.experts")
    )
    assert not exl3_mod._is_draft_layer(
        SimpleNamespace(layer_name="model.layers.30.mlp.experts")
    )


@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3"])
def test_calibrated_exl3_kv_scales_survive_loading_and_attention_postload(
    tmp_path, cache_dtype
):
    """Guard the 48-layer calibration against missing parameters and unit reset.

    Initialize only the quantization portion of real Attention layers; this is
    a CPU scale-loading contract, not backend/model execution acceptance.
    """
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.attention.attention import _init_kv_cache_quant
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.model_executor.models.utils import AutoWeightsLoader
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    values = {
        f"model.layers.{i}.self_attn.{scale}_scale": torch.tensor(
            (i + 1) * (0.0005 if scale == "k" else 0.001), dtype=torch.float32
        )
        for i in range(48)
        for scale in ("k", "v")
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(values, "scales.safetensors")})
    )
    config = Exl3Config(tensor_storage=_dense_storage())
    config.maybe_update_config(str(tmp_path))

    model = torch.nn.Module()
    model.quant_config = config
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList()
    for i in range(48):
        decoder = torch.nn.Module()
        decoder.self_attn = torch.nn.Module()
        layer = Attention.__new__(Attention)
        torch.nn.Module.__init__(layer)
        layer.kv_cache_dtype = cache_dtype
        layer.require_kv_cache_scales = False
        layer.quant_config = config
        layer.layer_name = f"model.layers.{i}.self_attn.attn"
        # Use the real inherited no-op backend post-load hook. No kernels run.
        layer.impl = TritonAttentionImpl.__new__(TritonAttentionImpl)
        _init_kv_cache_quant(layer, config, layer.layer_name)
        decoder.self_attn.attn = layer
        model.model.layers.append(decoder)

    loaded = AutoWeightsLoader(model).load_weights(values.items())
    assert len(loaded) == 96
    process_weights_after_loading(
        model,
        SimpleNamespace(
            dtype=torch.bfloat16,
            quantization="exl3",
            word_embeddings_untied_by_checkpoint=False,
        ),
        torch.device("cpu"),
    )
    for i, decoder in enumerate(model.model.layers):
        layer = decoder.self_attn.attn
        for scale in ("k", "v"):
            expected = values[f"model.layers.{i}.self_attn.{scale}_scale"]
            torch.testing.assert_close(getattr(layer, f"_{scale}_scale"), expected)
            torch.testing.assert_close(getattr(layer, f"_{scale}_scale_cpu"), expected)
            assert getattr(layer, f"_{scale}_scale_float") == expected.item()
            assert not hasattr(layer, f"{scale}_scale")
        assert layer._q_scale.item() == 1.0
        assert layer._prob_scale.item() == 1.0


def test_exl3_without_checkpoint_kv_calibration_preserves_attention_dispatch(tmp_path):
    from vllm.model_executor.layers.attention import Attention

    config = Exl3Config(tensor_storage=_dense_storage())
    config.maybe_update_config(str(tmp_path))
    layer = Attention.__new__(Attention)
    torch.nn.Module.__init__(layer)

    assert config.get_quant_method(layer, "model.layers.0.self_attn.attn") is None
    assert config._kv_cache_scale_names == set()


def test_exl3_kv_scale_index_is_read_once_and_preserves_model_mapper(
    monkeypatch, tmp_path
):
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
    from vllm.model_executor.models.utils import WeightsMapper

    weight_map = {
        "model.layers.0.self_attn.k_scale": "scales.safetensors",
        "model.layers.0.self_attn.v_scale": "scales.safetensors",
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    config = Exl3Config(tensor_storage=_dense_storage())
    config.maybe_update_config(str(tmp_path))
    config.apply_vllm_mapper(
        WeightsMapper(orig_to_new_prefix={"model.": "language_model."})
    )
    monkeypatch.setattr(
        exl3_module,
        "get_hf_file_to_dict",
        lambda *args, **kwargs: pytest.fail("index was read more than once"),
    )
    config.maybe_update_config(str(tmp_path))
    layer = Attention.__new__(Attention)
    torch.nn.Module.__init__(layer)

    assert isinstance(
        config.get_quant_method(layer, "language_model.layers.0.self_attn.attn"),
        BaseKVCacheMethod,
    )
    assert (
        config.get_quant_method(layer, "language_model.layers.1.self_attn.attn") is None
    )


def test_exl3_inventory_lookup_failure_is_not_cached_as_no_calibration(monkeypatch):
    config = Exl3Config(tensor_storage=_dense_storage())
    monkeypatch.setattr(exl3_module, "get_hf_file_to_dict", lambda *a, **kw: None)
    monkeypatch.setattr(exl3_module, "HF_HUB_OFFLINE", False, raising=False)
    monkeypatch.setattr(
        exl3_module, "file_or_path_exists", lambda *a, **kw: True, raising=False
    )

    with pytest.raises(ValueError, match="Could not read"):
        config.maybe_update_config("checkpoint")
    assert not config._kv_cache_scales_initialized

    inventory = {
        "weight_map": {
            "model.layers.0.self_attn.k_scale": "scales.safetensors",
            "model.layers.0.self_attn.v_scale": "scales.safetensors",
        }
    }
    monkeypatch.setattr(exl3_module, "get_hf_file_to_dict", lambda *a, **kw: inventory)
    config.maybe_update_config("checkpoint")
    assert config._has_kv_cache_scales("model.layers.0.self_attn.attn")


def test_exl3_unknown_offline_inventory_is_not_treated_as_absent(monkeypatch):
    config = Exl3Config(tensor_storage=_dense_storage())
    monkeypatch.setattr(exl3_module, "get_hf_file_to_dict", lambda *a, **kw: None)
    monkeypatch.setattr(exl3_module, "HF_HUB_OFFLINE", True, raising=False)
    monkeypatch.setattr(
        exl3_module, "try_get_local_file", lambda *a, **kw: None, raising=False
    )

    with pytest.raises(ValueError, match="offline mode"):
        config.maybe_update_config("uncached-checkpoint")
    assert not config._kv_cache_scales_initialized


@pytest.mark.parametrize("inventory", [{}, {"weight_map": None}, {"weight_map": []}])
def test_exl3_malformed_inventory_fails_without_initializing(monkeypatch, inventory):
    config = Exl3Config(tensor_storage=_dense_storage())
    monkeypatch.setattr(exl3_module, "get_hf_file_to_dict", lambda *a, **kw: inventory)
    with pytest.raises(ValueError, match="Invalid EXL3"):
        config.maybe_update_config("checkpoint")
    assert not config._kv_cache_scales_initialized


@pytest.mark.parametrize("scale", ["k", "v"])
def test_exl3_incomplete_kv_calibration_is_not_silently_ignored(monkeypatch, scale):
    config = Exl3Config(tensor_storage=_dense_storage())
    inventory = {"weight_map": {f"model.layers.0.self_attn.{scale}_scale": "s"}}
    monkeypatch.setattr(exl3_module, "get_hf_file_to_dict", lambda *a, **kw: inventory)
    with pytest.raises(ValueError, match="Incomplete EXL3"):
        config.maybe_update_config("checkpoint")
    assert not config._kv_cache_scales_initialized


def test_dense_exl3_warmup_covers_every_native_m_bucket_per_shard(monkeypatch):
    calls = []
    method = Exl3LinearMethod(Exl3Config())
    layer = torch.nn.Module()
    layer.exl3_shard_ids = ["q", "k"]
    for name in ("trellis", "suh", "svh", "mcg", "mul1"):
        setattr(layer, name, SimpleNamespace(exl3_tensors={}))
    for shard, packed_k in (("q", 128), ("k", 256)):
        layer.trellis.exl3_tensors[shard] = torch.zeros(
            (packed_k // 16, 64), dtype=torch.int16
        )
        layer.suh.exl3_tensors[shard] = torch.ones(packed_k)
        layer.svh.exl3_tensors[shard] = torch.ones(64)
    layer.mcg.exl3_tensors["q"] = torch.empty(0)
    layer.mul1.exl3_tensors["k"] = torch.empty(0)

    def record(x, trellis, suh, svh, mcg, mul1):
        calls.append((x.shape, trellis, suh, svh, mcg, mul1))

    monkeypatch.setattr(exl3_module, "_exl3_gemm", record)
    method._warm_autotune_buckets(layer)
    assert len(calls) == 8
    for offset, shard, packed_k in ((0, "q", 128), (4, "k", 256)):
        for index, rows in enumerate((2, 4, 8, 16)):
            shape, trellis, suh, svh, mcg, mul1 = calls[offset + index]
            assert shape == (rows, packed_k)
            assert trellis is layer.trellis.exl3_tensors[shard]
            assert suh is layer.suh.exl3_tensors[shard]
            assert svh is layer.svh.exl3_tensors[shard]
            assert mcg == (shard == "q")
            assert mul1 == (shard == "k")


def _required_kv_layer():
    from vllm.model_executor.layers.attention.attention import set_default_quant_scales
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

    layer = torch.nn.Module()
    layer.require_kv_cache_scales = True
    layer.layer_name = "mimo.target.self_attn.attn"
    layer.kv_cache_dtype = "fp8_e4m3"
    set_default_quant_scales(layer, register_buffer=True)
    method = BaseKVCacheMethod(None)
    method.create_weights(layer)
    return layer, method


@pytest.mark.parametrize("missing", ["k_scale", "v_scale", "both"])
def test_required_fp8_calibration_cannot_duplicate_or_default_missing_scales(missing):
    layer, method = _required_kv_layer()
    for name, value in (("k_scale", 0.3), ("v_scale", 0.7)):
        if missing not in (name, "both"):
            parameter = getattr(layer, name)
            parameter.weight_loader(parameter, torch.tensor(value))
    with pytest.raises(ValueError, match="required FP8"):
        method.process_weights_after_loading(layer)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_required_fp8_calibration_rejects_invalid_loaded_values(value):
    layer, _ = _required_kv_layer()
    with pytest.raises(ValueError, match="finite and positive"):
        layer.v_scale.weight_loader(layer.v_scale, torch.tensor(value))


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("q_scale", [None, 0.2])
def test_required_calibration_survives_attention_postload_without_unit_reset(
    q_scale, default_dtype, default_vllm_config
):
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    with set_default_torch_dtype(default_dtype):
        layer, method = _required_kv_layer()
    layer.quant_config = None
    layer.quant_method = method
    layer.impl = TritonAttentionImpl(32, 192, 192**-0.5, 2, None, None, "fp8_e4m3")
    quantizer = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)
    layer.query_quant = quantizer
    layer._checkpoint_query_quant = quantizer
    for name, value in (("k_scale", 0.3), ("v_scale", 0.7), ("q_scale", q_scale)):
        if value is not None:
            parameter = getattr(layer, name)
            assert parameter.dtype == torch.float32
            parameter.weight_loader(parameter, torch.tensor(value, dtype=torch.float32))
    method.process_weights_after_loading(layer)
    Attention.process_weights_after_loading(layer, torch.bfloat16)
    for name, value in (("_k_scale", 0.3), ("_v_scale", 0.7)):
        expected = torch.tensor(value, dtype=torch.float32).item()
        assert getattr(layer, name).dtype == torch.float32
        assert getattr(layer, name).item() == expected
        assert getattr(layer, name + "_cpu").dtype == torch.float32
        assert getattr(layer, name + "_cpu").item() == expected
        assert getattr(layer, name + "_float") == expected
    if q_scale is None:
        assert layer.query_quant is None
        assert not layer._q_scale_loaded
        assert not layer._prob_scale_loaded
        assert torch.isnan(layer._q_scale)
        assert torch.isnan(layer._prob_scale)
    else:
        assert layer.query_quant is quantizer
        assert layer._q_scale.item() == pytest.approx(q_scale)
    # Repeated post-load must retain the same producer-owned values.
    method.process_weights_after_loading(layer)
    Attention.process_weights_after_loading(layer, torch.bfloat16)


@pytest.mark.parametrize(
    "field",
    [
        "_k_scale",
        "_v_scale",
        "_k_scale_cpu",
        "_v_scale_cpu",
        "_k_scale_float",
        "_v_scale_float",
        "_q_scale",
    ],
)
def test_required_calibration_detects_overwritten_device_and_host_copies(field):
    layer, method = _required_kv_layer()
    for name, value in (("k_scale", 0.3), ("v_scale", 0.7)):
        parameter = getattr(layer, name)
        parameter.weight_loader(parameter, torch.tensor(value))
    method.process_weights_after_loading(layer)
    original = getattr(layer, field)
    if isinstance(original, torch.Tensor):
        original.fill_(1.0)
    else:
        setattr(layer, field, 1.0)
    with pytest.raises(ValueError):
        method.validate_required_scales(layer)


def test_required_calibration_blocks_default_reset_and_unknown_preprocessed_state():
    from vllm.model_executor.layers.attention.attention import set_default_quant_scales
    from vllm.model_executor.utils import weights_already_processed

    layer, method = _required_kv_layer()
    with pytest.raises(ValueError, match="cannot be reset"):
        set_default_quant_scales(layer)
    with weights_already_processed(), pytest.raises(ValueError, match="requires both"):
        method.process_weights_after_loading(layer)


def test_required_scale_policy_survives_draft_config_copy_and_replace():
    from copy import copy
    from dataclasses import replace

    from vllm.config import CacheConfig

    original = CacheConfig(cache_dtype="fp8_e4m3", require_kv_cache_scales=True)
    assert copy(original).require_kv_cache_scales
    assert replace(original, cache_dtype="fp8_e4m3").require_kv_cache_scales
    assert not CacheConfig().require_kv_cache_scales


def test_required_scales_reject_positive_unowned_dummy_values():
    layer, method = _required_kv_layer()
    layer.k_scale.data.fill_(0.3)
    layer.v_scale.data.fill_(0.7)
    with pytest.raises(ValueError, match="checkpoint-owned"):
        method.process_weights_after_loading(layer)


def test_required_scales_reject_conflicting_duplicate_and_reload_values():
    layer, method = _required_kv_layer()
    layer.k_scale.weight_loader(layer.k_scale, torch.tensor(0.3))
    with pytest.raises(ValueError, match="duplicate"):
        layer.k_scale.weight_loader(layer.k_scale, torch.tensor(0.6))
    layer.v_scale.weight_loader(layer.v_scale, torch.tensor(0.7))
    method.process_weights_after_loading(layer)
    method.create_weights(layer)
    with pytest.raises(ValueError, match="cache reinitialization"):
        layer.k_scale.weight_loader(layer.k_scale, torch.tensor(0.6))


def test_required_cache_writes_block_before_attention_validation():
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

    layer, method = _required_kv_layer()
    for name, value in (("k_scale", 0.3), ("v_scale", 0.7)):
        param = getattr(layer, name)
        param.weight_loader(param, torch.tensor(value))
    method.process_weights_after_loading(layer)
    with pytest.raises(RuntimeError, match="validation"):
        BaseKVCacheMethod.assert_scales_ready(layer)


@pytest.mark.parametrize("q_scale", [None, 0.2])
def test_required_calibration_survives_fresh_preprocessed_weight_transfer(q_scale):
    from vllm.model_executor.model_loader.weight_cache.daemon import (
        export_calibration_metadata,
    )
    from vllm.model_executor.model_loader.weight_cache.ipc_loader import (
        restore_calibration_metadata,
    )
    from vllm.model_executor.utils import weights_already_processed

    producer, method = _required_kv_layer()
    for name, value in (("k_scale", 0.3), ("v_scale", 0.7), ("q_scale", q_scale)):
        if value is not None:
            param = getattr(producer, name)
            param.weight_loader(param, torch.tensor(value))
    method.process_weights_after_loading(producer)
    metadata = export_calibration_metadata(producer)
    consumer, consumer_method = _required_kv_layer()
    for name in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale"):
        getattr(consumer, name).copy_(getattr(producer, name))
    restore_calibration_metadata(consumer, metadata)
    with weights_already_processed():
        consumer_method.process_weights_after_loading(consumer)
    consumer_method.validate_required_scales(consumer)
    assert consumer._k_scale_float == producer._k_scale_float
    assert consumer._q_scale_loaded == (q_scale is not None)
    with pytest.raises(ValueError, match="coverage"):
        restore_calibration_metadata(consumer, {})
    consumer._k_scale.fill_(1.0)
    with pytest.raises(ValueError, match="overwritten"):
        restore_calibration_metadata(consumer, metadata)


def test_required_scale_buffer_loader_invalidates_ready_attention():
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
    from vllm.model_executor.models.utils import AutoWeightsLoader

    layer, method = _required_kv_layer()
    for name, value in (("k_scale", 0.3), ("v_scale", 0.7)):
        param = getattr(layer, name)
        param.weight_loader(param, torch.tensor(value))
    method.process_weights_after_loading(layer)
    layer._kv_cache_scales_ready = True
    AutoWeightsLoader(layer).load_weights([("_k_scale", torch.tensor(1.0))])
    with pytest.raises(RuntimeError, match="validation"):
        BaseKVCacheMethod.assert_scales_ready(layer)
    with pytest.raises(ValueError, match="overwritten"):
        method.process_weights_after_loading(layer)


def test_required_state_dict_loading_invalidates_cache_readiness():
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

    layer = Attention.__new__(Attention)
    torch.nn.Module.__init__(layer)
    layer.require_kv_cache_scales = True
    layer.register_buffer("_k_scale", torch.tensor(0.3))
    layer._kv_cache_scales_ready = True
    layer.load_state_dict({"_k_scale": torch.tensor(1.0)})
    with pytest.raises(RuntimeError, match="validation"):
        BaseKVCacheMethod.assert_scales_ready(layer)


def test_required_attention_constructor_guards_direct_cache_write(default_vllm_config):
    from vllm.config import CacheConfig
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.attention.backends.triton_attn_diffkv import (
        TritonAttentionDiffKVBackend,
    )

    layer = Attention(
        32,
        192,
        192**-0.5,
        num_kv_heads=2,
        head_size_v=128,
        cache_config=CacheConfig(cache_dtype="fp8_e4m3", require_kv_cache_scales=True),
        attn_backend=TritonAttentionDiffKVBackend,
        prefix="model.layers.0.self_attn.attn",
    )
    from vllm.v1.attention.backends.triton_attn_diffkv import TritonAttentionDiffKVImpl

    assert isinstance(layer.impl, TritonAttentionDiffKVImpl)
    with pytest.raises(RuntimeError, match="validation"):
        layer.impl.do_kv_cache_update(layer, None, None, None, None)


def test_required_fp8_cache_policy_cannot_be_ignored_by_skip_layers(
    default_vllm_config,
):
    from vllm.config import CacheConfig
    from vllm.model_executor.layers.attention.attention import Attention

    cache = CacheConfig(
        cache_dtype="fp8_e4m3",
        require_kv_cache_scales=True,
        kv_cache_dtype_skip_layers=["0"],
    )
    with pytest.raises(ValueError, match="cannot be skipped"):
        Attention(
            32,
            192,
            192**-0.5,
            num_kv_heads=2,
            cache_config=cache,
            prefix="model.layers.0.self_attn.attn",
        )


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("construction_device", ["cpu", "meta"])
@pytest.mark.parametrize("default_dtype", [torch.float32, torch.bfloat16])
def test_kv_host_scales_stay_on_cpu_in_model_device_context(
    required, construction_device, default_dtype
):
    from vllm.model_executor.layers.attention.attention import set_default_quant_scales
    from vllm.utils.torch_utils import set_default_torch_dtype

    layer = torch.nn.Module()
    layer.require_kv_cache_scales = required
    with set_default_torch_dtype(default_dtype), torch.device(construction_device):
        set_default_quant_scales(layer, register_buffer=True)
    for name in ("_k_scale_cpu", "_v_scale_cpu"):
        value = getattr(layer, name)
        assert value.device.type == "cpu"
        assert value.dtype == torch.float32
        if required:
            assert torch.isnan(value)
        else:
            assert value.item() == 1.0
    # Device buffers still belong to the model's construction device.
    for name in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale"):
        value = getattr(layer, name)
        assert value.device.type == construction_device
        assert value.dtype == torch.float32


@pytest.mark.parametrize("q_scale", [None, 0.2])
@pytest.mark.parametrize("restore_device", ["cpu", "meta"])
def test_kv_host_scale_restore_ignores_ambient_device(q_scale, restore_device):
    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

    producer, method = _required_kv_layer()
    for name, value in (("k_scale", 0.3), ("v_scale", 0.7), ("q_scale", q_scale)):
        if value is not None:
            parameter = getattr(producer, name)
            parameter.weight_loader(parameter, torch.tensor(value))
    method.process_weights_after_loading(producer)
    metadata = BaseKVCacheMethod.export_calibration(producer)
    consumer, _ = _required_kv_layer()
    for name in ("_k_scale", "_v_scale", "_q_scale", "_prob_scale"):
        getattr(consumer, name).copy_(getattr(producer, name))
    with torch.device(restore_device):
        BaseKVCacheMethod.restore_calibration(consumer, metadata)
    for name in ("_k_scale", "_v_scale"):
        host_value = getattr(consumer, name + "_cpu")
        assert host_value.device.type == "cpu"
        assert host_value.dtype == torch.float32
        assert host_value.item() == getattr(consumer, name + "_float")
        assert host_value.item() == getattr(producer, name).item()


def test_exl3_gemm_retains_fp32_native_accumulation_and_fp16_public_output():
    """Static wrapper contract; this does not qualify native numeric behavior."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(exl3_module.__file__).read_text())
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    function = functions["_exl3_gemm"]
    output_allocation = next(
        node.value
        for node in function.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "output"
            for target in node.targets
        )
    )
    assert isinstance(output_allocation, ast.Call)
    assert ast.unparse(output_allocation.func) == "torch.empty"
    dtype = next(
        keyword.value
        for keyword in output_allocation.keywords
        if keyword.arg == "dtype"
    )
    assert ast.unparse(dtype) == "torch.float32"
    native_call = next(
        node.value
        for node in function.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "ext.exl3_gemm"
    )
    assert len(native_call.args) == 10
    assert ast.unparse(native_call.args[2]) == "output"
    result = next(node.value for node in function.body if isinstance(node, ast.Return))
    assert ast.unparse(result) == "output.to(torch.float16)"
    fake_result = next(
        node.value
        for node in functions["_exl3_gemm_fake"].body
        if isinstance(node, ast.Return)
    )
    assert isinstance(fake_result, ast.Call)
    fake_dtype = next(
        keyword.value for keyword in fake_result.keywords if keyword.arg == "dtype"
    )
    assert ast.unparse(fake_dtype) == "torch.float16"
