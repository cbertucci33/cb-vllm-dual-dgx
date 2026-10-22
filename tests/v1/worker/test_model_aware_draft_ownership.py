# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ownership contracts without model construction, weights or devices."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec


def test_v1_load_tags_added_attention_before_cache_filter(monkeypatch):
    import vllm.v1.spec_decode.llm_base_proposer as mod

    target = NS(is_draft_layer=False, get_kv_cache_spec=lambda _: object())
    draft = NS(is_draft_layer=False, get_kv_cache_spec=lambda _: object())
    no_cache = NS(is_draft_layer=False, get_kv_cache_spec=lambda _: None)
    calls = iter(
        [{"target": target}, {"target": target, "draft": draft, "no_cache": no_cache}]
    )
    monkeypatch.setattr(mod, "get_layers_from_vllm_config", lambda *_: next(calls))
    spec = NS(
        vllm_config=object(),
        _get_model=lambda: object(),
        supports_mm_inputs=False,
        parallel_drafting=False,
        _maybe_share_embeddings=Mock(),
        _maybe_share_lm_head=Mock(),
    )
    target_model = object()
    mod.SpecDecodeBaseProposer.load_model(spec, target_model)
    assert spec._draft_attn_layer_names == {"draft"}
    assert draft.is_draft_layer and no_cache.is_draft_layer
    assert not target.is_draft_layer
    spec._maybe_share_embeddings.assert_called_once_with(target_model)
    spec._maybe_share_lm_head.assert_called_once_with(target_model)


def test_v1_spec_tagging_does_not_mutate_shared_format(monkeypatch):
    import vllm.v1.worker.gpu_model_runner as mod
    from vllm.v1.attention.backend import AttentionBackend

    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=128, dtype=torch.bfloat16
    )
    backend = NS(
        customize_spec=AttentionBackend.customize_spec,
        get_supported_kernel_block_sizes=lambda _=None: [16],
    )
    layers = {
        name: NS(
            is_draft_layer=name == "draft",
            get_kv_cache_spec=lambda _: spec,
            get_attn_backend=lambda: backend,
        )
        for name in ["target", "draft"]
    }
    monkeypatch.setattr(mod, "get_layers_from_vllm_config", lambda *_: layers)
    monkeypatch.setattr(mod, "has_ec_transfer", lambda: False)
    runner = NS(vllm_config=object(), shared_kv_cache_layers={})
    specs = mod.GPUModelRunner.get_kv_cache_spec(runner)
    expected = mod.customize_attention_spec(backend, spec)
    assert specs["target"] == expected
    assert not specs["target"].is_draft
    assert not spec.is_draft and spec.block_stride_alignment is None
    assert specs["draft"].is_draft
    assert specs["draft"] == expected
    assert specs["draft"] is not spec
