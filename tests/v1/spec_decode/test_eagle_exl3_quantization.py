# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.config import LoadConfig, SpeculativeConfig
from vllm.model_executor.layers.quantization import exl3 as exl3_module
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


class _Exl3QuantConfig:
    @staticmethod
    def get_name():
        return "exl3"


@dataclass
class _SpecConfig:
    draft_model_config: object
    attention_backend: str | None = None
    moe_backend: str | None = None
    kv_cache_dtype: str | None = None
    draft_load_config: object = None

    apply_draft_overrides = SpeculativeConfig.apply_draft_overrides


@dataclass
class _VllmConfig:
    model_config: object
    quant_config: object
    speculative_config: _SpecConfig
    load_config: LoadConfig = field(default_factory=LoadConfig)


class _Captured(Exception):
    def __init__(self, vllm_config):
        self.vllm_config = vllm_config


def test_eagle_exl3_uses_independent_draft_config_and_runtime_owner():
    target_model_config = SimpleNamespace(runner_type="generate")
    draft_model_config = SimpleNamespace(runner_type="draft")
    target_quant_config = _Exl3QuantConfig()
    draft_quant_config = _Exl3QuantConfig()
    config = _VllmConfig(
        model_config=target_model_config,
        quant_config=target_quant_config,
        speculative_config=_SpecConfig(draft_model_config=draft_model_config),
    )

    def capture_model(*, vllm_config, model_config):
        assert model_config is draft_model_config
        raise _Captured(vllm_config)

    with (
        patch(
            "vllm.v1.worker.gpu.spec_decode.eagle.utils.get_draft_load_config",
            return_value=config.load_config,
        ),
        patch(
            "vllm.v1.worker.gpu.spec_decode.eagle.utils.get_pp_safe_draft_load_config",
            side_effect=lambda load_config: load_config,
        ),
        patch(
            "vllm.v1.worker.gpu.spec_decode.eagle.utils.get_draft_quant_config",
            return_value=draft_quant_config,
        ),
        patch(
            "vllm.v1.worker.gpu.spec_decode.eagle.utils.get_model",
            side_effect=capture_model,
        ),
        pytest.raises(_Captured) as exc,
    ):
        load_eagle_model(object(), config)

    used = exc.value.vllm_config
    assert used.model_config is draft_model_config
    assert used.quant_config is draft_quant_config
    assert config.model_config is target_model_config
    assert config.quant_config is target_quant_config

    target_owner = exl3_module._runtime_owner_token(
        target_quant_config,
        SimpleNamespace(exl3_is_draft=False),
    )
    draft_owner = exl3_module._runtime_owner_token(
        used.quant_config,
        SimpleNamespace(exl3_is_draft=used.model_config.runner_type == "draft"),
    )
    assert target_owner != draft_owner
    assert target_owner[0] != draft_owner[0]
    assert target_owner[1:] == (False,)
    assert draft_owner[1:] == (True,)


def test_eagle_non_exl3_preserves_existing_construction_config():
    target_model_config = SimpleNamespace(runner_type="generate")
    draft_model_config = SimpleNamespace(runner_type="draft")
    non_exl3 = SimpleNamespace(get_name=lambda: "fp8")
    config = _VllmConfig(
        model_config=target_model_config,
        quant_config=non_exl3,
        speculative_config=_SpecConfig(draft_model_config=draft_model_config),
    )

    def capture_model(*, vllm_config, model_config):
        assert model_config is draft_model_config
        raise _Captured(vllm_config)

    with (
        patch(
            "vllm.v1.worker.gpu.spec_decode.eagle.utils.get_draft_load_config",
            return_value=config.load_config,
        ),
        patch(
            "vllm.v1.worker.gpu.spec_decode.eagle.utils.get_pp_safe_draft_load_config",
            side_effect=lambda load_config: load_config,
        ),
        patch(
            "vllm.v1.worker.gpu.spec_decode.eagle.utils.get_draft_quant_config"
        ) as get_draft_quant,
        patch(
            "vllm.v1.worker.gpu.spec_decode.eagle.utils.get_model",
            side_effect=capture_model,
        ),
        pytest.raises(_Captured) as exc,
    ):
        load_eagle_model(object(), config)

    assert exc.value.vllm_config is config
    get_draft_quant.assert_not_called()
