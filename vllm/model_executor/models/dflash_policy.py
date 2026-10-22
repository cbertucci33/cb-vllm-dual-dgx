# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DFlashExecutionPolicy:
    fused_mxfp8_context_kv: bool = False
    parallel_padding: bool = False


_DEFAULT_POLICY = DFlashExecutionPolicy()
_GLM_MXFP8_POLICY = DFlashExecutionPolicy(True, True)
_MIMO_DFLASH_POLICY = DFlashExecutionPolicy(parallel_padding=True)


def resolve_dflash_execution_policy(vllm_config: Any) -> DFlashExecutionPolicy:
    """Resolve from the target, not the Qwen architecture shared by drafters."""
    target = vllm_config.model_config.hf_config
    spec = vllm_config.speculative_config
    if spec is None:
        return _DEFAULT_POLICY
    draft = spec.draft_model_config.hf_config
    quant = getattr(draft, "quantization_config", None) or {}
    if (
        getattr(target, "model_type", None) == "glm5_next"
        and "Glm5NextForConditionalGeneration"
        in (getattr(target, "architectures", None) or [])
        and getattr(draft, "model_type", None) == "qwen3"
        and "DFlash2DraftModel" in (getattr(draft, "architectures", None) or [])
        and quant.get("quant_method") == "modelopt"
        and (quant.get("quantization") or {}).get("quant_algo") == "MXFP8"
    ):
        return _GLM_MXFP8_POLICY
    if (
        getattr(target, "model_type", None) == "mimo_v2"
        and set(getattr(target, "architectures", None) or [])
        & {"MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM", "MiMoV2OmniForCausalLM"}
        and getattr(draft, "model_type", None) == "qwen3"
        and "DFlashDraftModel" in (getattr(draft, "architectures", None) or [])
    ):
        return _MIMO_DFLASH_POLICY
    return _DEFAULT_POLICY
