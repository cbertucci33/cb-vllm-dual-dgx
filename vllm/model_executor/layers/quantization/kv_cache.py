# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.utils import is_weights_pre_processed
from vllm.platforms import current_platform
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.kv_cache_interface import kv_cache_uses_per_token_head_scales

logger = init_logger(__name__)


class KVCacheScaleParameter(torch.nn.Parameter):
    """Scalar parameter for KV-cache scales.

    Initialized to -1.0 (an invalid sentinel) so call sites just write
    `KVCacheScaleParameter()`. The `weight_loader` accepts shape `()` or
    `(1,)` and rejects anything else — per-head scales go through a separate
    path (compressed-tensors' `_tp_aware_loader`), not this one. Per-instance
    overrides still work because instance attribute assignment shadows this
    class-level loader.
    """

    def __new__(cls) -> "KVCacheScaleParameter":
        return super().__new__(
            cls, torch.tensor(-1.0, dtype=torch.float32), requires_grad=False
        )

    @staticmethod
    def weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if loaded_weight.numel() != 1:
            raise ValueError(
                f"KV-cache scale expects a scalar weight, got shape "
                f"{tuple(loaded_weight.shape)}"
            )
        stored = loaded_weight.reshape(()).to(device=param.device, dtype=param.dtype)
        if getattr(param, "require_valid_scale", False) and not (
            loaded_weight.is_floating_point()
            and torch.isfinite(stored).all()
            and (stored > 0).all()
        ):
            raise ValueError("Required FP8 calibration must be finite and positive")
        if getattr(param, "require_valid_scale", False) and not (
            torch.isfinite(loaded_weight).all() and (loaded_weight > 0).all()
        ):
            raise ValueError("Required FP8 calibration must be finite and positive")
        if getattr(param, "require_valid_scale", False):
            expected = getattr(param, "expected_calibration", None)
            if expected is not None and not torch.equal(
                stored,
                torch.full_like(stored, expected),
            ):
                raise ValueError(
                    "Calibration cannot change without cache reinitialization"
                )
            if getattr(param, "_calibration_loaded", False) and not torch.equal(
                param.data, stored
            ):
                raise ValueError("Conflicting duplicate FP8 calibration assignment")
            param._calibration_loaded = True
        param.data.copy_(stored)


class BaseKVCacheMethod(QuantizeMethodBase):
    """Quant method that adds `_k_scale` and `_v_scale` attributes to the
    Attention layer to support loading those scaling factors from checkpoints.
    The k/v_scale will be used to:
        - quantize k/v_cache entries before saving them to the cache
        - dequantize k/v_cache entries before fetching them from the cache

    Args:
        quant_config: the appropriate QuantizationConfig

    """

    supports_pre_processed_weights = True

    def __init__(self, quant_config: QuantizationConfig | None):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module):
        """Create "weight" (aka q_scale, k_scale and v_scale)
        for an attention layer.
        """
        # Initialize the Q and KV cache scales to -1.0, an invalid value.
        # If the q and k/v_scales appear in the checkpoint, it will be
        # overwritten when loading weights.
        layer.q_scale = KVCacheScaleParameter()
        layer.k_scale = KVCacheScaleParameter()
        layer.v_scale = KVCacheScaleParameter()
        # Initialize P = softmax(QK^T) scales
        layer.prob_scale = KVCacheScaleParameter()
        layer._kv_cache_scales_loaded = False
        layer._kv_cache_scales_ready = False
        layer._q_scale_loaded = False
        layer._prob_scale_loaded = False
        if getattr(layer, "require_kv_cache_scales", False):
            for name in ("q_scale", "k_scale", "v_scale", "prob_scale"):
                getattr(layer, name).require_valid_scale = True
                getattr(layer, name).expected_calibration = getattr(
                    layer, "_kv_cache_checkpoint_scales", {}
                ).get(name)

    @staticmethod
    def export_calibration(layer: torch.nn.Module) -> dict:
        BaseKVCacheMethod.validate_required_scales(layer)
        return {
            "checkpoint": dict(layer._kv_cache_checkpoint_scales),
            "processed": dict(layer._kv_cache_scale_provenance),
            "q": getattr(layer, "_q_scale_provenance", None),
            "prob": getattr(layer, "_prob_scale_provenance", None),
        }

    @staticmethod
    def restore_calibration(layer: torch.nn.Module, metadata: dict) -> None:
        if set(metadata) != {"checkpoint", "processed", "q", "prob"}:
            raise ValueError("Invalid FP8 calibration metadata fields")
        checkpoint = metadata["checkpoint"]
        processed = metadata["processed"]
        if (
            not isinstance(checkpoint, dict)
            or not {"k_scale", "v_scale"} <= set(checkpoint)
            or set(checkpoint) - {"k_scale", "v_scale", "q_scale", "prob_scale"}
            or set(processed) != {"_k_scale", "_v_scale"}
        ):
            raise ValueError("Missing checkpoint-owned FP8 IPC calibration")
        factor = 2 if current_platform.is_fp8_fnuz() else 1
        for name, raw in checkpoint.items():
            if not isinstance(raw, float) or not math.isfinite(raw) or raw <= 0:
                raise ValueError("Invalid FP8 IPC checkpoint calibration")
            expected = raw * factor
            observed = (
                processed["_" + name]
                if name in ("k_scale", "v_scale")
                else metadata["q" if name == "q_scale" else "prob"]
            )
            if observed != expected:
                raise ValueError("FP8 IPC calibration provenance disagrees")
        for name, key in (("q_scale", "q"), ("prob_scale", "prob")):
            if name not in checkpoint and metadata[key] is not None:
                raise ValueError("Unowned FP8 IPC query/probability calibration")
        layer._kv_cache_checkpoint_scales = dict(checkpoint)
        layer._kv_cache_scale_provenance = dict(processed)
        layer._kv_cache_scales_loaded = True
        layer._kv_cache_scales_ready = False
        layer._q_scale_loaded = "q_scale" in checkpoint
        layer._prob_scale_loaded = "prob_scale" in checkpoint
        layer._validated_q_scale_policy = layer._q_scale_loaded
        layer._q_scale_provenance = metadata["q"]
        layer._prob_scale_provenance = metadata["prob"]
        layer._k_scale_float = processed["_k_scale"]
        layer._v_scale_float = processed["_v_scale"]
        layer._k_scale_cpu = torch.tensor(
            layer._k_scale_float, dtype=torch.float32, device="cpu"
        )
        layer._v_scale_cpu = torch.tensor(
            layer._v_scale_float, dtype=torch.float32, device="cpu"
        )
        layer._q_scale_float = metadata["q"] if layer._q_scale_loaded else float("nan")
        layer._prob_scale_float = (
            metadata["prob"] if layer._prob_scale_loaded else float("nan")
        )
        BaseKVCacheMethod.validate_required_scales(layer)

    @staticmethod
    def assert_scales_ready(layer: torch.nn.Module) -> None:
        if getattr(layer, "require_kv_cache_scales", False) and not getattr(
            layer, "_kv_cache_scales_ready", False
        ):
            raise RuntimeError("Required FP8 calibration has not completed validation")

    @staticmethod
    def validate_required_scales(layer: torch.nn.Module) -> None:
        if not getattr(layer, "_kv_cache_scales_loaded", False):
            raise ValueError(
                f"{getattr(layer, 'layer_name', '<attention>')}: "
                "static FP8 K/V requires both checkpoint calibration scales"
            )
        for name in ("_k_scale", "_v_scale"):
            scale = getattr(layer, name)
            if not (torch.isfinite(scale).all() and (scale > 0).all()):
                raise ValueError(f"Invalid required FP8 calibration: {name}")
            expected = getattr(layer, "_kv_cache_scale_provenance", {}).get(name)
            if expected is None or not torch.equal(
                scale, torch.full_like(scale, expected)
            ):
                raise ValueError(f"Required FP8 calibration was overwritten: {name}")
            host_name = name + "_float"
            cpu_name = name + "_cpu"
            if getattr(layer, host_name) != expected or not torch.equal(
                getattr(layer, cpu_name),
                torch.full_like(getattr(layer, cpu_name), expected),
            ):
                raise ValueError(f"Required FP8 calibration copies disagree: {name}")
        if layer._q_scale_loaded:
            expected = getattr(layer, "_q_scale_provenance", None)
            if (
                expected is None
                or not torch.equal(
                    layer._q_scale, torch.full_like(layer._q_scale, expected)
                )
                or layer._q_scale_float != expected
            ):
                raise ValueError("Required Q calibration was overwritten")
        elif not (
            torch.isnan(layer._q_scale).all() and math.isnan(layer._q_scale_float)
        ):
            raise ValueError("Uncalibrated Q must not acquire a unit-scale fallback")

        if layer._prob_scale_loaded:
            expected = getattr(layer, "_prob_scale_provenance", None)
            if (
                expected is None
                or not torch.equal(
                    layer._prob_scale, torch.full_like(layer._prob_scale, expected)
                )
                or layer._prob_scale_float != expected
            ):
                raise ValueError("Required probability calibration was overwritten")
        elif not (
            torch.isnan(layer._prob_scale).all() and math.isnan(layer._prob_scale_float)
        ):
            raise ValueError("Uncalibrated probability must not acquire unit scales")

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError(f"{self.__class__.__name__}.apply should not be called.")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        required = getattr(layer, "require_kv_cache_scales", False)
        if is_weights_pre_processed():
            if required:
                self.validate_required_scales(layer)
            # Scale buffers were exported with post-processed values; drop the
            # consumed scale placeholders and rebuild the host-side copies.
            for name in ("k_scale", "v_scale", "q_scale", "prob_scale"):
                if hasattr(layer, name):
                    delattr(layer, name)
            if hasattr(layer, "_k_scale"):
                layer._k_scale_float = layer._k_scale.item()
                layer._v_scale_float = layer._v_scale.item()
                layer._q_scale_float = layer._q_scale.item()
                layer._k_scale_cpu = layer._k_scale.detach().to("cpu")
                layer._v_scale_cpu = layer._v_scale.detach().to("cpu")
            return

        # skip if there are no weights to process (for example, weight reloading)
        if not hasattr(layer, "q_scale"):
            if required:
                self.validate_required_scales(layer)
            assert not hasattr(layer, "k_scale")
            assert not hasattr(layer, "v_scale")
            assert not hasattr(layer, "prob_scale")
            return

        if required:
            for name in ("k_scale", "v_scale"):
                scale = getattr(layer, name)
                if not getattr(scale, "_calibration_loaded", False):
                    raise ValueError(f"Missing checkpoint-owned required FP8 {name}")
                if not (torch.isfinite(scale).all() and (scale > 0).all()):
                    raise ValueError(
                        f"{getattr(layer, 'layer_name', '<attention>')}: "
                        f"missing or invalid required FP8 {name}"
                    )
        old_q_policy = getattr(layer, "_validated_q_scale_policy", None)
        if required:
            layer._kv_cache_checkpoint_scales = {
                name: getattr(layer, name).item()
                for name in ("k_scale", "v_scale", "q_scale", "prob_scale")
                if getattr(getattr(layer, name), "_calibration_loaded", False)
            }
        layer._q_scale_loaded = bool(
            torch.isfinite(layer.q_scale).all()
            and (layer.q_scale > 0).all()
            and (not required or getattr(layer.q_scale, "_calibration_loaded", False))
        )

        if (
            required
            and old_q_policy is not None
            and old_q_policy != layer._q_scale_loaded
        ):
            raise ValueError(
                "Changing Q calibration presence requires graph reinitialization"
            )
        if required:
            layer._validated_q_scale_policy = layer._q_scale_loaded

        layer._prob_scale_loaded = bool(
            torch.isfinite(layer.prob_scale).all()
            and (layer.prob_scale > 0).all()
            and (
                not required or getattr(layer.prob_scale, "_calibration_loaded", False)
            )
        )

        # Per-token-head quantized KV cache: scales are computed dynamically
        # per (token, head) in the kernel at cache-write time.
        if kv_cache_uses_per_token_head_scales(layer.kv_cache_dtype):
            layer._k_scale.copy_(1.0)
            layer._v_scale.copy_(1.0)
            layer._k_scale_float = 1.0
            layer._v_scale_float = 1.0
            del layer.k_scale
            del layer.v_scale
            del layer.q_scale
            del layer.prob_scale
            return

        # If the kv-cache is not quantized, we enforce the k/v_scale to be 1.0
        # regardless whether the kv-scale is available in the checkpoint.
        # No need to process kv scales after loading if we are going to
        # calculate them on the fly.
        if is_quantized_kv_cache(layer.kv_cache_dtype):
            if layer.k_scale > 0.0 and layer.v_scale > 0.0:
                # We prefer to use separate k_scale and v_scale if present
                k_scale = layer.k_scale.to("cpu").tolist()
                v_scale = layer.v_scale.to("cpu").tolist()
                if current_platform.is_fp8_fnuz():
                    k_scale *= 2
                    v_scale *= 2
            elif layer.k_scale < 0.0 and layer.v_scale < 0.0:
                # If no scales were loaded (both scales are invalid negative
                # values), use the default value of 1.0
                k_scale = 1.0
                v_scale = 1.0
            else:
                # If we find a single kv_scale in the checkpoint, we remap
                # kv_scale to k_scale during weight loading, and duplicate
                # k_scale to v_scale here
                assert layer.k_scale > 0.0
                scale_to_duplicate = max(layer.k_scale, layer.v_scale)
                k_scale = scale_to_duplicate.to("cpu").tolist()
                v_scale = scale_to_duplicate.to("cpu").tolist()
                if current_platform.is_fp8_fnuz():
                    k_scale *= 2
                    v_scale *= 2

            if not isinstance(k_scale, float) or not isinstance(v_scale, float):
                raise ValueError(
                    "Only support per-tensor scaling factor for fp8 KV cache"
                )

            if layer.q_scale < 0.0 and not required:
                logger.warning_once(
                    "Checkpoint does not provide Q calibration; legacy Q scale is 1.0"
                )

            # These are used in the final Attention.forward()
            layer._k_scale.copy_(k_scale)
            layer._v_scale.copy_(v_scale)
            layer._k_scale_float = k_scale
            layer._v_scale_float = v_scale
            # Sync host (cpu) scale copies read by AITER fused kernels.
            layer._k_scale_cpu.fill_(k_scale)
            layer._v_scale_cpu.fill_(v_scale)
            layer._kv_cache_scale_provenance = {
                "_k_scale": k_scale,
                "_v_scale": v_scale,
            }
            layer._kv_cache_scales_loaded = True
            if k_scale == 1.0 and v_scale == 1.0 and "e5m2" not in layer.kv_cache_dtype:
                logger.warning_once(
                    "Using KV cache scaling factor 1.0 for fp8_e4m3. "
                    "If this is unintended, verify that k/v_scale "
                    "scaling factors are properly set in the checkpoint."
                )

        if layer._q_scale_loaded:
            q_scale = layer.q_scale
            if current_platform.is_fp8_fnuz():
                q_scale *= 2
        else:
            q_scale = float("nan") if required else 1.0
        if layer._prob_scale_loaded:
            prob_scale = layer.prob_scale
            if current_platform.is_fp8_fnuz():
                prob_scale *= 2
        else:
            prob_scale = float("nan") if required else 1.0

        is_singleton_float = lambda x: (
            isinstance(x, float)
            or isinstance(x, torch.Tensor)
            and x.numel() == 1
            and x.is_floating_point()
        )
        if not is_singleton_float(q_scale) or not is_singleton_float(prob_scale):
            raise ValueError(
                "Only support per-tensor scaling factorfor fp8-quantized Q/prob"
            )

        # These are used in the final Attention.forward()
        layer._q_scale.copy_(q_scale)
        layer._q_scale_float = (
            q_scale.item() if isinstance(q_scale, torch.Tensor) else q_scale
        )

        if layer._q_scale_loaded:
            layer._q_scale_provenance = layer._q_scale_float

        layer._prob_scale.copy_(prob_scale)
        if required:
            layer._prob_scale_float = (
                prob_scale.item()
                if isinstance(prob_scale, torch.Tensor)
                else prob_scale
            )
            if layer._prob_scale_loaded:
                layer._prob_scale_provenance = layer._prob_scale_float
        if layer.kv_cache_dtype == "fp8" and (q_scale == 1.0 or prob_scale == 1.0):
            logger.warning_once(
                f"Using uncalibrated q_scale {q_scale} and/or prob_scale "
                f"{prob_scale} with fp8 attention. This may cause accuracy "
                "issues. Please make sure q/prob scaling factors are "
                "available in the fp8 checkpoint."
            )

        del layer.k_scale
        del layer.v_scale
        del layer.q_scale
        del layer.prob_scale
