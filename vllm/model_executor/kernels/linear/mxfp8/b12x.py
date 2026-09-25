# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm import envs
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.b12x import B12xWarmupUnit, reuse_packed_weight_storage
from vllm.utils.b12x import (
    get_b12x_mxfp8_linear as _import_b12x_mxfp8,
)

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


def _b12x_mxfp8_max_m() -> int:
    """Return the largest row count routed through the B12X GEMM."""
    return int(envs.VLLM_B12X_MXFP8_MAX_M)


def _apply_flashinfer_mxfp8_fallback(
    input_2d: torch.Tensor,
    bias: torch.Tensor | None,
    weight: torch.Tensor,
    weight_scale_swizzled: torch.Tensor,
) -> torch.Tensor:
    """Run the proven v2.8 FlashInfer CUTLASS path for larger row counts."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.utils import flashinfer as vllm_flashinfer

    input_mxfp8, input_scale = mxfp8_e4m3_quantize(
        input_2d, is_sf_swizzled_layout=True
    )
    output = vllm_flashinfer.mm_mxfp8(
        input_mxfp8,
        weight.t(),
        input_scale,
        weight_scale_swizzled,
        out_dtype=input_2d.dtype,
        backend="cutlass",
    )
    return output if bias is None else output + bias


def _apply_b12x_mxfp8_packed_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    packed_weight = layer.b12x_mxfp8_packed_weight

    input_2d = x.reshape(-1, x.shape[-1]).contiguous()
    output_shape = [*x.shape[:-1], int(packed_weight.out_features)]

    max_m = _b12x_mxfp8_max_m()
    if max_m > 0 and int(input_2d.shape[0]) > max_m:
        fallback_weight = getattr(layer, "b12x_mxfp8_fallback_weight", None)
        fallback_scale = getattr(layer, "b12x_mxfp8_fallback_scale", None)
        if fallback_weight is None or fallback_scale is None:
            raise RuntimeError(
                "B12X MXFP8 has no FlashInfer CUTLASS fallback for "
                f"M={input_2d.shape[0]}"
            )
        return _apply_flashinfer_mxfp8_fallback(
            input_2d, bias, fallback_weight, fallback_scale
        ).view(*output_shape)

    mxfp8 = _import_b12x_mxfp8()
    assert mxfp8 is not None
    output = mxfp8.mm(
        input_2d,
        packed_weight,
        bias=bias,
        expected_m=max(1, int(input_2d.shape[0])),
    )
    return output.view(*output_shape)


class B12xMxfp8LinearKernel(Mxfp8LinearKernel):
    """ModelOpt MXFP8 linear through the native b12x SM120 dense GEMM path."""

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "b12x MXFP8 kernels are only available on CUDA"
        if not current_platform.is_device_capability_family(120):
            return False, "b12x MXFP8 kernels require a Blackwell 12x device"
        mxfp8 = _import_b12x_mxfp8()
        if mxfp8 is None:
            return False, "Install the B12X backend with `pip install vllm[b12x]`"
        if not mxfp8.is_supported():
            return False, "b12x.gemm.mxfp8_linear is not supported"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        del c
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        assert weight.dtype == MXFP8_VALUE_DTYPE, (
            f"b12x MXFP8 requires {MXFP8_VALUE_DTYPE}, got {weight.dtype}"
        )
        assert weight.ndim == 2, f"b12x MXFP8 weight must be 2D, got {weight.ndim}D"
        assert hasattr(layer, "weight_scale"), "b12x MXFP8 linear requires weight_scale"

        out_features, in_features = map(int, weight.shape)
        assert in_features % MXFP8_BLOCK_SIZE == 0, (
            "b12x MXFP8 requires input features divisible by "
            f"{MXFP8_BLOCK_SIZE}, got {in_features}"
        )
        weight_scale = layer.weight_scale.data
        assert weight_scale.dtype == MXFP8_SCALE_DTYPE, (
            f"b12x MXFP8 requires {MXFP8_SCALE_DTYPE} weight_scale, "
            f"got {weight_scale.dtype}"
        )
        assert weight_scale.ndim == 2, (
            f"b12x MXFP8 weight_scale must be 2D, got {weight_scale.ndim}D"
        )

        mxfp8 = _import_b12x_mxfp8()
        assert mxfp8 is not None
        scale_k = in_features // MXFP8_BLOCK_SIZE
        packed_weight = mxfp8.pack_weight(
            weight[:out_features, :in_features].detach(),
            weight_scale[:out_features, :scale_k].detach(),
        )
        layer.b12x_mxfp8_packed_weight = reuse_packed_weight_storage(
            getattr(layer, "b12x_mxfp8_packed_weight", None),
            packed_weight,
        )
        layer.b12x_mxfp8_fallback_weight = None
        layer.b12x_mxfp8_fallback_scale = None
        if _b12x_mxfp8_max_m() > 0 and in_features >= 128 and out_features >= 128:
            from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
                swizzle_mxfp8_scale,
            )
            from vllm.utils.flashinfer import has_flashinfer

            if has_flashinfer():
                layer.b12x_mxfp8_fallback_weight = weight[
                    :out_features, :in_features
                ].detach()
                layer.b12x_mxfp8_fallback_scale = swizzle_mxfp8_scale(
                    weight_scale[:out_features, :scale_k].contiguous(),
                    M=out_features,
                    K=in_features,
                ).contiguous()
        replace_parameter(layer, "weight", weight.new_empty((0,)))
        replace_parameter(layer, "weight_scale", weight_scale.new_empty((0,)))
        layer.b12x_warmup_provider = self

    def get_b12x_warmup_unit(
        self,
        layer: torch.nn.Module,
        token_counts: tuple[int, ...],
        output_dtype: torch.dtype,
    ) -> B12xWarmupUnit:
        packed_weight = layer.b12x_mxfp8_packed_weight
        device = torch.device(packed_weight.weight.values.device)

        def compile() -> None:
            for tokens in token_counts:
                source = torch.zeros(
                    (tokens, int(packed_weight.in_features)),
                    dtype=output_dtype,
                    device=device,
                )
                _apply_b12x_mxfp8_packed_linear(layer, source, None)

        return B12xWarmupUnit(
            name="MXFP8",
            key=(
                type(self),
                device,
                int(packed_weight.in_features),
                int(packed_weight.padded_in_features),
                int(packed_weight.out_features),
                output_dtype,
            ),
            compile=compile,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _apply_b12x_mxfp8_packed_linear(layer, x, bias)
