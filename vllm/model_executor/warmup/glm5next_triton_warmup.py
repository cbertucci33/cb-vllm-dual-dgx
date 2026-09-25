# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm GLM-5.3 vision Triton kernels used after image input arrives."""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _warm_vision_rotary(model: torch.nn.Module, device: torch.device) -> None:
    from vllm.models.glm5next.nvidia.multimodal import (
        Glm5NextVisionTransformer,
    )

    for visual in model.modules():
        if not isinstance(visual, Glm5NextVisionTransformer):
            continue
        if visual.device.type != device.type:
            continue

        attention = visual.blocks[0].attn
        num_heads = int(attention.num_attention_heads_per_partition)
        head_size = int(attention.hidden_size_per_attention_head)
        # Cover Triton's %16==0 and non-%16 integer specialization classes.
        grids = ((1, 32, 32), (1, visual.spatial_merge_size, visual.spatial_merge_size))
        for grid in grids:
            cos, sin, _ = visual.rot_pos_emb([list(grid)])
            qk = torch.empty(
                (2, grid[0] * grid[1] * grid[2], num_heads, head_size),
                dtype=cos.dtype,
                device=cos.device,
            )
            attention.apply_rotary_emb(qk, cos, sin)

        logger.info("Warmed GLM-5.3 vision rotary kernels on grids=%s.", grids)


def _warm_cache_checkpoint_store(
    model: torch.nn.Module, device: torch.device
) -> None:
    from vllm.models.glm5next.nvidia.kda import Glm5NextLinearAttention
    from vllm.models.kimi_k3.nvidia.kda import _store_cache_checkpoints_kernel
    from vllm.triton_utils import triton
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    warmed: set[tuple[object, ...]] = set()
    for layer in model.modules():
        if not isinstance(layer, Glm5NextLinearAttention):
            continue
        if layer.kda_prefill_backend != "flashkda":
            continue

        conv_state, recurrent_state = layer.kv_cache
        if not layer._conv_state_dim_first:
            conv_state = conv_state.transpose(-1, -2)
        state_len = int(layer.conv_size - 1)
        width = int(3 * layer.local_projection_size)
        recurrent_row_size = int(recurrent_state[0].numel())
        key = (
            conv_state.dtype,
            recurrent_state.dtype,
            tuple(conv_state.stride()),
            int(recurrent_state.stride(0)),
            state_len,
            width,
            recurrent_row_size,
        )
        if key in warmed:
            continue

        x = torch.empty((1, width), dtype=conv_state.dtype, device=device)
        checkpoint_state = torch.empty(
            (1, *recurrent_state.shape[1:]),
            dtype=recurrent_state.dtype,
            device=device,
        )
        query_start_loc = torch.zeros(2, dtype=torch.int32, device=device)
        checkpoint_offsets = torch.zeros(1, dtype=torch.int32, device=device)
        checkpoint_state_indices = torch.full(
            (1,), NULL_BLOCK_ID, dtype=torch.int32, device=device
        )
        block_size = 256
        _store_cache_checkpoints_kernel[
            (
                1,
                triton.cdiv(
                    max(width * state_len, recurrent_row_size), block_size
                ),
            )
        ](
            x,
            conv_state,
            checkpoint_state,
            recurrent_state,
            query_start_loc,
            checkpoint_offsets,
            checkpoint_state_indices,
            x.stride(0),
            x.stride(1),
            conv_state.stride(0),
            conv_state.stride(1),
            conv_state.stride(2),
            checkpoint_state.stride(0),
            recurrent_state.stride(0),
            checkpoint_offsets.stride(0),
            state_len,
            width,
            recurrent_row_size,
            NULL_BLOCK_ID,
            block_size,
        )
        warmed.add(key)

    if warmed:
        logger.info("Warmed %d GLM-5.3 cache checkpoint kernel key(s).", len(warmed))


@torch.inference_mode()
def glm5next_triton_warmup(runner: "GPUModelRunner") -> None:
    """Warm GLM-5.3 vision Triton kernels against loaded model geometry."""
    model = runner.get_model()
    _warm_vision_rotary(model, runner.device)
    _warm_cache_checkpoint_store(model, runner.device)
    if runner.device.type == "cuda":
        torch.accelerator.synchronize(runner.device)
