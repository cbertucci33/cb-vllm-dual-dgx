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


@torch.inference_mode()
def glm5next_triton_warmup(runner: "GPUModelRunner") -> None:
    """Warm GLM-5.3 vision Triton kernels against loaded model geometry."""
    _warm_vision_rotary(runner.get_model(), runner.device)
    if runner.device.type == "cuda":
        torch.accelerator.synchronize(runner.device)
