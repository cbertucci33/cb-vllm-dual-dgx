# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from typing import Any

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature, PreTrainedConfig
from typing_extensions import TypedDict

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.multimodal import MultiModalDummyOptions
from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.models.vision import is_vit_use_data_parallel
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    ImageSize,
    MultiModalDataItems,
    VideoProcessorItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.processing.processor import HFMultiModalInputs
from vllm.transformers_utils.configs.mimo_v2_omni import Mimo_VLVisionConfig
from vllm.transformers_utils.processors.mimo_v2_omni import (
    MiMoOmniProcessor,
    VideoAudioInput,
    _format_timestamp,
    _smart_resize,
)

from .interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
)
from .mimo_audio import MimoAudioEncoder
from .mimo_v2 import MiMoV2FlashForCausalLM, _check_pending_fp8_qkv_projections
from .qwen2_5_vl import (
    Qwen2_5_VisionMLP,
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VLImageEmbeddingInputs,
    Qwen2_5_VLImageInputs,
    Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLVideoEmbeddingInputs,
    Qwen2_5_VLVideoInputs,
    Qwen2_5_VLVideoPixelInputs,
)
from .qwen2_vl import (
    Qwen2VLMultiModalDataParser,
    _create_qwen2vl_field_factory,
)
from .utils import AutoWeightsLoader, IntermediateTensors, WeightsMapper, maybe_prefix


class MiMoV2OmniAudioInputs(TypedDict):
    type: str
    audio_features: torch.Tensor | list[torch.Tensor]
    audio_token_lens: torch.Tensor | list[int] | None


class MiMoV2OmniMultiModalInputs(TypedDict, total=False):
    image: Qwen2_5_VLImageInputs | None
    video: Qwen2_5_VLVideoInputs | None
    audio: MiMoV2OmniAudioInputs | None


class MiMoVisionMLP(Qwen2_5_VisionMLP):
    def forward(self, x: torch.Tensor):
        from vllm.utils.mem_utils import check_startup_memory

        rows = x.numel() // x.shape[-1]
        gate_up = self.gate_up_proj.output_size_per_partition
        check_startup_memory(
            device_bytes=rows
            * x.element_size()
            * (gate_up + gate_up // 2 + self.down_proj.output_size),
            label="MiMo vision gate/up, activation and down projection",
            device=x.device,
        )
        return super().forward(x)


class MiMoVisionPatchEmbed(Qwen2_5_VisionPatchEmbed):
    pass


class MiMoVisionPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        spatial_merge_size: int = 2,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.ln_q = norm_layer(context_dim)

        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.0",
                return_bias=False,
                disable_tp=use_data_parallel,
            ),
            nn.GELU(),
            RowParallelLinear(
                self.hidden_size,
                d_model,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.2",
                return_bias=False,
                disable_tp=use_data_parallel,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from vllm.utils.mem_utils import check_startup_memory

        groups = x.numel() // self.hidden_size
        check_startup_memory(
            device_bytes=x.nbytes
            + groups
            * x.element_size()
            * (
                self.mlp[0].output_size_per_partition
                + self.mlp[0].output_size_per_partition
                + self.mlp[2].output_size
            ),
            label="MiMo vision merger norm, linear/GELU and output",
            device=x.device,
        )
        x = self.ln_q(x)
        x = x.view(-1, self.hidden_size)
        out = self.mlp(x)
        return out


class MiMoVisionAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        qk_channels: int,
        kv_channels: int,
        use_sink: bool = False,
        visual_token_window_size: int = 64,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.tp_size = (
            1
            if use_data_parallel
            else parallel_state.get_tensor_model_parallel_world_size()
        )
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qk_channels = qk_channels
        self.kv_channels = kv_channels
        self.embed_dim = embed_dim

        self.num_heads_per_partition = dist_utils.divide(num_heads, self.tp_size)
        self.num_kv_heads_per_partition = dist_utils.divide(num_kv_heads, self.tp_size)

        # Attention scale uses the Q/K head dimension (qk_channels)
        self.scale = qk_channels**-0.5

        # QKV: Q is (num_heads * qk_channels), KV are (num_kv_heads * kv_channels)
        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=qk_channels,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            v_head_size=kv_channels,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=use_data_parallel,
        )

        # Output projection: input is (num_heads * kv_channels) after attention
        self.proj = RowParallelLinear(
            input_size=num_heads * kv_channels,
            output_size=embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=use_data_parallel,
        )

        # For full attention (non-window blocks)
        self.attn = MMEncoderAttention(
            num_heads=self.num_heads_per_partition,
            head_size=kv_channels,
            scale=self.scale,
            num_kv_heads=self.num_kv_heads_per_partition,
            prefix=f"{prefix}.attn",
        )

        # Rotary embeddings applied separately to Q and K
        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)

        # Per-head sink logits, applied in the window attention path.
        # The checkpoint stores these only for non-full-attention blocks
        self.use_sink = use_sink
        if use_sink:
            self.sinks = nn.Parameter(
                torch.empty(num_heads),
                requires_grad=False,
            )
        else:
            self.sinks = None

        self.visual_token_window_size = visual_token_window_size

    def _forward_window_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
    ) -> torch.Tensor:
        """Window attention with the per-head sink applied to key 0.

        The reference adds ``sinks[h]`` to the logit of each sequence's first
        key, which the Triton prefill kernel supports directly, so the softmax
        normalizes over the biased scores in one pass.
        """
        from vllm.v1.attention.ops.triton_prefill_attention import (
            context_attention_fwd,
        )

        w = self.visual_token_window_size
        output = torch.empty_like(q)
        head_start = self.tp_rank * self.num_heads_per_partition
        sinks = (
            self.sinks[head_start : head_start + self.num_heads_per_partition]
            if self.sinks is not None
            else None
        )
        context_attention_fwd(
            q,
            k,
            v,
            output,
            b_start_loc=cu_seqlens[:-1],
            b_seq_len=cu_seqlens[1:] - cu_seqlens[:-1],
            max_input_len=max_seqlen,
            is_causal=False,
            softmax_scale=self.scale,
            sliding_window_q=w,
            sliding_window_k=w,
            sinks=sinks,
            sinks_bias_key0=True,
        )
        return output

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        full_attn: bool = True,
    ) -> torch.Tensor:
        """Args:
        x: [seq_len, batch=1, embed_dim]  (seq-first convention)
        cu_seqlens: cumulative sequence lengths [num_seqs+1], int32
        rotary_pos_emb_cos: [seq_len, qk_channels // 2]
        rotary_pos_emb_sin: [seq_len, qk_channels // 2]
        max_seqlen: maximum sequence length
        full_attn: if True, full attention; if False, window attention

        """
        # [seq_len, 1, embed_dim] -> QKV projection
        from vllm.utils.mem_utils import check_startup_memory

        rows = x.numel() // x.shape[-1]
        q_width = self.num_heads_per_partition * self.qk_channels
        k_width = self.num_kv_heads_per_partition * self.qk_channels
        v_width = self.num_kv_heads_per_partition * self.kv_channels
        context_width = self.num_heads_per_partition * self.kv_channels
        check_startup_memory(
            device_bytes=rows
            * x.element_size()
            * (
                q_width
                + k_width
                + v_width
                + q_width
                + k_width
                + context_width
                + context_width
                + self.embed_dim
            ),
            label="MiMo vision QKV, rotary, context/layout and output",
            device=x.device,
        )
        qkv, _ = self.qkv(x)  # [seq_len, 1, q_size + kv_size + kv_size]
        seq_len, batch_size, _ = qkv.shape

        q_size = self.num_heads_per_partition * self.qk_channels
        kv_size = self.num_kv_heads_per_partition * self.kv_channels
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Rearrange to [batch, seq, head, head_dim] for rotary application
        q = einops.rearrange(q, "s b (h d) -> b s h d", h=self.num_heads_per_partition)
        k = einops.rearrange(
            k, "s b (h d) -> b s h d", h=self.num_kv_heads_per_partition
        )
        v = einops.rearrange(
            v, "s b (h d) -> b s h d", h=self.num_kv_heads_per_partition
        )

        # Apply rotary embeddings to Q and K independently (handles GQA)
        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            q = self.apply_rotary_emb(q, rotary_pos_emb_cos, rotary_pos_emb_sin)
            k = self.apply_rotary_emb(k, rotary_pos_emb_cos, rotary_pos_emb_sin)

        if full_attn:
            # Full attention via MMEncoderAttention
            # Flatten to [batch, seq, heads * head_dim]
            q_flat = q.reshape(batch_size, seq_len, -1)
            k_flat = k.reshape(batch_size, seq_len, -1)
            v_flat = v.reshape(batch_size, seq_len, -1)
            context_layer = self.attn(
                query=q_flat,
                key=k_flat,
                value=v_flat,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            # context_layer: [batch, seq, num_heads, head_dim] or [batch, seq, hidden]
            # Ensure shape is [seq, batch, num_heads * kv_channels]
            if context_layer.dim() == 4:
                context_layer = einops.rearrange(
                    context_layer, "b s h d -> s b (h d)"
                ).contiguous()
            else:
                context_layer = einops.rearrange(
                    context_layer, "b s d -> s b d"
                ).contiguous()
        else:
            # Window attention via flash_attn_varlen_func with window_size
            # Flatten batch dimension: [seq, head, head_dim]
            q_varlen = einops.rearrange(q, "b s h d -> (b s) h d")
            k_varlen = einops.rearrange(k, "b s h d -> (b s) h d")
            v_varlen = einops.rearrange(v, "b s h d -> (b s) h d")
            output = self._forward_window_attn(
                q_varlen, k_varlen, v_varlen, cu_seqlens, max_seqlen
            )
            # output: [total_tokens, num_heads, kv_channels]
            context_layer = einops.rearrange(
                output, "(b s) h d -> s b (h d)", b=batch_size
            ).contiguous()

        output, _ = self.proj(context_layer)
        return output


class MiMoVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        qk_channels: int,
        kv_channels: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        norm_eps: float = 1e-6,
        use_sink: bool = False,
        visual_token_window_size: int = 64,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        self.attn = MiMoVisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qk_channels=qk_channels,
            kv_channels=kv_channels,
            use_sink=use_sink,
            visual_token_window_size=visual_token_window_size,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = MiMoVisionMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_fn=act_fn,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        full_attn: bool = True,
    ) -> torch.Tensor:
        # x: [seq_len, batch=1, dim]
        x_attn = self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
            full_attn=full_attn,
        )
        # Fused residual add + norm2
        x_norm, residual = self.norm2(x, residual=x_attn)
        x = residual + self.mlp(x_norm)
        return x


class MiMoVisionTransformer(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            "mlp.gate_proj": ("mlp.gate_up_proj", 0),
            "mlp.up_proj": ("mlp.gate_up_proj", 1),
        }
    )

    def __init__(
        self,
        vision_cfg: PreTrainedConfig,
        *,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.spatial_merge_size = vision_cfg.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.fullatt_block_indexes = vision_cfg.fullatt_block_indexes
        self.vit_window_attn_types = vision_cfg.vit_window_attn_types
        self.visual_token_window_size = vision_cfg.visual_token_window_size
        self.hidden_size = vision_cfg.hidden_size
        self.num_heads = vision_cfg.num_heads
        self.num_kv_heads = vision_cfg.num_key_value_heads
        self.qk_channels = vision_cfg.qk_channels
        self.kv_channels = vision_cfg.kv_channels

        self.patch_embed = MiMoVisionPatchEmbed(
            patch_size=vision_cfg.patch_size,
            temporal_patch_size=vision_cfg.temporal_patch_size,
            in_channels=vision_cfg.in_channels,
            hidden_size=vision_cfg.hidden_size,
        )

        norm_layer = partial(RMSNorm, eps=norm_eps)

        # Rotary embedding for 2D positions.
        # With partial_rotary_factor=0.5 and head_size=qk_channels:
        #   rotary_dim = qk_channels // 2
        #   get_cos_sin returns cos, sin each of shape [pos, rotary_dim // 2]
        # After indexing with 2D pos_ids and flattening:
        #   result shape = [tokens, rotary_dim] = [tokens, qk_channels // 2]
        # which is what ApplyRotaryEmb expects as cos/sin input.
        self.rotary_pos_emb = get_rope(
            head_size=vision_cfg.qk_channels,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={"partial_rotary_factor": 0.5},
        )

        self.blocks = nn.ModuleList(
            [
                MiMoVisionBlock(
                    dim=vision_cfg.hidden_size,
                    num_heads=vision_cfg.num_heads,
                    num_kv_heads=vision_cfg.num_key_value_heads,
                    qk_channels=vision_cfg.qk_channels,
                    kv_channels=vision_cfg.kv_channels,
                    mlp_hidden_dim=vision_cfg.intermediate_size,
                    act_fn=get_act_and_mul_fn(vision_cfg.hidden_act),
                    norm_eps=norm_eps,
                    use_sink=(
                        vision_cfg.use_sink
                        and i not in vision_cfg.fullatt_block_indexes
                    ),
                    visual_token_window_size=vision_cfg.visual_token_window_size,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{i}",
                )
                for i in range(vision_cfg.depth)
            ]
        )

        self.merger = MiMoVisionPatchMerger(
            d_model=vision_cfg.out_hidden_size,
            context_dim=vision_cfg.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=vision_cfg.spatial_merge_size,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def apply_index(self, tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Reindex tensor at the spatial_merge_unit granularity."""
        tensor = tensor.unflatten(0, (-1, self.spatial_merge_unit))
        from vllm.utils.mem_utils import check_startup_memory

        check_startup_memory(
            device_bytes=tensor.nbytes,
            label="MiMo vision row/column layout copy",
            device=tensor.device,
        )
        tensor = tensor[index]
        tensor = tensor.flatten(0, 1)
        return tensor

    def get_window_index_1d(
        self, grid_thw: torch.Tensor, col: bool = True
    ) -> torch.Tensor:
        """Compute 1D window indices for col-based or row-based SWA reordering."""
        window_index: list[torch.Tensor] = []
        window_index_id = 0
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            index_new = index.transpose(1, 2).reshape(-1) if col else index.reshape(-1)
            window_index.append(index_new + window_index_id)
            window_index_id += int((grid_t * llm_grid_h * llm_grid_w).item())
        return torch.cat(window_index, dim=0)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute 2D rotary position embedding cos/sin for given grid sizes.

        Returns:
            cos: [total_tokens, qk_channels // 2]
            sin: [total_tokens, qk_channels // 2]

        """
        cos_list, sin_list = [], []
        for i in range(grid_thw.size(0)):
            t, h, w = int(grid_thw[i, 0]), int(grid_thw[i, 1]), int(grid_thw[i, 2])

            # Build 2D position IDs with spatial_merge_size interleaving
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = (
                hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = (
                wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
            # pos_ids: [t*h*w, 2]

            max_grid_size = max(h, w)
            # get_cos_sin returns cos, sin each of shape [max_grid_size, rotary_dim//2]
            # where rotary_dim = qk_channels // 2 (from partial_rotary_factor=0.5)
            cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)

            # [t*h*w, 2, rotary_dim//2] -> [t*h*w, rotary_dim] (= qk_channels // 2)
            cos_img = cos[pos_ids].flatten(1)
            sin_img = sin[pos_ids].flatten(1)
            cos_list.append(cos_img)
            sin_list.append(sin_img)

        return torch.cat(cos_list, dim=0), torch.cat(sin_list, dim=0)

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [total_tokens, C] pre-flattened patches
            grid_thw: [num_images, 3] tensor of (t, h, w) for each image/video
        Returns:
            [merged_tokens, out_hidden_size]

        """
        # Ensure grid_thw is a tensor
        if not isinstance(grid_thw, torch.Tensor):
            grid_thw = torch.tensor(grid_thw, dtype=torch.long)

        # Move to visual model device/dtype
        x = x.to(device=self.device, dtype=self.dtype)

        # Patch embedding: [total_tokens, hidden_size]
        x = self.patch_embed(x)

        # Compute 2D rotary positional embeddings
        # cos, sin: [total_tokens, qk_channels // 2]
        rotary_cos, rotary_sin = self.rot_pos_emb(grid_thw)
        rotary_cos = rotary_cos.to(device=x.device)
        rotary_sin = rotary_sin.to(device=x.device)

        # Compute cu_seqlens for flash_attn (per-image/video sequence lengths)
        seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        )
        cu_seqlens = torch.cat(
            [
                torch.tensor([0], device=x.device, dtype=torch.int32),
                seqlens.cumsum(dim=0).to(device=x.device, dtype=torch.int32),
            ]
        )
        max_seqlen = seqlens.max()

        # Precompute col-based window index for type=1 (col SWA) layers
        window_index_1d_col = self.get_window_index_1d(grid_thw, col=True).to(
            device=x.device
        )
        reverse_window_index_1d_col = torch.argsort(window_index_1d_col)

        # Col-based rotary embeddings (reordered at spatial_merge_unit granularity).
        # apply_index reorders groups of spatial_merge_unit tokens, just like x.
        col_cos = self.apply_index(rotary_cos, window_index_1d_col)
        col_sin = self.apply_index(rotary_sin, window_index_1d_col)

        # Add batch dimension: [total_tokens, 1, hidden_size]
        x = x.unsqueeze(1)

        for i, blk in enumerate(self.blocks):
            window_attn_type = self.vit_window_attn_types[i]

            # Reorder tokens to col-based layout when entering col-SWA region
            if window_attn_type == 1 and (
                i == 0 or self.vit_window_attn_types[i - 1] != 1
            ):
                x = self.apply_index(x, window_index_1d_col)

            # Restore row-based order when leaving col-SWA region
            if (
                i > 0
                and window_attn_type != 1
                and self.vit_window_attn_types[i - 1] == 1
            ):
                x = self.apply_index(x, reverse_window_index_1d_col)

            # Use col-based embeddings for col-SWA layers
            cos_now = col_cos if window_attn_type == 1 else rotary_cos
            sin_now = col_sin if window_attn_type == 1 else rotary_sin

            full_attn = i in self.fullatt_block_indexes
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=cos_now,
                rotary_pos_emb_sin=sin_now,
                max_seqlen=max_seqlen,
                full_attn=full_attn,
            )

        # Restore row-based order if last block was col-SWA
        if self.vit_window_attn_types[-1] == 1:
            x = self.apply_index(x, reverse_window_index_1d_col)

        # Remove batch dim and merge spatial tokens
        # x: [total_tokens, 1, hidden_size] -> [total_tokens, hidden_size]
        x = x.squeeze(1)
        x = self.merger(x)
        return x

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class MiMoV2OmniMultiModalDataParser(Qwen2VLMultiModalDataParser):
    # Video stays raw: get_video_replacement also reads second_per_grid_ts and
    # video_start_times, which a metadata-only rewrite would not carry.
    embedding_fields = {
        "image": Qwen2VLMultiModalDataParser.embedding_fields["image"],
    }


class MiMoVideoProcessorItems(VideoProcessorItems):
    def get_item_for_hash(self, index: int) -> Any:
        item = self.data[index]
        if isinstance(item, VideoAudioInput):
            return vars(item)
        metadata = (
            self.metadata[index] if isinstance(self.metadata, list) else self.metadata
        )
        return (
            {"frames": super().get_item_for_hash(index), "metadata": metadata}
            if metadata is not None
            else super().get_item_for_hash(index)
        )

    def get_processor_data(self) -> Mapping[str, object]:
        result = []
        for index in range(self.get_count()):
            item = self.get(index)
            metadata = (
                self.metadata[index]
                if isinstance(self.metadata, list)
                else self.metadata
            )
            if metadata is not None and not isinstance(item, VideoAudioInput):
                item = (item, metadata)
            result.append(item)
        return {"videos": result}

    def get_num_frames(self, item_idx: int) -> int:
        item = self.get(item_idx)
        if not isinstance(item, VideoAudioInput):
            return super().get_num_frames(item_idx)
        frames = item.video[0] if isinstance(item.video, tuple) else item.video
        return len(frames)

    def get_frame_size(self, item_idx: int) -> ImageSize:
        item = self.get(item_idx)
        if not isinstance(item, VideoAudioInput):
            return super().get_frame_size(item_idx)
        frames = item.video[0] if isinstance(item.video, tuple) else item.video
        image = frames[0]
        if image.shape[-1] in (1, 3, 4):
            height, width = image.shape[:2]
        else:
            _, height, width = image.shape
        return ImageSize(width=width, height=height)


class MiMoDataParser(MiMoV2OmniMultiModalDataParser):
    def _parse_video_data(self, data):
        items = [data] if isinstance(data, VideoAudioInput) else data
        if isinstance(items, (list, tuple)) and any(
            isinstance(item, VideoAudioInput) for item in items
        ):
            normalized: list[Any] = []
            metadata: list[dict[str, Any] | None] = []
            for item in items:
                if isinstance(item, VideoAudioInput):
                    normalized.append(item)
                    metadata.append(None)
                else:
                    parsed = super()._parse_video_data([item])
                    if not isinstance(parsed, VideoProcessorItems):
                        raise ValueError(
                            "Mixed MiMo video input requires decoded frames"
                        )
                    normalized.append(parsed.get(0))
                    metadata.append(
                        parsed.metadata[0]
                        if isinstance(parsed.metadata, list)
                        else parsed.metadata
                    )
            return MiMoVideoProcessorItems(normalized, metadata=metadata)
        parsed = super()._parse_video_data(data)
        if isinstance(parsed, VideoProcessorItems):
            return MiMoVideoProcessorItems(parsed.data, metadata=parsed.metadata)
        return parsed


class MiMoV2OmniProcessingInfo(BaseProcessingInfo):
    profile_each_modality = True

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None, "image": None, "video": None}

    def get_hf_config(self):
        config = self.ctx.get_hf_config()
        if isinstance(config.vision_config, dict):
            config.vision_config = Mimo_VLVisionConfig.from_dict(config.vision_config)
        return config

    def get_hf_processor(self, **kwargs: object) -> MiMoOmniProcessor:
        hf_config = self.get_hf_config()
        tokenizer = self.get_tokenizer()
        baseline = MiMoOmniProcessor.from_hf_config(
            tokenizer, hf_config, **self.ctx.get_merged_mm_kwargs({})
        )
        candidate = baseline.with_overrides(dict(kwargs))
        if kwargs:
            original = baseline.mimo_processor
            requested = candidate.mimo_processor
            for modality, fields in (
                ("_img_kw", ("max_pixels",)),
                ("_vid_kw", ("max_pixels", "total_max_pixels", "max_frames")),
            ):
                before = getattr(original, modality)
                after = getattr(requested, modality)
                for name in fields:
                    if before[name] is not None and (
                        after[name] is None or after[name] > before[name]
                    ):
                        raise ValueError(
                            f"Request {name} exceeds profiled MiMo bounds; "
                            "declare it in startup mm_processor_kwargs"
                        )
            for name, value in candidate._constructor_kwargs.items():
                if (
                    name.startswith("audio_")
                    and value != baseline._constructor_kwargs[name]
                ):
                    raise ValueError(
                        f"Request {name} differs from profiled MiMo audio settings"
                    )
        return candidate

    def get_image_processor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).image_processor

    def get_data_parser(self):
        # Without embedding_fields the EC producer publishes no metadata and
        # the EPD proxy never rewrites the media item.
        return MiMoDataParser(
            self.get_hf_config().vision_config.spatial_merge_size,
            target_sr=24000.0,
            expected_hidden_size=self._get_expected_hidden_size(),
            allow_missing_mm_embeddings=self.allow_missing_mm_embeddings,
        )

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        limits = {}
        if mm_counts.get("image", 0):
            limits["image"] = self.get_max_image_tokens()
        if mm_counts.get("video", 0):
            limits["video"] = self.get_max_video_tokens(seq_len, mm_counts)
        if mm_counts.get("audio", 0):
            limits["audio"] = self._get_audio_profile(seq_len)[1]
        return limits

    def _get_audio_profile(self, seq_len: int) -> tuple[int, int]:
        """Return PCM samples and feature tokens for public audio profiling.

        AudioMediaIO already enforces the decoder duration limit. This only
        sizes profiling inputs; it does not add a limit to audio processing.
        """
        processor = self.get_hf_processor().mimo_processor
        duration = envs.VLLM_MAX_AUDIO_DECODE_DURATION_S
        if duration <= 0:
            raise ValueError(
                "MiMo audio profiling requires a positive decoder duration"
            )
        # Each input includes independent audio-start/audio-end wrapper tokens.
        token_budget = seq_len - 2
        if token_budget <= 0:
            raise ValueError("MiMo audio profiling needs room for audio feature tokens")

        def token_count(samples: int) -> int:
            # Match centered MelSpectrogram, convolution, pooling and grouping
            # in MiMoVLProcessor.preprocess_audio without allocating a waveform.
            frames = samples // processor.audio_hop_length + 1
            encoded = frames + 3 - processor.audio_kernel_size
            encoded = (
                encoded + 2 - processor.audio_kernel_size
            ) // processor.audio_stride_size + 1
            pooled = math.ceil(encoded / processor.audio_avg_pooler)
            return math.ceil(pooled / processor.audio_group_size)

        low, high = 0, duration * processor.audio_sampling_rate
        while low < high:
            middle = (low + high + 1) // 2
            if token_count(middle) <= token_budget:
                low = middle
            else:
                high = middle - 1
        return low, token_count(low)

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 1,
        do_resize: bool = True,
        image_processor,
        mm_kwargs: Mapping[str, object],
        is_video: bool = False,
    ) -> tuple[ImageSize, int]:
        processor = self.get_hf_processor(**mm_kwargs).mimo_processor
        unit = processor.patch_size * processor.merge_size
        if is_video:
            minimum = processor._vid_kw["min_pixels"]
            if not isinstance(minimum, int):
                raise ValueError("MiMo minimum pixel bounds must be integers")
            maximum = processor.video_pixel_budget(num_frames)
        else:
            minimum = processor._img_kw["min_pixels"]
            maximum = processor._img_kw["max_pixels"]
        if do_resize:
            height, width = _smart_resize(
                image_height, image_width, unit, minimum, maximum
            )
        else:
            height, width = image_height, image_width
        temporal = processor.temporal_patch_size * processor.temporal_compression_ratio
        grid_t = (
            math.ceil(num_frames / temporal) * processor.temporal_compression_ratio
            if is_video
            else 1
        )
        tokens = grid_t * (height // unit) * (width // unit)
        return ImageSize(width=width, height=height), tokens

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor,
        mm_kwargs: Mapping[str, object],
    ) -> int:
        _, num_image_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            num_frames=1,
            image_processor=image_processor,
            mm_kwargs=mm_kwargs,
        )
        return num_image_tokens

    def get_num_video_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int,
        image_processor,
        mm_kwargs: Mapping[str, object],
    ) -> int:
        _, num_video_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            num_frames=num_frames,
            image_processor=image_processor,
            mm_kwargs=mm_kwargs,
            is_video=True,
        )
        return num_video_tokens

    def get_image_size_with_most_features(
        self, max_pixels: int | None = None
    ) -> ImageSize:
        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size

        if max_pixels is None:
            max_pixels = self.get_hf_processor().mimo_processor._img_kw["max_pixels"]

        unit = patch_size * merge_size
        max_seq_len = max_pixels // (unit * unit)

        def closest_factor_pair(n: int) -> tuple[int, int]:
            for d in range(math.isqrt(n), 0, -1):
                if n % d == 0:
                    return d, n // d
            return 1, n

        height_factor, width_factor = 1, max_seq_len
        for seq_len in range(max_seq_len, 0, -1):
            height_factor, width_factor = closest_factor_pair(seq_len)
            if width_factor / height_factor <= 200:
                break

        return ImageSize(width=unit * width_factor, height=unit * height_factor)

    def get_max_image_tokens(self) -> int:
        image_processor = self.get_image_processor()
        target_width, target_height = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
            image_processor=image_processor,
            mm_kwargs={},
        )

    def _get_max_video_frames(self, max_tokens: int, start_num_frames: int = 1) -> int:
        image_processor = self.get_image_processor()
        target_width, target_height = self.get_image_size_with_most_features()
        num_frames = start_num_frames
        while True:
            next_num_frames = num_frames + 1
            next_max_tokens = self.get_num_video_tokens(
                image_width=target_width,
                image_height=target_height,
                num_frames=next_num_frames,
                image_processor=image_processor,
                mm_kwargs={},
            )
            if next_max_tokens > max_tokens:
                break
            num_frames = next_num_frames
        return num_frames

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        max_frames_per_video: int = 14,
    ) -> int:
        max_videos = mm_counts.get("video", 0)
        max_total_frames = self._get_max_video_frames(seq_len)
        max_frames_per_video = min(
            max_total_frames // max(max_videos, 1), max_frames_per_video
        )
        return max(max_frames_per_video, 1)

    def get_max_video_tokens(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        image_processor = self.get_image_processor()
        target_width, target_height = self.get_image_size_with_most_features()
        return self.get_num_video_tokens(
            image_width=target_width,
            image_height=target_height,
            num_frames=self.get_num_frames_with_most_features(seq_len, mm_counts),
            image_processor=image_processor,
            mm_kwargs={},
        )


class MiMoV2OmniMultiModalProcessor(BaseMultiModalProcessor[MiMoV2OmniProcessingInfo]):
    """vLLM multimodal processor for MiMo-Omni (image + video).

    Key differences from Qwen2.5-VL:
    - Videos use timestamp tokens between temporal grid positions.
    - The HF processor expects ``(TCHW_tensor, timestamps_T_tensor)`` video
      tuples rather than plain numpy arrays.
    - ``video_start_times`` is tracked so prompt-update reconstruction can
      regenerate the exact same timestamp token IDs.
    """

    # fps assumed for vllm-decoded video (numpy T,H,W,C arrays).
    # The video loader samples ~32 frames; treat each frame as 1 s apart so
    # MiMoVLProcessor sees 1 fps input and resamples internally.
    _INPUT_FPS: float = 1.0

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        merge_size = self.info.get_hf_config().vision_config.spatial_merge_size
        fields: dict[str, MultiModalFieldConfig] = dict(
            **_create_qwen2vl_field_factory(merge_size)(hf_inputs),
            second_per_grid_ts=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
            video_start_times=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
            video_grid_timestamps=MultiModalFieldConfig.batched(
                "video", keep_on_cpu=True
            ),
            audio_features=MultiModalFieldConfig.batched("audio"),
            audio_token_lens=MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
        )
        # video_audio fields: only present when video_audio content was processed
        if "video_audio_n_segs" in hf_inputs:
            fields["video_audio_n_segs"] = MultiModalFieldConfig.batched(
                "video", keep_on_cpu=True
            )
        # video_audio_seg_lens: list of per-video 1D tensors, batched("video")
        if "video_audio_seg_lens" in hf_inputs:
            fields["video_audio_seg_lens"] = MultiModalFieldConfig.batched(
                "video", keep_on_cpu=True
            )
        if "video_audio_frame_counts" in hf_inputs:
            fields["video_audio_frame_counts"] = MultiModalFieldConfig.batched(
                "video", keep_on_cpu=True
            )
        if "video_audio_start_offsets" in hf_inputs:
            fields["video_audio_start_offsets"] = MultiModalFieldConfig.batched(
                "video", keep_on_cpu=True
            )
        if "va_audio_features" in hf_inputs:
            fields["va_audio_features"] = MultiModalFieldConfig.batched("video")
        return fields

    def _get_hf_mm_inputs(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> HFMultiModalInputs:
        """Convert numpy video arrays to (TCHW, timestamps) tuples for MiMo."""
        hf_inputs = super()._get_hf_mm_inputs(mm_items, hf_kwargs)
        mm_data = hf_inputs.hf_data

        if "videos" in mm_data:
            converted = []
            videos = mm_data["videos"]
            if not isinstance(videos, (list, tuple)):
                raise ValueError("MiMo processor video items must be a sequence")
            for item in videos:
                metadata = None
                video = item.video if isinstance(item, VideoAudioInput) else item
                if (
                    isinstance(video, tuple)
                    and len(video) == 2
                    and isinstance(video[0], torch.Tensor)
                    and isinstance(video[1], torch.Tensor)
                ):
                    pair = video
                else:
                    if isinstance(video, tuple) and len(video) == 2:
                        video, metadata = video
                        if not isinstance(metadata, Mapping):
                            raise ValueError("MiMo video metadata must be a mapping")
                    frames = (
                        torch.as_tensor(np.asarray(video))
                        if not isinstance(video, torch.Tensor)
                        else video
                    )
                    if frames.ndim != 4:
                        raise ValueError("MiMo decoded video must have four dimensions")
                    if frames.shape[-1] in (1, 3, 4):
                        frames = frames.permute(0, 3, 1, 2)
                    frames = frames.float()
                    if isinstance(video, tuple):
                        raise ValueError("Unresolved MiMo video payload")
                    if metadata is not None:
                        if (
                            "frames_indices" not in metadata
                            or not metadata.get("fps", 0) > 0
                        ):
                            raise ValueError(
                                "MiMo video metadata needs frame indices and FPS"
                            )
                        timestamps = torch.as_tensor(
                            metadata["frames_indices"], dtype=torch.float32
                        ) / float(metadata["fps"])
                        if timestamps.numel() != frames.shape[0]:
                            raise ValueError(
                                "MiMo video timestamp/frame counts disagree"
                            )
                    else:
                        timestamps = (
                            torch.arange(frames.shape[0], dtype=torch.float32)
                            / self._INPUT_FPS
                        )
                    pair = (frames, timestamps)
                converted.append(
                    replace(item, video=pair)
                    if isinstance(item, VideoAudioInput)
                    else pair
                )
            mm_data["videos"] = converted

        return hf_inputs

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        hf_config = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        merge_size = hf_config.vision_config.spatial_merge_size
        p = hf_processor.mimo_processor

        image_pad_id = vocab[hf_processor.image_token]
        video_pad_id = vocab[hf_processor.video_token]
        audio_pad_id = vocab.get("<|audio_pad|>")
        vision_start_id = p.vision_start_token_id
        vision_end_id = p.vision_end_token_id
        video_start_id = p.video_start_token_id
        video_end_id = p.video_end_token_id
        audio_start_id = p.audio_start_token_id
        audio_end_id = p.audio_end_token_id
        assert isinstance(vision_start_id, int)
        assert isinstance(vision_end_id, int)
        assert isinstance(video_start_id, int)
        assert isinstance(video_end_id, int)

        def get_image_replacement(item_idx: int) -> PromptUpdateDetails:
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            n_tokens = int(grid_thw.prod()) // merge_size**2
            return PromptUpdateDetails.from_seq([image_pad_id] * n_tokens)

        def get_video_replacement(item_idx: int) -> PromptUpdateDetails:
            out_item = out_mm_kwargs["video"][item_idx]
            grid_thw = out_item["video_grid_thw"].data
            second_per_grid_ts = out_item["second_per_grid_ts"].data
            video_start_time = out_item["video_start_times"].data
            assert isinstance(grid_thw, torch.Tensor)
            assert isinstance(second_per_grid_ts, (int, float, torch.Tensor))
            assert isinstance(video_start_time, (int, float, torch.Tensor))
            spt = float(second_per_grid_ts)
            start = float(video_start_time)
            timestamp_field = out_item.get("video_grid_timestamps")
            grid_timestamps = None
            if timestamp_field is not None:
                timestamps = timestamp_field.data
                if not isinstance(timestamps, torch.Tensor):
                    raise ValueError("MiMo video timestamps must be a tensor")
                grid_timestamps = timestamps.tolist()

            T, H, W = map(int, grid_thw)
            n_per_grid = H * W // (merge_size * merge_size)

            # Check if this is a video_audio item
            n_segs_field = out_item.get("video_audio_n_segs")
            if n_segs_field is None:
                n_segs_val = 0
            else:
                n_segs_data = n_segs_field.data
                assert isinstance(n_segs_data, (int, torch.Tensor))
                n_segs_val = int(n_segs_data)
            va_seg_lens: list[int] | None = None
            if n_segs_val > 0:
                seg_lens_field = out_item.get("video_audio_seg_lens")
                if seg_lens_field is not None:
                    seg_lens_data = seg_lens_field.data
                    assert isinstance(seg_lens_data, torch.Tensor)
                    va_seg_lens = seg_lens_data[:n_segs_val].tolist()

            full: list[int] = [video_start_id]
            is_embed_mask: list[bool] = [False]

            if va_seg_lens is None:
                # Regular video: timestamp + vision tokens per grid
                for j in range(T):
                    ts_text = _format_timestamp(
                        grid_timestamps[j]
                        if grid_timestamps is not None
                        else start + j * spt
                    )
                    ts_ids = tokenizer.encode(ts_text, add_special_tokens=False)
                    full.extend(ts_ids)
                    is_embed_mask.extend([False] * len(ts_ids))
                    full.append(vision_start_id)
                    is_embed_mask.append(False)
                    full.extend([video_pad_id] * n_per_grid)
                    is_embed_mask.extend([True] * n_per_grid)
                    full.append(vision_end_id)
                    is_embed_mask.append(False)
            else:
                # video_audio: interleaved vision+audio per group
                n_groups = len(va_seg_lens)
                frame_counts = out_item["video_audio_frame_counts"].data
                assert isinstance(frame_counts, torch.Tensor)
                frame_counts = frame_counts[:n_groups].tolist()
                if sum(frame_counts) != T:
                    raise ValueError("MiMo video/audio frame groups do not cover video")
                frame0 = 0
                for g, frames_per_group in enumerate(frame_counts):
                    # Timestamp for first frame of this group
                    ts_text = _format_timestamp(
                        grid_timestamps[frame0]
                        if grid_timestamps is not None
                        else start + frame0 * spt
                    )
                    ts_ids = tokenizer.encode(ts_text, add_special_tokens=False)
                    full.extend(ts_ids)
                    is_embed_mask.extend([False] * len(ts_ids))
                    # Vision tokens for all frames in this group
                    for f in range(frames_per_group):
                        full.append(vision_start_id)
                        is_embed_mask.append(False)
                        full.extend([video_pad_id] * n_per_grid)
                        is_embed_mask.extend([True] * n_per_grid)
                        full.append(vision_end_id)
                        is_embed_mask.append(False)
                    # Audio tokens for this group
                    seg_len = va_seg_lens[g]
                    assert audio_start_id is not None
                    assert audio_pad_id is not None
                    assert audio_end_id is not None
                    full.append(audio_start_id)
                    is_embed_mask.append(False)
                    full.extend([audio_pad_id] * seg_len)
                    is_embed_mask.extend([True] * seg_len)
                    full.append(audio_end_id)
                    is_embed_mask.append(False)
                    frame0 += frames_per_group

            full.append(video_end_id)
            is_embed_mask.append(False)

            embed_t = torch.tensor(is_embed_mask)
            return PromptUpdateDetails(
                full=full,
                is_embed=lambda _seq: embed_t,
            )

        def get_audio_replacement(item_idx: int) -> PromptUpdateDetails:
            out_item = out_mm_kwargs["audio"][item_idx]
            audio_token_lens = out_item["audio_token_lens"].data
            assert isinstance(audio_token_lens, (int, torch.Tensor))
            assert audio_pad_id is not None
            tok_len = int(audio_token_lens)
            return PromptUpdateDetails.from_seq([audio_pad_id] * tok_len)

        updates: list[PromptUpdate] = [
            PromptReplacement(
                modality="image",
                target=[image_pad_id],
                replacement=get_image_replacement,
            ),
            PromptReplacement(
                modality="video",
                target=[video_pad_id],
                replacement=get_video_replacement,
            ),
        ]
        if audio_pad_id is not None and audio_start_id is not None:
            updates.append(
                PromptReplacement(
                    modality="audio",
                    target=[audio_pad_id],
                    replacement=get_audio_replacement,
                )
            )
        return updates


class MiMoV2OmniDummyInputsBuilder(BaseDummyInputsBuilder[MiMoV2OmniProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        num_audios = mm_counts.get("audio", 0)
        image_ph = "<|vision_start|><|image_pad|><|vision_end|>"
        video_ph = "<|vision_start|><|video_pad|><|vision_end|>"
        audio_ph = "<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>"
        return image_ph * num_images + video_ph * num_videos + audio_ph * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> MultiModalDataDict:
        target_width, target_height = self.info.get_image_size_with_most_features()
        target_num_frames = self.info.get_num_frames_with_most_features(
            seq_len, mm_counts
        )
        video_size = ImageSize(width=target_width, height=target_height)

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            ),
            "video": self._get_dummy_videos(
                width=video_size.width,
                height=video_size.height,
                num_frames=target_num_frames,
                num_videos=mm_counts.get("video", 0),
                overrides=mm_options.get("video"),
            ),
            "audio": self._get_dummy_audios(
                length=(
                    self.info._get_audio_profile(seq_len)[0]
                    if mm_counts.get("audio", 0)
                    else 0
                ),
                num_audios=mm_counts.get("audio", 0),
                overrides=mm_options.get("audio"),
            ),
        }


def _interleave_video_audio_embeddings(
    video: torch.Tensor,
    audio: torch.Tensor,
    num_grids: int,
    frame_counts: list[int],
    audio_lengths: list[int],
) -> torch.Tensor:
    if (
        len(frame_counts) != len(audio_lengths)
        or sum(frame_counts) != num_grids
        or sum(audio_lengths) != audio.shape[0]
        or num_grids <= 0
        or video.shape[0] % num_grids
        or any(count <= 0 for count in frame_counts)
        or any(length < 0 for length in audio_lengths)
    ):
        raise ValueError("Invalid MiMo video/audio interleave boundaries")
    from vllm.utils.mem_utils import check_startup_memory

    check_startup_memory(
        device_bytes=video.nbytes + audio.nbytes,
        label="MiMo combined video/audio embedding assembly",
        device=video.device,
    )
    video_groups = video.split(
        [count * (video.shape[0] // num_grids) for count in frame_counts]
    )
    audio_groups = audio.split(audio_lengths)
    return torch.cat(
        [item for pair in zip(video_groups, audio_groups) for item in pair], dim=0
    )


@MULTIMODAL_REGISTRY.register_processor(
    MiMoV2OmniMultiModalProcessor,
    info=MiMoV2OmniProcessingInfo,
    dummy_inputs=MiMoV2OmniDummyInputsBuilder,
)
class MiMoV2OmniForCausalLM(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsQuant, SupportsEagle3
):
    packed_modules_mapping = dict(MiMoV2FlashForCausalLM.packed_modules_mapping)

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # audio encoder
            "speech_embeddings.": "audio_encoder.speech_embeddings.",
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            # mapping for original checkpoint
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
            "audio_tokenizer.": None,
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        if modality.startswith("audio"):
            return "<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>"

        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        vllm_config.cache_config.require_kv_cache_scales = True
        config = vllm_config.model_config.hf_config
        self.config = config
        # Omni ViT/Audio Encoder BF16
        vision_config = (
            Mimo_VLVisionConfig.from_dict(config.vision_config)
            if isinstance(config.vision_config, dict)
            else config.vision_config
        )
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = MiMoVisionTransformer(
                vision_config,
                norm_eps=getattr(vllm_config, "rms_norm_eps", 1e-6),
                quant_config=None,
                prefix=maybe_prefix(prefix, "visual"),
            )
        audio_config = getattr(config, "audio_config", None)
        model_path = vllm_config.model_config.model
        self.audio_encoder: MimoAudioEncoder | None
        if audio_config is not None:
            with self._mark_tower_model(vllm_config, "audio"):
                self.audio_encoder = MimoAudioEncoder(
                    audio_config, model_path=model_path
                )
        else:
            self.audio_encoder = None
        with self._mark_language_model(vllm_config):
            self.language_model = MiMoV2FlashForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Qwen2_5_VLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return Qwen2_5_VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        assert image_embeds is not None
        return Qwen2_5_VLImageEmbeddingInputs(
            type="image_embeds",
            image_embeds=image_embeds,
            image_grid_thw=image_grid_thw,
        )

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> Qwen2_5_VLVideoInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            return Qwen2_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )

        assert video_embeds is not None
        return Qwen2_5_VLVideoEmbeddingInputs(
            type="video_embeds",
            video_embeds=video_embeds,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )

    def _process_image_input(
        self, image_input: Qwen2_5_VLImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"]
            image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(
        self, video_input: Qwen2_5_VLVideoInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"]
            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw_list)

        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return video_embeds.split(sizes)

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> MiMoV2OmniAudioInputs | None:
        audio_features = kwargs.pop("audio_features", None)
        audio_token_lens = kwargs.pop("audio_token_lens", None)
        if audio_features is None:
            return None
        assert isinstance(audio_features, (torch.Tensor, list))
        if isinstance(audio_features, list):
            assert all(isinstance(item, torch.Tensor) for item in audio_features)
        assert audio_token_lens is None or isinstance(
            audio_token_lens, (torch.Tensor, list)
        )
        return {
            "type": "audio",
            "audio_features": audio_features,
            "audio_token_lens": audio_token_lens,
        }

    def _parse_and_validate_multimodal_inputs(
        self, **kwargs: object
    ) -> MiMoV2OmniMultiModalInputs:
        mm_input_by_modality: MiMoV2OmniMultiModalInputs = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            if (
                input_key in ("pixel_values_videos", "video_embeds")
                and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(
                    **kwargs
                )
            if input_key == "audio_features" and "audio" not in mm_input_by_modality:
                mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(
                    **kwargs
                )
        return mm_input_by_modality

    def _process_audio_input(
        self, audio_input: MiMoV2OmniAudioInputs
    ) -> tuple[torch.Tensor, ...]:
        mel_specs = audio_input["audio_features"]
        if self.audio_encoder is None:
            return ()
        # Normalize to List[2D-Tensor].
        # MultiModalBatchedField._reduce_data either wraps a single [T, 128]
        # into [1, T, 128] via unsqueeze(0) or stacks N same-T items into
        # [N, T, 128]. Indexing along dim-0 extracts the per-item [T, 128].
        if isinstance(mel_specs, torch.Tensor):
            mel_specs = list(mel_specs)  # [1,T,128] or [N,T,128] → [[T,128],...]
        if not mel_specs:
            return ()
        audio_embeds, item_token_lens = self.audio_encoder.get_audio_feature(mel_specs)
        return tuple(audio_embeds.split(item_token_lens))

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        # Pop video_audio-specific fields before main mm parsing
        video_audio_n_segs = kwargs.pop("video_audio_n_segs", None)
        video_audio_seg_lens = kwargs.pop("video_audio_seg_lens", None)
        va_audio_features = kwargs.pop("va_audio_features", None)
        video_audio_frame_counts = kwargs.pop("video_audio_frame_counts", None)
        video_audio_start_offsets = kwargs.pop("video_audio_start_offsets", None)

        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality and va_audio_features is None:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor corresponding to a multimodal data item (image, video, or audio).
        multimodal_embeddings: list[torch.Tensor] = []

        # Pre-process va audio: one mel spec per va video → per-video audio embeddings
        # keyed by va video index (0-based among va videos only)
        va_audio_embs_list: list[torch.Tensor | None] = []
        if va_audio_features is not None:
            if self.audio_encoder is None:
                raise ValueError("Video/audio input requires the MiMo audio encoder")
            assert isinstance(va_audio_features, (torch.Tensor, list, tuple))
            for mel_spec in va_audio_features:
                assert isinstance(mel_spec, torch.Tensor)
                if mel_spec.numel() == 0:
                    va_audio_embs_list.append(None)
                else:
                    embs, _ = self.audio_encoder.get_audio_feature([mel_spec])
                    va_audio_embs_list.append(embs)

        assert video_audio_n_segs is None or isinstance(
            video_audio_n_segs, torch.Tensor
        )
        assert video_audio_seg_lens is None or isinstance(
            video_audio_seg_lens, torch.Tensor
        )

        # NOTE: Iterate in dict insertion order to preserve token sequence order.
        for modality in mm_input_by_modality:
            if modality == "image":
                image_input = mm_input_by_modality["image"]
                assert isinstance(
                    image_input,
                    (Qwen2_5_VLImagePixelInputs, Qwen2_5_VLImageEmbeddingInputs),
                )
                multimodal_embeddings.extend(self._process_image_input(image_input))
            elif modality == "video":
                video_input = mm_input_by_modality["video"]
                assert isinstance(
                    video_input,
                    (Qwen2_5_VLVideoPixelInputs, Qwen2_5_VLVideoEmbeddingInputs),
                )
                video_embs_tuple = self._process_video_input(video_input)
                if video_audio_n_segs is None:
                    multimodal_embeddings.extend(video_embs_tuple)
                else:
                    grid_thw = video_input["video_grid_thw"]
                    for i, vid_embs in enumerate(video_embs_tuple):
                        n_segs = int(video_audio_n_segs[i])
                        if n_segs == 0:
                            multimodal_embeddings.append(vid_embs)
                        else:
                            assert video_audio_seg_lens is not None
                            assert isinstance(video_audio_frame_counts, torch.Tensor)
                            if len(va_audio_embs_list) != len(video_embs_tuple):
                                raise ValueError(
                                    "MiMo video/audio item counts disagree"
                                )
                            if not isinstance(video_audio_start_offsets, torch.Tensor):
                                raise ValueError(
                                    "Missing MiMo video/audio start offsets"
                                )
                            full_va_embs = va_audio_embs_list[i]
                            if full_va_embs is None:
                                raise ValueError(
                                    "Missing audio features for combined video"
                                )
                            start = int(video_audio_start_offsets[i])
                            audio_length = int(video_audio_seg_lens[i][:n_segs].sum())
                            if (
                                start < 0
                                or start + audio_length > full_va_embs.shape[0]
                            ):
                                raise ValueError(
                                    "MiMo video/audio clip exceeds encoded audio"
                                )
                            clip_audio = full_va_embs[start : start + audio_length]
                            multimodal_embeddings.append(
                                _interleave_video_audio_embeddings(
                                    vid_embs,
                                    clip_audio,
                                    int(grid_thw[i][0]),
                                    video_audio_frame_counts[i][:n_segs].tolist(),
                                    video_audio_seg_lens[i][:n_segs].tolist(),
                                )
                            )
            elif modality == "audio":
                audio_input = mm_input_by_modality["audio"]
                assert audio_input is not None
                multimodal_embeddings.extend(self._process_audio_input(audio_input))
        return tuple(multimodal_embeddings)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for Qwen2.5-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch. **NOTE**: If mrope is enabled (default setting for
                Qwen2.5-VL opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            intermediate_tensors: Intermediate tensors from prior forward pass.
            inputs_embeds: Optional tensor of input embeddings.

        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def process_weights_after_loading(self) -> None:
        _check_pending_fp8_qkv_projections(self)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        audio_loaded: set[str] = set()

        loader = AutoWeightsLoader(self)
        auto_loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        return audio_loaded | auto_loaded
