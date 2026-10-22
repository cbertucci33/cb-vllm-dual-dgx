# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiMo-V2 vision window attention has to apply the per-head sink logits."""

import pytest
import torch

from vllm.platforms import current_platform

EMBED_DIM = 256
NUM_HEADS = 4
HEAD_DIM = 64
WINDOW = 8
# One sequence shorter than the window, one longer.
SEQ_LENS = [5, 37]


@pytest.fixture
def vision_attn_env(dist_init):
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    yield
    torch.set_default_dtype(default_dtype)


def _reference(q, k, v, cu_seqlens, sinks, scale):
    """Dense windowed softmax with the sink added to each sequence's key 0."""
    groups = q.shape[1] // k.shape[1]
    out = torch.empty_like(q, dtype=torch.float32)
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        qs = q[start:end].float()
        ks = k[start:end].float().repeat_interleave(groups, dim=1)
        vs = v[start:end].float().repeat_interleave(groups, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qs, ks) * scale
        scores[..., 0] += sinks.float().view(-1, 1)
        pos = torch.arange(end - start, device=q.device)
        outside = (pos.view(-1, 1) - pos.view(1, -1)).abs() > WINDOW
        scores.masked_fill_(outside, -torch.inf)
        out[start:end] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vs)
    return out


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires flash-attn")
@pytest.mark.parametrize("num_kv_heads", [NUM_HEADS, NUM_HEADS // 2])
def test_window_attention_applies_sinks(vision_attn_env, num_kv_heads):
    from vllm.model_executor.models.mimo_v2_omni import MiMoVisionAttention

    torch.manual_seed(0)
    attn = MiMoVisionAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=num_kv_heads,
        qk_channels=HEAD_DIM,
        kv_channels=HEAD_DIM,
        use_sink=True,
        visual_token_window_size=WINDOW,
    ).cuda()
    attn.sinks.data.normal_()

    total = sum(SEQ_LENS)
    opts = dict(device="cuda", dtype=torch.bfloat16)
    q = torch.randn(total, NUM_HEADS, HEAD_DIM, **opts)
    k = torch.randn(total, num_kv_heads, HEAD_DIM, **opts)
    v = torch.randn(total, num_kv_heads, HEAD_DIM, **opts)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(SEQ_LENS).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )

    out = attn._forward_window_attn(q, k, v, cu_seqlens, max(SEQ_LENS))
    ref = _reference(q, k, v, cu_seqlens, attn.sinks, attn.scale)

    # bf16 attention lands at ~2e-3 here; dropping the sinks lands at ~1e-1.
    error = ((out.float() - ref).norm() / ref.norm()).item()
    assert error < 1e-2, f"sink-corrected output is off by {error:.2e}"


def test_omni_exl3_qkv_mapping_selects_separate_quantized_projections():
    from vllm.model_executor.layers.quantization.exl3 import Exl3Config
    from vllm.model_executor.models.mimo_v2_omni import MiMoV2OmniForCausalLM

    prefix = "model.layers.0.self_attn"
    storage = {
        f"{prefix}.{leaf}": {"quant_format": "exl3", "stored_tensors": {}}
        for leaf in ("q_proj", "k_proj", "v_proj")
    }
    config = Exl3Config(tensor_storage=storage)
    config.packed_modules_mapping = MiMoV2OmniForCausalLM.packed_modules_mapping
    assert config._linear_prefix_is_exl3(f"{prefix}.qkv_proj")


def test_processor_preserves_all_declared_audio_code_channels():
    from types import SimpleNamespace

    from vllm.transformers_utils.processors.mimo_v2_omni import (
        AudioInput,
        MiMoOmniProcessor,
    )

    config = SimpleNamespace(
        vision_config={},
        processor_config={
            "audio_channels": 20,
            "audio_group_size": 4,
            "audio_zeroemb_idx": [1024] * 20,
        },
    )
    processor = MiMoOmniProcessor.from_hf_config(
        SimpleNamespace(pad_token_id=0), config
    ).mimo_processor
    codes = torch.arange(5 * 20).reshape(5, 20)
    grouped = processor.process_audio(AudioInput(codes))
    assert grouped.shape == (2, 4, 20)
    torch.testing.assert_close(grouped.reshape(8, 20)[:5], codes)
    torch.testing.assert_close(grouped.reshape(8, 20)[5:], codes[-1:].expand(3, 20))
    assert processor.audio_zeroemb_idxs.tolist() == [1024] * 20


@pytest.fixture
def mimo_audio_profile_info(monkeypatch):
    from types import SimpleNamespace

    from transformers import PreTrainedConfig

    import vllm.envs as envs
    from vllm.config.multimodal import MultiModalConfig
    from vllm.model_executor.models.mimo_v2_omni import MiMoV2OmniProcessingInfo
    from vllm.multimodal.processing.context import InputProcessingContext

    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_DURATION_S", 2)
    hf = PreTrainedConfig(
        vision_config={
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        processor_config={
            "audio_channels": 20,
            "audio_zeroemb_idx": [1024] * 20,
        },
    )
    mm = MultiModalConfig(
        mm_device_do_normalize=False,
        limit_per_prompt={"audio": 1, "image": 0, "video": 0},
    )
    model_config = SimpleNamespace(
        hf_config=hf,
        multimodal_config=mm,
        dtype=torch.bfloat16,
        max_model_len=800000,
        get_multimodal_config=lambda: mm,
    )
    tokenizer = SimpleNamespace(pad_token_id=0, encode=lambda text: [0])
    return MiMoV2OmniProcessingInfo(InputProcessingContext(model_config, tokenizer))


def test_independent_audio_gets_encoder_budget(mimo_audio_profile_info):
    from types import SimpleNamespace

    from vllm.model_executor.models.mimo_v2_omni import (
        MiMoV2OmniDummyInputsBuilder,
        MiMoV2OmniMultiModalProcessor,
    )
    from vllm.multimodal.encoder_budget import MultiModalBudget

    info = mimo_audio_profile_info
    processor = MiMoV2OmniMultiModalProcessor(info, MiMoV2OmniDummyInputsBuilder(info))
    scheduler = SimpleNamespace(
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        max_num_encoder_input_tokens=4096,
        encoder_cache_size=4096,
        disable_chunked_mm_input=False,
        enable_chunked_prefill=True,
    )
    budget = MultiModalBudget(
        SimpleNamespace(model_config=info.ctx.model_config, scheduler_config=scheduler),
        SimpleNamespace(create_processor=lambda config: processor),
        enable_cache=False,
    )
    assert budget.mm_max_toks_per_item == {"audio": 13}
    assert budget.encoder_compute_budget >= 13
    assert budget.encoder_cache_size >= 13
    assert budget.mm_max_items_per_batch["audio"] == 1


@pytest.mark.parametrize("seq_len", [3, 12, 1000])
def test_audio_profile_matches_real_mel_and_context_bound(
    mimo_audio_profile_info, seq_len
):
    from vllm.model_executor.models.mimo_v2_omni import MiMoV2OmniDummyInputsBuilder
    from vllm.transformers_utils.processors.mimo_v2_omni import AudioInput

    info = mimo_audio_profile_info
    mm = info.ctx.model_config.multimodal_config
    builder = MiMoV2OmniDummyInputsBuilder(info)
    data = builder.get_dummy_mm_data(seq_len, {"audio": 1}, mm.limit_per_prompt)
    assert len(data["audio"]) == 1
    waveform = data["audio"][0]
    processor = info.get_hf_processor().mimo_processor
    _, actual_tokens = processor.process_audio(AudioInput(waveform))
    predicted = info.get_mm_max_tokens_per_item(seq_len, {"audio": 1})["audio"]
    assert predicted == actual_tokens
    assert actual_tokens + 2 <= seq_len
    assert len(waveform) <= 2 * processor.audio_sampling_rate
    if len(waveform) < 2 * processor.audio_sampling_rate:
        _, next_tokens = processor.process_audio(
            AudioInput(torch.zeros(len(waveform) + 1).numpy())
        )
        assert next_tokens + 2 > seq_len


def test_audio_profile_uses_decoder_policy_without_new_limit(
    mimo_audio_profile_info, monkeypatch
):
    import vllm.envs as envs

    info = mimo_audio_profile_info
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_DECODE_DURATION_S", 600)
    assert info._get_audio_profile(800000) == (14_400_000, 3751)


def test_audio_disabled_does_not_make_dummy_waveforms(mimo_audio_profile_info):
    from vllm.model_executor.models.mimo_v2_omni import MiMoV2OmniDummyInputsBuilder

    info = mimo_audio_profile_info
    data = MiMoV2OmniDummyInputsBuilder(info).get_dummy_mm_data(
        1000, {"audio": 0}, info.ctx.model_config.multimodal_config.limit_per_prompt
    )
    assert data["audio"] == []
    assert "audio" not in info.get_mm_max_tokens_per_item(1000, {"audio": 0})


@pytest.fixture
def mimo_video_profile_info():
    from types import SimpleNamespace

    from transformers import PreTrainedConfig

    from vllm.config.multimodal import MultiModalConfig
    from vllm.model_executor.models.mimo_v2_omni import MiMoV2OmniProcessingInfo
    from vllm.multimodal.processing.context import InputProcessingContext

    hf = PreTrainedConfig(
        vision_config={
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        processor_config={
            "image_min_pixels": 1024,
            "image_max_pixels": 8192,
            "video_min_pixels": 1024,
            "video_max_pixels": 8192,
            "video_total_max_pixels": 262144,
            "max_frames": 96,
            "audio_channels": 20,
            "audio_zeroemb_idx": [1024] * 20,
            "image_token_id": 1,
            "video_token_id": 2,
            "audio_token_id": 9,
            "vision_start_token_id": 3,
            "vision_end_token_id": 4,
            "video_start_token_id": 5,
            "video_end_token_id": 6,
            "audio_start_token_id": 7,
            "audio_end_token_id": 8,
        },
    )
    mm = MultiModalConfig(
        mm_device_do_normalize=False,
        limit_per_prompt={"audio": 0, "image": 0, "video": 1},
    )
    model = SimpleNamespace(
        hf_config=hf,
        multimodal_config=mm,
        dtype=torch.bfloat16,
        max_model_len=800000,
        get_multimodal_config=lambda: mm,
    )
    tokenizer = SimpleNamespace(pad_token_id=0, encode=lambda *args, **kwargs: [100])
    return MiMoV2OmniProcessingInfo(InputProcessingContext(model, tokenizer))


def test_video_budget_covers_32_frames_and_dummy_matches_real_processor(
    mimo_video_profile_info,
):
    from vllm.model_executor.models.mimo_v2_omni import MiMoV2OmniDummyInputsBuilder
    from vllm.transformers_utils.processors.mimo_v2_omni import VideoInput

    info = mimo_video_profile_info
    native = info.get_hf_processor().mimo_processor
    frames = torch.zeros(32, 3, 64, 128)
    _, grid, _, _ = native.process_video(VideoInput((frames, torch.arange(32).float())))
    actual_tokens = int(grid.prod()) // native.merge_size**2
    assert actual_tokens == 128
    maximum = info.get_mm_max_tokens_per_item(800000, {"video": 1})["video"]
    assert maximum >= actual_tokens
    count, size, predicted = info._get_video_profile(800000)
    data = MiMoV2OmniDummyInputsBuilder(info).get_dummy_mm_data(
        800000, {"video": 1}, info.ctx.model_config.multimodal_config.limit_per_prompt
    )
    dummy = torch.from_numpy(data["video"][0]).permute(0, 3, 1, 2)
    _, grid, _, _ = native.process_video(
        VideoInput((dummy, torch.arange(count).float()))
    )
    assert int(grid.prod()) // native.merge_size**2 == predicted == maximum
    assert tuple(dummy.shape[-2:]) == (size.height, size.width)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_pixels": 4096},
        {"size": {"shortest_edge": 1024, "longest_edge": 4096}},
        {"images_kwargs": {"max_pixels": 4096}},
    ],
)
def test_processor_overrides_change_actual_image_geometry_and_match_sizing(
    mimo_video_profile_info,
    kwargs,
):
    from copy import deepcopy

    from vllm.transformers_utils.processors.mimo_v2_omni import ImageInput

    info = mimo_video_profile_info
    before = deepcopy(kwargs)
    overridden = info.get_hf_processor(**kwargs).mimo_processor
    image = torch.zeros(3, 64, 128)
    resized = overridden.process_image(ImageInput(image))
    actual = (
        resized.shape[-2]
        * resized.shape[-1]
        // (overridden.patch_size * overridden.merge_size) ** 2
    )
    predicted = info.get_num_image_tokens(
        image_width=128,
        image_height=64,
        image_processor=info.get_image_processor(),
        mm_kwargs=kwargs,
    )
    assert actual == predicted == 2
    assert kwargs == before
    assert info.get_hf_processor().mimo_processor._img_kw["max_pixels"] == 8192


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_pixels": 0},
        {"unexpected_pixel_setting": 4096},
        {"max_pixels": 4096, "image_max_pixels": 8192},
        {"patch_size": 8},
    ],
)
def test_processor_cannot_ignore_invalid_unknown_or_conflicting_overrides(
    mimo_video_profile_info,
    kwargs,
):
    with pytest.raises(ValueError):
        mimo_video_profile_info.get_hf_processor(**kwargs)


def test_combined_video_audio_returns_one_cache_item_with_nonuniform_groups():
    from vllm.model_executor.models.mimo_v2_omni import (
        _interleave_video_audio_embeddings,
    )

    video = torch.arange(6).reshape(6, 1).float()
    audio = torch.tensor([[10.0], [11.0], [12.0]])
    combined = _interleave_video_audio_embeddings(video, audio, 3, [1, 2], [2, 1])
    assert combined.shape == (9, 1)
    assert combined[:, 0].tolist() == [0, 1, 10, 11, 2, 3, 4, 5, 12]
    with pytest.raises(ValueError):
        _interleave_video_audio_embeddings(video, audio, 3, [1, 1], [2, 1])


def test_combined_video_audio_parser_keeps_audio_in_video_hash(
    mimo_video_profile_info,
):
    from vllm.transformers_utils.processors.mimo_v2_omni import VideoAudioInput

    frames = torch.zeros(2, 3, 32, 32)
    audio = (torch.zeros(24000), 24000)
    parser = mimo_video_profile_info.get_data_parser()
    items = parser.parse_mm_data({"video": VideoAudioInput(frames, audio)})
    assert items.get_count("video") == 1
    video = items["video"]
    assert video.get_num_frames(0) == 2
    assert video.get_frame_size(0) == (32, 32)
    assert video.get_item_for_hash(0)["audio"] is audio


@pytest.mark.parametrize(
    "field",
    [
        "causal",
        "mm_prefix_range",
        "mm_prefix_range_tensor",
        "rswa_prefix_lens",
        "rswa_window",
    ],
)
def test_diffkv_rejects_unsupported_metadata_instead_of_ignoring_it(field):
    from types import SimpleNamespace

    from vllm.v1.attention.backends.triton_attn_diffkv import (
        TritonAttentionDiffKVBackend,
        TritonAttentionDiffKVImpl,
    )

    assert not TritonAttentionDiffKVBackend.supports_non_causal()
    assert not TritonAttentionDiffKVBackend.supports_mm_prefix()
    assert not TritonAttentionDiffKVBackend.supports_rswa()
    impl = TritonAttentionDiffKVImpl(32, 192, 192**-0.5, 2, None, None, "fp8_e4m3")
    metadata = SimpleNamespace(
        use_cascade=False,
        causal=True,
        mm_prefix_range=None,
        mm_prefix_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
    )
    setattr(metadata, field, False if field == "causal" else torch.tensor([1]))
    q = torch.zeros(1, 32, 192)
    k = torch.zeros(1, 2, 192)
    v = torch.zeros(1, 2, 128)
    with pytest.raises(NotImplementedError, match="causal|MM-prefix"):
        impl.forward(None, q, k, v, torch.empty(0), metadata, torch.empty(0))


def test_flash_diffkv_constructor_binds_actual_value_dimension(monkeypatch):
    import vllm.v1.attention.backends.flash_attn as parent
    import vllm.v1.attention.backends.flash_attn_diffkv as child

    seen = []

    def support(dtype, **kwargs):
        seen.append(kwargs["head_size_v"])
        return kwargs["head_size_v"] == 128

    monkeypatch.setattr(parent, "get_flash_attn_version", lambda **kwargs: 4)
    monkeypatch.setattr(child, "get_flash_attn_version", lambda **kwargs: 4)
    monkeypatch.setattr(parent, "flash_attn_supports_kv_cache_dtype", support)
    monkeypatch.setattr(parent, "flash_attn_supports_quant_query_input", lambda: True)
    impl = child.FlashAttentionDiffKVImpl(32, 192, 192**-0.5, 2, None, None, "fp8_e4m3")
    assert impl.head_size_v == 128
    assert seen == [128]


def test_mimo_profiles_every_active_modality_and_full_cache_capacity(
    mimo_video_profile_info,
):
    from types import SimpleNamespace

    from vllm.multimodal.encoder_budget import MultiModalBudget

    budget = object.__new__(MultiModalBudget)
    budget.processor = SimpleNamespace(info=mimo_video_profile_info)
    budget.mm_max_toks_per_item = {"image": 8, "video": 256, "audio": 13}
    budget.encoder_cache_size = 512
    assert set(budget.get_encoder_profile_modalities()) == {"image", "video", "audio"}
    assert budget.get_profile_cache_padding("audio", 2) == 486
    assert budget.get_profile_cache_padding("audio", 2, 24) == 488
    with pytest.raises(ValueError):
        budget.get_profile_cache_padding("audio", 2, 27)
    budget.processor = SimpleNamespace(info=SimpleNamespace())
    assert budget.get_encoder_profile_modalities() == ("video",)
    assert budget.get_profile_cache_padding("video", 1) == 0


def test_combined_video_audio_processor_carries_group_counts_and_audio(
    mimo_video_profile_info,
):
    from vllm.transformers_utils.processors.mimo_v2_omni import VideoAudioInput

    processor = mimo_video_profile_info.get_hf_processor()
    clip = VideoAudioInput(
        video=(torch.zeros(2, 3, 32, 32), torch.tensor([0.0, 1.0])),
        audio=(torch.zeros(48000), 24000),
    )
    result = processor(
        text="<|vision_start|><|video_pad|><|vision_end|>",
        videos=[clip],
        return_tensors="pt",
    )
    assert result["video_audio_n_segs"].tolist() == [1]
    assert result["video_audio_frame_counts"].tolist() == [[1]]
    assert len(result["va_audio_features"]) == 1
    assert result["video_grid_timestamps"][0].tolist() == [0.0]


def test_mixed_video_audio_keeps_regular_decoder_metadata(mimo_video_profile_info):
    import numpy as np

    from vllm.transformers_utils.processors.mimo_v2_omni import VideoAudioInput

    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    decoder = {"fps": 2.0, "frames_indices": [4, 9]}
    combined = VideoAudioInput(
        video=(torch.zeros(2, 3, 32, 32), torch.tensor([0.0, 1.0])),
        audio=(torch.zeros(48000), 24000),
    )
    items = mimo_video_profile_info.get_data_parser().parse_mm_data(
        {"video": [(frames, decoder), combined]}
    )["video"]
    assert items.metadata[0] == decoder
    assert items.metadata[1] is None
    assert items.get_item_for_hash(0)["metadata"] == decoder


def test_item_image_override_cannot_escape_profile(mimo_video_profile_info):
    from vllm.transformers_utils.processors.mimo_v2_omni import ImageInput

    processor = mimo_video_profile_info.get_hf_processor().mimo_processor
    with pytest.raises(ValueError, match="profile bounds"):
        processor.process_image(ImageInput(torch.zeros(3, 64, 128), max_pixels=16384))
