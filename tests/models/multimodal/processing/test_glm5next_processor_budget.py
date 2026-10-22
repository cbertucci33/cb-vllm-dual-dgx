# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise processor budget ownership without loading model weights."""

from types import SimpleNamespace

import pytest
from transformers.models.glm5_next.image_processing_glm5_next import (
    Glm5NextImageProcessor,
)
from transformers.models.glm5_next.video_processing_glm5_next import (
    Glm5NextVideoProcessor,
    smart_resize,
)

from vllm.model_executor.models.glm4_1v import Glm4vProcessingInfo
from vllm.models.glm5next.common.multimodal import Glm5NextProcessingInfo
from vllm.multimodal.video import glm5next_sample_frames


def _processing_info(monkeypatch, max_tokens):
    # Real official sub-processors. Only the parent factory is replaced: no
    # tokenizer download or model/vision allocation is needed for this boundary.
    processor = SimpleNamespace(
        image_processor=Glm5NextImageProcessor(),
        video_processor=Glm5NextVideoProcessor(max_image_tokens=max_tokens),
    )
    received = []

    def parent_factory(self, **kwargs):
        received.append(kwargs)
        return processor

    monkeypatch.setattr(Glm4vProcessingInfo, "get_hf_processor", parent_factory)
    info = object.__new__(Glm5NextProcessingInfo)
    return info, processor, received


@pytest.mark.parametrize("max_tokens", [16000, 30000, 30001, 48000, 240000, 400000])
def test_glm_video_processor_preserves_configured_token_budget(monkeypatch, max_tokens):
    info, processor, received = _processing_info(monkeypatch, max_tokens)
    image_state = vars(processor.image_processor).copy()
    video_state = vars(processor.video_processor).copy()

    for _ in range(2):
        assert info.get_hf_processor(probe="forwarded") is processor
        assert processor.video_processor.max_image_tokens == max_tokens
        assert (
            processor.video_processor.sample_frames.__func__ is glm5next_sample_frames
        )
        assert (
            processor.video_processor.sample_frames.__self__
            is processor.video_processor
        )
        assert vars(processor.image_processor) == image_state
        # The shared frame sampler is the only intended processor mutation.
        assert {
            key: value
            for key, value in vars(processor.video_processor).items()
            if key != "sample_frames"
        } == video_state
    assert received == [{"probe": "forwarded"}, {"probe": "forwarded"}]


@pytest.mark.parametrize(
    "frames,height,width,expected_grid",
    [
        (8, 480, 640, (4, 36, 46)),
        (40, 1080, 1920, (20, 78, 138)),
        (120, 1080, 1920, (60, 78, 138)),
        (602, 720, 1280, (301, 42, 74)),
    ],
)
def test_glm_video_geometry_retains_official_budget(
    monkeypatch, frames, height, width, expected_grid
):
    info, processor, _ = _processing_info(monkeypatch, 240000)
    vp = info.get_hf_processor().video_processor
    resized_h, resized_w = smart_resize(
        frames,
        height,
        width,
        temporal_factor=vp.temporal_patch_size,
        factor=vp.patch_size * vp.merge_size * vp.patch_expand_factor,
        min_pixels=vp.min_image_tokens,
        max_pixels=vp.max_image_tokens,
    )
    grid = (
        frames // vp.temporal_patch_size,
        resized_h // vp.patch_size,
        resized_w // vp.patch_size,
    )
    assert grid == expected_grid
    assert grid[0] * grid[1] * grid[2] // vp.merge_size**2 <= 240000


@pytest.mark.parametrize("frames", [5, 6, 9, 10, 61, 62, 601, 602])
@pytest.mark.parametrize("processor_type", ["image", "video"])
def test_glm_resize_budget_counts_temporal_padding(frames, processor_type):
    from transformers.models.glm5_next import (
        image_processing_glm5_next,
        video_processing_glm5_next,
    )

    resize = (
        image_processing_glm5_next.smart_resize
        if processor_type == "image"
        else video_processing_glm5_next.smart_resize
    )
    max_tokens = (frames // 2) * 16
    height, width = resize(
        frames,
        112,
        112,
        temporal_factor=2,
        factor=28,
        min_pixels=1,
        max_pixels=max_tokens,
    )
    actual_grid_t = (frames + 1) // 2
    assert actual_grid_t * (height // 28) * (width // 28) <= max_tokens


@pytest.mark.parametrize("frames", [5, 6, 9, 10, 13, 14])
def test_glm_video_resize_and_patchify_preserve_frames_within_budget(frames):
    import torch

    vp = Glm5NextVideoProcessor()
    pixels = torch.zeros((1, frames, 3, 112, 112), dtype=torch.uint8)
    max_tokens = (frames // 2) * 16
    resized = vp.resize(
        pixels,
        vp.resample,
        factor=28,
        temporal_factor=2,
        min_image_tokens=1,
        max_image_tokens=max_tokens,
    )
    patches, grid_t, grid_h, grid_w = vp.patchify(
        resized, patch_size=14, merge_size=2, temporal_patch_size=2
    )
    assert resized.shape[1] == frames
    assert grid_t == (frames + 1) // 2
    assert patches.shape[1] == grid_t * grid_h * grid_w
    assert patches.shape[1] // 4 <= max_tokens


@pytest.mark.parametrize(
    "stale_file",
    [
        None,
        "image_processing_glm5_next.py",
        "video_processing_glm5_next.py",
        "modular_glm5_next.py",
    ],
)
def test_glm_temporal_budget_packaging_rejects_stale_sources(tmp_path, stale_file):
    import importlib.util
    import shutil
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    patch = root / "build/patches/transformers-glm5next-temporal-budget.patch"
    transformers_spec = importlib.util.find_spec("transformers")
    assert transformers_spec is not None and transformers_spec.origin is not None
    package = Path(transformers_spec.origin).parent
    relative = Path("models/glm5_next")
    (tmp_path / relative).mkdir(parents=True)
    for name in (
        "image_processing_glm5_next.py",
        "video_processing_glm5_next.py",
        "modular_glm5_next.py",
    ):
        shutil.copyfile(package / relative / name, tmp_path / relative / name)
    if stale_file:
        subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "apply",
                "--reverse",
                f"--include={relative / stale_file}",
                str(patch),
            ],
            check=True,
            capture_output=True,
        )

    spec = importlib.util.spec_from_file_location(
        "glm_budget_verifier", root / "build/verify_runtime.py"
    )
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    if stale_file:
        with pytest.raises(subprocess.CalledProcessError):
            verifier.verify_glm5next_temporal_sources(tmp_path, patch)
    else:
        verifier.verify_glm5next_temporal_sources(tmp_path, patch)


def test_glm_video_profile_matches_the_full_processor_budget(monkeypatch):
    info, processor, _ = _processing_info(monkeypatch, 240000)
    info.ctx = SimpleNamespace(get_merged_mm_kwargs=lambda kwargs: kwargs)
    vision = SimpleNamespace(patch_size=14, spatial_merge_size=2, temporal_patch_size=2)
    monkeypatch.setattr(
        info, "get_hf_config", lambda: SimpleNamespace(vision_config=vision)
    )
    width, height = info.get_image_size_with_most_features()
    frames = info.get_num_frames_with_most_features(800000, {"video": 1})
    _, planned_tokens = info._get_vision_info(
        image_width=width,
        image_height=height,
        num_frames=frames,
        max_image_pixels=info._get_video_max_pixels(),
    )
    assert planned_tokens == 240000
    vp = processor.video_processor
    resized_h, resized_w = smart_resize(
        frames,
        height,
        width,
        temporal_factor=2,
        factor=28,
        min_pixels=vp.min_image_tokens,
        max_pixels=vp.max_image_tokens,
    )
    actual_tokens = ((frames + 1) // 2) * (resized_h // 28) * (resized_w // 28)
    assert actual_tokens == planned_tokens
