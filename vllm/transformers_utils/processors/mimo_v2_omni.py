# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
"""MiMo-Omni multimodal processor for vLLM.

Ported from SGLang's MiMoV2OmniProcessor / MiMoVLProcessor implementations.
"""

import contextlib
import inspect
import logging
import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import regex as re
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BatchFeature, TensorType
from transformers.processing_utils import ProcessorMixin

try:
    import torchaudio
    from torchaudio.transforms import MelSpectrogram as _MelSpectrogram

    _HAS_TORCHAUDIO = True
except ImportError:
    torchaudio = None  # type: ignore[assignment]
    _MelSpectrogram = None  # type: ignore[assignment,misc]
    _HAS_TORCHAUDIO = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PIXEL_MEAN = [123.675, 116.28, 103.53]
_PIXEL_STD = [58.395, 57.12, 57.375]
_mean_std_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ImageInput:
    # PIL.Image | torch.Tensor (C,H,W)
    image: Any
    max_pixels: int | None = None
    min_pixels: int | None = None


@dataclass
class VideoInput:
    # tuple[frames_TCHW: torch.Tensor, timestamps_T: torch.Tensor]
    video: Any
    min_pixels: int | None = None
    max_pixels: int | None = None
    total_max_pixels: int | None = None
    fps: float | None = None
    num_frames: int | None = None
    max_frames: int | None = None
    min_frames: int | None = None
    do_include_last_frame: bool | None = False
    start_time: float | None = None
    end_time: float | None = None
    segment_type: Literal["individual", "partial"] = "individual"


@dataclass
class AudioInput:
    # tuple[waveform_1D, sr]
    # | np.ndarray | torch.Tensor (T,n_vq)
    audio: Any


@dataclass
class VideoAudioInput:
    video: Any  # same as VideoInput.video
    audio: Any  # same as AudioInput.audio
    min_pixels: int | None = None
    max_pixels: int | None = None
    total_max_pixels: int | None = None
    fps: float | None = None
    num_frames: int | None = None
    max_frames: int | None = None
    min_frames: int | None = None
    do_include_last_frame: bool | None = False
    start_time: float | None = None
    end_time: float | None = None
    segment_type: Literal["individual", "partial"] = "individual"


@dataclass
class Content:
    type: Literal["text", "image", "video", "audio", "video_audio"]
    content: Any
    is_target: bool | None = None


@dataclass
class MiMoVLInputSample:
    input_ids: torch.Tensor
    labels: torch.Tensor | None
    pixel_values: list[torch.Tensor]
    pixel_values_videos: list[torch.Tensor]
    image_thw_grids: list[torch.Tensor]
    video_thw_grids: list[torch.Tensor]
    audio_inputs: list[torch.Tensor]
    second_per_grid_ts: list[float] = field(default_factory=list)
    video_start_times: list[float] = field(default_factory=list)
    video_grid_timestamps: list[torch.Tensor] = field(default_factory=list)
    audio_token_lens: list[int] = field(default_factory=list)
    va_audio_inputs: list[torch.Tensor] = field(default_factory=list)
    video_audio_n_segs: list[int] = field(default_factory=list)
    video_audio_seg_lens: list[int] = field(default_factory=list)
    video_audio_frame_counts: list[int] = field(default_factory=list)
    video_audio_start_offsets: list[int] = field(default_factory=list)
    position_ids: torch.Tensor | None = None
    rope_deltas: torch.Tensor | None = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Vision utilities
# ---------------------------------------------------------------------------


def _format_timestamp(ts: float) -> str:
    return f"{int(ts // 60):02d}:{int(ts % 60):02d}"


def _smart_resize(
    h: int, w: int, factor: int, min_px: int, max_px: int
) -> tuple[int, int]:
    if h <= 0 or w <= 0 or factor <= 0:
        raise ValueError("MiMo resize requires positive dimensions and patch geometry")
    if min(h, w) < factor:
        if h < w:
            h, w = factor, int(w * factor / h)
        else:
            w, h = factor, int(h * factor / w)
    elif max(h, w) / min(h, w) > 200:
        raise ValueError(f"Aspect ratio > 200 not allowed: {h}x{w}")
    h_bar = round(h / factor) * factor
    w_bar = round(w / factor) * factor
    if h_bar * w_bar > max_px:
        beta = math.sqrt((h * w) / max_px)
        h_bar = math.floor(h / beta / factor) * factor
        w_bar = math.floor(w / beta / factor) * factor
    elif h_bar * w_bar < min_px:
        beta = math.sqrt(min_px / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    if (
        h_bar <= 0
        or w_bar <= 0
        or not min_px <= h_bar * w_bar <= max_px
        or max(h_bar, w_bar) / min(h_bar, w_bar) > 200
    ):
        raise ValueError(
            "MiMo patch-aligned resize cannot satisfy declared pixel bounds"
        )
    return int(h_bar), int(w_bar)


def _standardize(images: torch.Tensor) -> torch.Tensor:
    key = str(images.device)
    if key not in _mean_std_cache:
        mean = torch.tensor(_PIXEL_MEAN, device=images.device).view(1, -1, 1, 1)
        std = torch.tensor(_PIXEL_STD, device=images.device).view(1, -1, 1, 1)
        _mean_std_cache[key] = (mean, std)
    mean, std = _mean_std_cache[key]
    return (images - mean) / std


def _admit_processor_bytes(nbytes: int, device, label: str) -> None:
    from vllm.utils.mem_utils import check_startup_memory

    is_host = device is None or torch.device(device).type == "cpu"
    check_startup_memory(
        host_bytes=nbytes if is_host else 0,
        device_bytes=0 if is_host else nbytes,
        device=device,
        label=label,
    )


def _transform_batch(
    frames: torch.Tensor,
    factor: int,
    min_px: int,
    max_px: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, int, int]:
    count, channels, height, width = frames.shape
    h_out, w_out = _smart_resize(height, width, factor, min_px, max_px)
    resized_bytes = count * channels * h_out * w_out * 4
    input_float_bytes = frames.numel() * 4
    _admit_processor_bytes(
        max(input_float_bytes + resized_bytes, resized_bytes * 3)
        + (frames.nbytes if device is not None and frames.device != device else 0),
        device or frames.device,
        "MiMo video float, resize and standardization intermediates",
    )
    if device is not None:
        frames = frames.to(device)
    _, _, h, w = frames.shape
    h_bar, w_bar = _smart_resize(h, w, factor, min_px, max_px)
    resized = F.interpolate(
        frames.float(), (h_bar, w_bar), mode="bilinear", align_corners=False
    )
    return _standardize(resized), w_bar, h_bar


def _transform_single(
    img: Any,
    factor: int,
    min_px: int,
    max_px: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, int, int]:
    if isinstance(img, torch.Tensor):
        channels, height, width = img.shape
        input_bytes = img.numel() * 4
        target_device = device or img.device
    elif isinstance(img, Image.Image):
        width, height = img.size
        channels = 3
        input_bytes = width * height * channels * (1 + 4)
        target_device = device
    else:
        raise TypeError(f"Expected Tensor or PIL.Image, got {type(img)}")
    h_out, w_out = _smart_resize(height, width, factor, min_px, max_px)
    resized_bytes = channels * h_out * w_out * 4
    _admit_processor_bytes(
        max(input_bytes + resized_bytes, resized_bytes * 3),
        target_device,
        "MiMo image float, resize and standardization intermediates",
    )
    if isinstance(img, torch.Tensor):
        t = img.float()
        _, h, w = t.shape
    elif isinstance(img, Image.Image):
        img = img.convert("RGB")
        w, h = img.size
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
    else:
        raise TypeError(f"Expected Tensor or PIL.Image, got {type(img)}")
    if device is not None:
        t = t.to(device)
    h_bar, w_bar = _smart_resize(h, w, factor, min_px, max_px)
    out = F.interpolate(
        t.unsqueeze(0), (h_bar, w_bar), mode="bilinear", align_corners=False
    )
    return _standardize(out).squeeze(0), w_bar, h_bar


# ---------------------------------------------------------------------------
# Core processor
# ---------------------------------------------------------------------------


class MiMoVLProcessor:
    """Core MiMo-VL multimodal processor.

    Handles image/video/audio preprocessing and token sequence construction.
    Ported from SGLang's MiMoVLProcessor.

    Media strings (URL / path / data:) are resolved upstream in vLLM's media
    pipeline; this processor accepts only already-decoded inputs (``PIL.Image``
    for images, ``(waveform, sr)`` tuples for audio).
    """

    def __init__(
        self,
        tokenizer: Any,
        patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
        temporal_compression_ratio: int = 1,
        use_video_timestamps: bool = True,
        video_audio_interleave_length: int = 0,
        audio_kernel_size: int = 3,
        audio_stride_size: int = 2,
        audio_avg_pooler: int = 2,
        audio_sampling_rate: int = 24000,
        audio_nfft: int = 960,
        audio_hop_length: int = 240,
        audio_window_size: int = 960,
        audio_fmin: float = 0.0,
        audio_fmax: float | None = None,
        audio_n_mels: int = 128,
        audio_segment_size: int = 6000,
        audio_channels: int = 8,
        audio_group_size: int = 4,
        audio_input_id_per_second: float = 25.0,
        audio_zeroemb_idx: int = 4096,
        image_min_pixels: int | None = None,
        image_max_pixels: int | None = None,
        video_min_pixels: int | None = None,
        video_max_pixels: int | None = None,
        video_total_max_pixels: int | None = None,
        fps: float | None = None,
        num_frames: int | None = None,
        max_frames: int | None = None,
        min_frames: int | None = None,
        image_token_id: int | None = None,
        video_token_id: int | None = None,
        audio_token_id: int | None = None,
        vision_start_token_id: int | None = None,
        vision_end_token_id: int | None = None,
        audio_start_token_id: int | None = None,
        audio_end_token_id: int | None = None,
        video_start_token_id: int | None = None,
        video_end_token_id: int | None = None,
        pad_token_id: int | None = None,
        rope_type: str = "rope",
        video_process_num_threads: int = 16,
        device: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.video_process_num_threads = video_process_num_threads
        self.device = torch.device(device) if isinstance(device, str) else device

        self.rope_type = "rope" if rope_type == "1d" else rope_type
        assert self.rope_type in ("rope", "mrope"), (
            f"Unknown rope_type: {self.rope_type}"
        )

        # video timestamps require 1-D rope
        assert use_video_timestamps, "use_video_timestamps must be True"
        assert self.rope_type == "rope", (
            "use_video_timestamps requires rope_type='rope'"
        )
        self.use_video_timestamps = use_video_timestamps
        self.video_audio_interleave_length = video_audio_interleave_length

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.pad_token_id = pad_token_id

        if any(
            value <= 0
            for value in (
                patch_size,
                merge_size,
                temporal_patch_size,
                temporal_compression_ratio,
                audio_sampling_rate,
                audio_hop_length,
                audio_kernel_size,
                audio_stride_size,
                audio_avg_pooler,
                audio_group_size,
                audio_channels,
            )
        ):
            raise ValueError("MiMo processor geometry and rates must be positive")
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size
        self.temporal_compression_ratio = temporal_compression_ratio

        self.audio_sampling_rate = audio_sampling_rate
        self.audio_nfft = audio_nfft
        self.audio_hop_length = audio_hop_length
        self.audio_window_size = audio_window_size
        self.audio_fmin = audio_fmin
        self.audio_fmax = audio_fmax
        self.audio_n_mels = audio_n_mels
        self.audio_segment_size = audio_segment_size
        self.audio_kernel_size = audio_kernel_size
        self.audio_stride_size = audio_stride_size
        self.audio_avg_pooler = audio_avg_pooler
        self.audio_channels = audio_channels
        self.audio_group_size = audio_group_size
        self.audio_input_id_per_second = audio_input_id_per_second

        self._mel_spec_kwargs = dict(
            sample_rate=audio_sampling_rate,
            n_fft=audio_nfft,
            hop_length=audio_hop_length,
            win_length=audio_window_size,
            f_min=audio_fmin,
            f_max=audio_fmax,
            n_mels=audio_n_mels,
            power=1.0,
            center=True,
        )
        self._mel_spectrogram: Any | None = None
        self._resamplers: OrderedDict = OrderedDict()
        self._resamplers_max = 16

        if isinstance(audio_zeroemb_idx, int):
            self.audio_zeroemb_idxs = torch.tensor(
                [audio_zeroemb_idx] * audio_channels, dtype=torch.int32
            )
        else:
            self.audio_zeroemb_idxs = torch.tensor(audio_zeroemb_idx, dtype=torch.int32)

        assert image_min_pixels is not None, "image_min_pixels must be set"
        assert image_max_pixels is not None, "image_max_pixels must be set"
        assert video_min_pixels is not None, "video_min_pixels must be set"
        assert video_max_pixels is not None, "video_max_pixels must be set"
        assert video_total_max_pixels is not None, "video_total_max_pixels must be set"
        assert fps is not None or num_frames is not None, (
            "fps or num_frames must be set"
        )

        for lower, upper in (
            (image_min_pixels, image_max_pixels),
            (video_min_pixels, video_max_pixels),
        ):
            if (
                not isinstance(lower, int)
                or not isinstance(upper, int)
                or lower <= 0
                or upper < lower
            ):
                raise ValueError("MiMo pixel bounds must be positive and ordered")
        if video_total_max_pixels <= 0:
            raise ValueError("MiMo total video pixel budget must be positive")
        if max_frames is not None and (
            not isinstance(max_frames, int) or max_frames <= 0
        ):
            raise ValueError("MiMo max_frames must be positive")
        self._img_kw = {"min_pixels": image_min_pixels, "max_pixels": image_max_pixels}
        self._vid_kw = {
            "min_pixels": video_min_pixels,
            "max_pixels": video_max_pixels,
            "total_max_pixels": video_total_max_pixels,
            "fps": fps,
            "num_frames": num_frames,
            "max_frames": max_frames,
            "min_frames": min_frames,
        }

    @property
    def mel_spectrogram(self) -> Any:
        if self._mel_spectrogram is None:
            if _MelSpectrogram is None:
                raise RuntimeError(
                    "torchaudio is required for audio. "
                    "Install with: pip install torchaudio"
                )
            self._mel_spectrogram = _MelSpectrogram(**self._mel_spec_kwargs)
        return self._mel_spectrogram

    def _resolve_img_kw(self, img: ImageInput) -> dict:
        minimum = (
            img.min_pixels if img.min_pixels is not None else self._img_kw["min_pixels"]
        )
        maximum = (
            img.max_pixels if img.max_pixels is not None else self._img_kw["max_pixels"]
        )
        if (
            not isinstance(minimum, int)
            or not isinstance(maximum, int)
            or minimum <= 0
            or maximum < minimum
            or maximum > self._img_kw["max_pixels"]
        ):
            raise ValueError(
                "Image pixel overrides exceed declared MiMo profile bounds"
            )
        return {"min_px": minimum, "max_px": maximum}

    def _resolve_vid_kw(self, vid: VideoInput) -> dict:
        kw: dict = {}
        for k in ("min_pixels", "max_pixels", "total_max_pixels"):
            value = getattr(vid, k)
            kw[k] = self._vid_kw[k] if value is None else value
            if kw[k] <= 0:
                raise ValueError(f"MiMo video {k} must be positive")
        if vid.num_frames is not None:
            kw["num_frames"] = vid.num_frames
        elif vid.fps is not None:
            kw["fps"] = vid.fps
            if vid.max_frames is not None:
                kw["max_frames"] = vid.max_frames
            if vid.min_frames is not None:
                kw["min_frames"] = vid.min_frames
        elif self._vid_kw["num_frames"] is not None:
            kw["num_frames"] = self._vid_kw["num_frames"]
        elif self._vid_kw["fps"] is not None:
            kw["fps"] = self._vid_kw["fps"]
            if self._vid_kw["max_frames"] is not None:
                kw["max_frames"] = self._vid_kw["max_frames"]
            if self._vid_kw["min_frames"] is not None:
                kw["min_frames"] = self._vid_kw["min_frames"]
        else:
            raise ValueError(
                "No video sampling strategy specified (fps or num_frames)."
            )
        kw["max_frames"] = (
            self._vid_kw["max_frames"] if vid.max_frames is None else vid.max_frames
        )
        return kw

    def preprocess_audio(self, audio: Any) -> tuple[torch.Tensor, int]:
        """Convert a pre-loaded ``(waveform, sr)`` tuple into (mel_spec, token_len)."""
        if not isinstance(audio, tuple):
            raise ValueError(
                f"Unsupported audio source type: {type(audio)}. Audio must be a "
                "pre-decoded (waveform, sample_rate) tuple; URL/path/bytes "
                "resolution is handled upstream in vLLM's media pipeline."
            )
        waveform, original_sr = audio

        target_samples = math.ceil(
            waveform.shape[-1] * self.audio_sampling_rate / original_sr
        )
        channels = waveform.numel() // waveform.shape[-1]
        frames = 1 + target_samples // self._mel_spec_kwargs["hop_length"]
        frequencies = self._mel_spec_kwargs["n_fft"] // 2 + 1
        sample_bytes = waveform.element_size()
        _admit_processor_bytes(
            target_samples * channels * sample_bytes
            + frames * frequencies * (2 * sample_bytes + sample_bytes)
            + frames * self.audio_n_mels * sample_bytes * 3,
            waveform.device,
            "MiMo audio resampled waveform, STFT, mel, clip and log outputs",
        )
        if original_sr != self.audio_sampling_rate:
            if original_sr not in self._resamplers:
                if len(self._resamplers) >= self._resamplers_max:
                    self._resamplers.popitem(last=False)
                self._resamplers[original_sr] = torchaudio.transforms.Resample(
                    orig_freq=original_sr, new_freq=self.audio_sampling_rate
                )
            self._resamplers.move_to_end(original_sr)
            waveform = self._resamplers[original_sr](waveform)

        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        spec = self.mel_spectrogram(waveform[None, :])
        spec = torch.log(torch.clip(spec, min=1e-7)).squeeze().transpose(0, 1)

        n = spec.shape[0]
        n = n + 3 - self.audio_kernel_size
        n = (n + 2 - self.audio_kernel_size) // self.audio_stride_size + 1
        n = n // self.audio_avg_pooler + int(n % self.audio_avg_pooler != 0)
        token_len = math.ceil(n / self.audio_group_size)
        return spec, token_len

    def process_image(self, image: ImageInput) -> torch.Tensor:
        kw = self._resolve_img_kw(image)
        src = image.image
        tensor, _, _ = _transform_single(
            src,
            factor=self.patch_size * self.merge_size,
            device=self.device,
            **kw,
        )
        return tensor

    def video_pixel_budget(self, num_frames: int, kw: dict | None = None) -> int:
        kw = self._vid_kw if kw is None else kw
        if num_frames <= 0:
            raise ValueError("MiMo video requires at least one decoded frame")
        return max(
            kw["min_pixels"],
            min(
                kw["total_max_pixels"]
                * self.temporal_patch_size
                * self.temporal_compression_ratio
                // num_frames,
                kw["max_pixels"],
            ),
        )

    def process_video(
        self, video_input: VideoInput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        kw = self._resolve_vid_kw(video_input)
        video = video_input.video
        if not isinstance(video, tuple):
            raise ValueError(
                f"video must be a (frames_TCHW, timestamps_T) tuple, "
                f"got {type(video)}. "
                "Decode the video before calling the processor."
            )
        frames, timestamps = video
        if (
            video_input.fps is not None
            or video_input.num_frames is not None
            or video_input.min_frames is not None
            or video_input.do_include_last_frame
        ):
            raise ValueError(
                "Decoded-video sampling must be applied by the media pipeline"
            )
        for key in ("min_pixels", "max_pixels", "total_max_pixels"):
            if kw[key] > self._vid_kw[key if key != "min_pixels" else "max_pixels"]:
                raise ValueError(
                    f"Per-item video {key} exceeds profiled processor bounds"
                )
        declared_frames = self._vid_kw["max_frames"]
        if (
            video_input.max_frames is not None
            and declared_frames is not None
            and video_input.max_frames > declared_frames
        ):
            raise ValueError("Per-item max_frames exceeds profiled processor bounds")

        fps = (
            1.0
            if len(timestamps) < 2
            else float(1.0 / (float(timestamps[1]) - float(timestamps[0])))
        )
        start = (
            video_input.start_time
            if video_input.start_time is not None
            else float(timestamps[0])
        )
        end = (
            video_input.end_time
            if video_input.end_time is not None
            else float(timestamps[-1]) + 1.0 / fps
        )

        if video_input.segment_type != "individual":
            mask = (timestamps >= start) & (timestamps < end)
            idxs = torch.where(mask)[0]
            if len(idxs) == 0:
                idxs = torch.where(timestamps <= start)[0][-1:]
            frames, timestamps = frames[idxs], timestamps[idxs]

        tp = self.temporal_patch_size * self.temporal_compression_ratio
        n = frames.shape[0]
        max_frames = kw.get("max_frames", self._vid_kw["max_frames"])
        if max_frames is not None and n > max_frames:
            raise ValueError(
                f"Decoded MiMo video has {n} frames, exceeding configured "
                f"max_frames={max_frames}; adjust the declared processor policy"
            )
        max_px = self.video_pixel_budget(n, kw)

        if n % tp != 0:
            pad = tp - n % tp
            _admit_processor_bytes(
                frames.nbytes
                + 2 * pad * frames[0].nbytes
                + timestamps.nbytes
                + 2 * pad * timestamps.element_size(),
                frames.device,
                "MiMo temporal video padding",
            )
            frames = torch.cat(
                [frames, frames[-1:].repeat(pad, *([1] * (frames.ndim - 1)))],
                dim=0,
            )
            timestamps = torch.cat([timestamps, timestamps[-1:].repeat(pad)], dim=0)

        transformed, _, _ = _transform_batch(
            frames,
            factor=self.patch_size * self.merge_size,
            min_px=kw["min_pixels"],
            max_px=max_px,
            device=self.device,
        )
        patches, thw = self._flatten_visual(transformed, "video")
        meta = {
            "fps_sampled": fps,
            "segment_start_time": start,
            "segment_end_time": end,
        }
        return patches, thw, timestamps, meta

    def process_audio(self, audio: AudioInput) -> Any:
        src = audio.audio
        if isinstance(src, np.ndarray):
            src = (torch.from_numpy(src).float(), self.audio_sampling_rate)
        if isinstance(src, tuple):
            return self.preprocess_audio(src)
        # Pre-tokenized tensor (T, n_vq)
        assert isinstance(src, torch.Tensor) and src.ndim == 2
        T = src.shape[0]
        src = src[:, : self.audio_channels].to(torch.long)
        pad_T = (
            (T + self.audio_group_size - 1)
            // self.audio_group_size
            * self.audio_group_size
        )
        padding = (
            torch.zeros(pad_T - T, self.audio_channels, dtype=torch.long) + src[-1]
        )
        src = torch.cat([src, padding], dim=0)
        return src.reshape(
            pad_T // self.audio_group_size, self.audio_group_size, self.audio_channels
        )

    def _flatten_visual(
        self, visual: torch.Tensor, kind: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        copies = self.temporal_patch_size if kind == "image" else 1
        _admit_processor_bytes(
            visual.nbytes * copies * 2,
            visual.device,
            "MiMo temporal repeat and contiguous vision patches",
        )
        if kind == "image":
            h, w = visual.shape[-2:]
            patches = visual.unsqueeze(0).repeat(self.temporal_patch_size, 1, 1, 1)
        else:  # video / video_audio
            temporal_stride = self.temporal_compression_ratio * self.temporal_patch_size
            assert visual.shape[0] % temporal_stride == 0
            patches = visual
            h, w = patches.shape[-2:]

        C = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = h // self.patch_size, w // self.patch_size

        patches = (
            patches.contiguous()
            .view(
                grid_t,
                self.temporal_patch_size,
                C,
                grid_h // self.merge_size,
                self.merge_size,
                self.patch_size,
                grid_w // self.merge_size,
                self.merge_size,
                self.patch_size,
            )
            .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            .contiguous()
            .view(
                grid_t * grid_h * grid_w,
                C * self.temporal_patch_size * self.patch_size * self.patch_size,
            )
        )
        thw = torch.tensor([grid_t, grid_h, grid_w], dtype=torch.int32)
        return patches, thw

    def process(
        self, contents: list[Content], verbose: bool = False
    ) -> MiMoVLInputSample:
        input_ids: list[int] = []
        labels: list[int] = []
        img_pv: list[torch.Tensor] = []
        img_grids: list[torch.Tensor] = []
        vid_pv: list[torch.Tensor] = []
        vid_grids: list[torch.Tensor] = []
        audio_inputs: list[torch.Tensor] = []
        is_audio_tokenized: list[bool] = []
        audio_token_lens: list[int] = []
        second_per_grid_ts: list[float] = []
        video_start_times: list[float] = []
        video_grid_timestamps: list[torch.Tensor] = []
        va_audio_inputs: list[torch.Tensor] = []
        video_audio_n_segs: list[int] = []
        video_audio_seg_lens: list[int] = []
        video_audio_frame_counts: list[int] = []
        video_audio_start_offsets: list[int] = []

        # Pre-decode videos in parallel
        vid_info = [
            (i, c.content, c.type == "video_audio")
            for i, c in enumerate(contents)
            if c.type in ("video", "video_audio")
        ]
        vid_results: dict[int, tuple] = {}
        if vid_info:
            n_t = min(self.video_process_num_threads, len(vid_info))
            if n_t > 1 and len(vid_info) > 1:
                with ThreadPoolExecutor(max_workers=n_t) as ex:
                    fut_map = {
                        ex.submit(self.process_video, vi): idx
                        for idx, vi, _ in vid_info
                    }
                    for fut in as_completed(fut_map):
                        vid_results[fut_map[fut]] = fut.result()
            else:
                for idx, vi, _ in vid_info:
                    vid_results[idx] = self.process_video(vi)

        for ci, content in enumerate(contents):
            _ids: list[int] = []
            _lbls: list[int] | None = None

            if content.type == "text":
                _ids = (
                    self.tokenizer.encode(content.content)
                    if isinstance(content.content, str)
                    else list(content.content)
                )
                if content.is_target:
                    _lbls = _ids

            elif content.type == "image":
                tensor = self.process_image(content.content)
                patches, thw = self._flatten_visual(tensor, "image")
                t, h, w = thw.tolist()
                n_tok = (t * h * w) // (self.merge_size**2)
                img_pv.append(patches)
                img_grids.append(thw)
                _ids = (
                    [self.vision_start_token_id]
                    + [self.image_token_id] * n_tok
                    + [self.vision_end_token_id]
                )

            elif content.type == "video":
                patches, thw, ts, meta = vid_results[ci]
                t, h, w = thw.tolist()
                n_per_grid = h * w // (self.merge_size**2)
                vid_pv.append(patches)
                vid_grids.append(thw)
                second_per_grid_ts.append(
                    self.temporal_patch_size / meta["fps_sampled"]
                )
                video_start_times.append(float(ts[0]))
                video_grid_timestamps.append(
                    ts[:: self.temporal_patch_size * self.temporal_compression_ratio]
                )
                video_audio_n_segs.append(0)
                video_audio_start_offsets.append(0)

                stride = self.temporal_patch_size * self.temporal_compression_ratio
                ts_texts = [_format_timestamp(float(x)) for x in ts[::stride]]
                ts_ids_list = [self.tokenizer.encode(s) for s in ts_texts]

                _ids = [self.video_start_token_id]
                for ts_ids in ts_ids_list:
                    _ids += (
                        ts_ids
                        + [self.vision_start_token_id]
                        + [self.video_token_id] * n_per_grid
                        + [self.vision_end_token_id]
                    )
                _ids += [self.video_end_token_id]

            elif content.type == "audio":
                processed = self.process_audio(content.content)
                if isinstance(processed, tuple):
                    is_audio_tokenized.append(False)
                    spec, tok_len = processed
                    audio_inputs.append(spec)
                else:
                    is_audio_tokenized.append(True)
                    tok_len = processed.shape[0]
                    audio_inputs.append(processed)
                audio_token_lens.append(tok_len)
                _ids = (
                    [self.audio_start_token_id]
                    + [self.audio_token_id] * tok_len
                    + [self.audio_end_token_id]
                )

            elif content.type == "video_audio":
                patches, thw, ts, meta = vid_results[ci]
                second_per_grid_ts.append(
                    self.temporal_patch_size / meta["fps_sampled"]
                )
                video_start_times.append(float(ts[0]))
                video_grid_timestamps.append(
                    ts[:: self.temporal_patch_size * self.temporal_compression_ratio]
                )
                processed_audio = self.process_audio(content.content)
                tok_per_sec = self.audio_input_id_per_second / self.audio_group_size

                t, h, w = thw.tolist()
                vid_pv.append(patches)
                vid_grids.append(thw)

                if isinstance(processed_audio, tuple):
                    # Mel spec (not pre-tokenized): store in va_audio_inputs separately
                    spec, total_atok = processed_audio
                    va_audio_inputs.append(spec)
                    _va_is_tokenized = False
                else:
                    # Pre-tokenized: not expected in vLLM, but handle defensively
                    total_atok = processed_audio.shape[0]
                    _va_is_tokenized = True
                    va_audio_inputs.append(processed_audio)

                n_per_grid = h * w // (self.merge_size**2)
                stride = self.temporal_patch_size * self.temporal_compression_ratio
                grid_ts = ts[::stride]
                ts_texts = [_format_timestamp(float(x)) for x in grid_ts]
                ts_ids_list = [self.tokenizer.encode(s) for s in ts_texts]

                units: list[tuple] = []
                for i in range(len(grid_ts)):
                    a_start = int(float(grid_ts[i]) * tok_per_sec)
                    a_end = (
                        int(float(grid_ts[i + 1]) * tok_per_sec)
                        if i < len(grid_ts) - 1
                        else int(meta["segment_end_time"] * tok_per_sec)
                    )
                    seg_len = min(a_end, total_atok) - a_start
                    assert seg_len > 0, f"Zero-length audio segment at grid index {i}"
                    seg = (
                        processed_audio[a_start : a_start + seg_len]
                        if _va_is_tokenized
                        else None
                    )
                    units.append(
                        (
                            float(grid_ts[i]),
                            ts_texts[i],
                            ts_ids_list[i],
                            n_per_grid,
                            seg_len,
                            seg,
                        )
                    )

                il = self.video_audio_interleave_length
                if il == -1:
                    groups: list[list] = [list(enumerate(units))]
                elif il == 0:
                    groups = [[(i, u)] for i, u in enumerate(units)]
                else:
                    groups, cur, t_ptr = [], [], 0.0
                    for i, u in enumerate(units):
                        while u[0] >= t_ptr + il:
                            if cur:
                                groups.append(cur)
                                cur = []
                            t_ptr += il
                        cur.append((i, u))
                    if cur:
                        groups.append(cur)

                # Track n_segs (= num groups) and per-group audio token counts
                video_audio_n_segs.append(len(groups))
                video_audio_start_offsets.append(int(float(grid_ts[0]) * tok_per_sec))
                for group in groups:
                    video_audio_frame_counts.append(len(group))
                    group_seg_len = sum(u[4] for _, u in group)
                    video_audio_seg_lens.append(group_seg_len)

                _ids = [self.video_start_token_id]
                for group in groups:
                    _ids += group[0][1][2]  # first-unit timestamp token ids
                    _vid_tok: list[int] = []
                    _aud_tok: list[int] = []
                    for _, u in group:
                        _, _, _, vid_n, seg_n, seg_audio = u
                        _vid_tok += (
                            [self.vision_start_token_id]
                            + [self.video_token_id] * vid_n
                            + [self.vision_end_token_id]
                        )
                        _aud_tok += [self.audio_token_id] * seg_n
                        if seg_audio is not None:
                            # Pre-tokenized per-frame segments (rare in vLLM)
                            audio_inputs.append(seg_audio)
                    _ids += (
                        _vid_tok
                        + [self.audio_start_token_id]
                        + _aud_tok
                        + [self.audio_end_token_id]
                    )
                _ids += [self.video_end_token_id]

            input_ids.extend(_ids)
            labels.extend(
                _lbls if _lbls is not None else [self.pad_token_id] * len(_ids)
            )

        ids_t = torch.tensor(input_ids)
        lbl_arr = np.roll(labels, shift=-1)
        lbl_arr[-1] = self.pad_token_id
        lbl_t = torch.tensor(lbl_arr)

        extra: dict = {}
        if is_audio_tokenized:
            assert all(is_audio_tokenized) or not any(is_audio_tokenized)
            extra["is_audio_tokenized"] = is_audio_tokenized[0]

        position_ids = torch.arange(ids_t.shape[0]).expand(3, -1)
        rope_deltas = torch.zeros((1, 1), dtype=torch.int32)

        return MiMoVLInputSample(
            input_ids=ids_t,
            labels=lbl_t,
            pixel_values=img_pv,
            pixel_values_videos=vid_pv,
            image_thw_grids=img_grids,
            video_thw_grids=vid_grids,
            audio_inputs=audio_inputs,
            second_per_grid_ts=second_per_grid_ts,
            video_start_times=video_start_times,
            video_grid_timestamps=video_grid_timestamps,
            audio_token_lens=audio_token_lens,
            va_audio_inputs=va_audio_inputs,
            video_audio_n_segs=video_audio_n_segs,
            video_audio_seg_lens=video_audio_seg_lens,
            video_audio_frame_counts=video_audio_frame_counts,
            video_audio_start_offsets=video_audio_start_offsets,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            extra=extra,
        )


# ---------------------------------------------------------------------------
# vLLM ProcessorMixin wrapper
# ---------------------------------------------------------------------------


class MiMoOmniProcessor(ProcessorMixin):
    """HuggingFace-compatible ProcessorMixin wrapper for MiMo-Omni.

    Accepts PIL images, pre-decoded video tuples (frames_TCHW, timestamps_T),
    and audio (file path / (waveform, sr) tuple / numpy array).
    """

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    # Single or multi-pad placeholders produced by the chat template / prior expansion
    _IMG_RE = re.compile(r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>")
    _VID_RE = re.compile(r"<\|vision_start\|>(?:<\|video_pad\|>)+<\|vision_end\|>")
    _AUD_RE = re.compile(
        r"<\|mimo_audio_start\|>(?:<\|audio_pad\|>)+<\|mimo_audio_end\|>"
    )

    _MM_RE = re.compile(
        r"(<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
        r"|<\|vision_start\|>(?:<\|video_pad\|>)+<\|vision_end\|>"
        r"|<\|mimo_audio_start\|>(?:<\|audio_pad\|>)+<\|mimo_audio_end\|>)"
    )

    def __init__(
        self,
        tokenizer: Any,
        *,
        patch_size: int = 14,
        merge_size: int = 2,
        temporal_patch_size: int = 2,
        temporal_compression_ratio: int = 1,
        image_min_pixels: int | None = None,
        image_max_pixels: int | None = None,
        video_min_pixels: int | None = None,
        video_max_pixels: int | None = None,
        video_total_max_pixels: int | None = None,
        fps: float = 2.0,
        num_frames: int | None = None,
        max_frames: int = 256,
        min_frames: int = 8,
        video_audio_interleave_length: int = 0,
        audio_sampling_rate: int = 24000,
        audio_nfft: int = 960,
        audio_hop_length: int = 240,
        audio_window_size: int = 960,
        audio_fmin: float = 0.0,
        audio_fmax: float | None = None,
        audio_n_mels: int = 128,
        audio_segment_size: int = 6000,
        audio_kernel_size: int = 3,
        audio_stride_size: int = 2,
        audio_avg_pooler: int = 2,
        audio_channels: int = 8,
        audio_group_size: int = 4,
        audio_input_id_per_second: float = 25.0,
        audio_zeroemb_idx: int = 4096,
        image_token_id: int | None = None,
        video_token_id: int | None = None,
        audio_token_id: int | None = None,
        vision_start_token_id: int | None = None,
        vision_end_token_id: int | None = None,
        audio_start_token_id: int | None = None,
        audio_end_token_id: int | None = None,
        video_start_token_id: int | None = None,
        video_end_token_id: int | None = None,
        rope_type: str = "rope",
    ) -> None:
        self.tokenizer = tokenizer
        self._constructor_kwargs = {
            name: value
            for name, value in locals().items()
            if name not in ("self", "tokenizer", "__class__")
        }

        unit = patch_size * merge_size
        self.mimo_processor = MiMoVLProcessor(
            tokenizer=tokenizer,
            patch_size=patch_size,
            merge_size=merge_size,
            temporal_patch_size=temporal_patch_size,
            temporal_compression_ratio=temporal_compression_ratio,
            use_video_timestamps=True,
            video_audio_interleave_length=video_audio_interleave_length,
            audio_sampling_rate=audio_sampling_rate,
            audio_nfft=audio_nfft,
            audio_hop_length=audio_hop_length,
            audio_window_size=audio_window_size,
            audio_fmin=audio_fmin,
            audio_fmax=audio_fmax,
            audio_n_mels=audio_n_mels,
            audio_segment_size=audio_segment_size,
            audio_kernel_size=audio_kernel_size,
            audio_stride_size=audio_stride_size,
            audio_avg_pooler=audio_avg_pooler,
            audio_channels=audio_channels,
            audio_group_size=audio_group_size,
            audio_input_id_per_second=audio_input_id_per_second,
            audio_zeroemb_idx=audio_zeroemb_idx,
            image_min_pixels=(4 * unit * unit)
            if image_min_pixels is None
            else image_min_pixels,
            image_max_pixels=(4096 * unit * unit)
            if image_max_pixels is None
            else image_max_pixels,
            video_min_pixels=(4 * unit * unit)
            if video_min_pixels is None
            else video_min_pixels,
            video_max_pixels=(4096 * unit * unit)
            if video_max_pixels is None
            else video_max_pixels,
            video_total_max_pixels=(16384 * unit * unit)
            if video_total_max_pixels is None
            else video_total_max_pixels,
            fps=fps,
            num_frames=num_frames,
            max_frames=max_frames,
            min_frames=min_frames,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            audio_token_id=audio_token_id,
            vision_start_token_id=vision_start_token_id,
            vision_end_token_id=vision_end_token_id,
            audio_start_token_id=audio_start_token_id,
            audio_end_token_id=audio_end_token_id,
            video_start_token_id=video_start_token_id,
            video_end_token_id=video_end_token_id,
            pad_token_id=tokenizer.pad_token_id,
            rope_type=rope_type,
        )

    def with_overrides(self, kwargs: dict[str, Any]) -> "MiMoOmniProcessor":
        if not kwargs:
            return self
        overrides = dict(kwargs)
        for modality, nested_name in (
            ("image", "images_kwargs"),
            ("video", "videos_kwargs"),
            ("audio", "audio_kwargs"),
        ):
            nested = dict(overrides.pop(nested_name, {}))
            nested_size = nested.pop("size", None)
            if nested_size is not None:
                for key, alias in (
                    ("shortest_edge", "min_pixels"),
                    ("longest_edge", "max_pixels"),
                ):
                    if key in nested_size:
                        if alias in nested and nested[alias] != nested_size[key]:
                            raise ValueError(f"Conflicting MiMo {nested_name} size")
                        nested[alias] = nested_size[key]
                if set(nested_size) - {"shortest_edge", "longest_edge"}:
                    raise ValueError(f"Unsupported MiMo {nested_name} size")
            for key, value in nested.items():
                if key in ("min_pixels", "max_pixels", "total_max_pixels"):
                    key = f"{modality}_{key}"
                if key in overrides and overrides[key] != value:
                    raise ValueError(f"Conflicting MiMo processor override: {key}")
                overrides[key] = value
        size = overrides.pop("size", None)
        if size is not None:
            unknown = set(size) - {"shortest_edge", "longest_edge"}
            if unknown:
                raise ValueError(f"Unsupported MiMo size fields: {sorted(unknown)}")
            for key, alias in (
                ("shortest_edge", "min_pixels"),
                ("longest_edge", "max_pixels"),
            ):
                if key in size:
                    if alias in overrides and overrides[alias] != size[key]:
                        raise ValueError(
                            f"Conflicting MiMo processor override: {alias}"
                        )
                    overrides[alias] = size[key]
        for alias in ("min_pixels", "max_pixels"):
            if alias in overrides:
                value = overrides.pop(alias)
                for modality in ("image", "video"):
                    name = f"{modality}_{alias}"
                    if name in overrides and overrides[name] != value:
                        raise ValueError(f"Conflicting MiMo processor override: {name}")
                    overrides[name] = value
        sampling = {"fps", "num_frames", "min_frames"}
        if any(
            name in overrides and overrides[name] != self._constructor_kwargs[name]
            for name in sampling
        ):
            raise ValueError(
                "Decoded-video sampling overrides belong in media_io_kwargs, "
                "not mm_processor_kwargs"
            )
        accepted = set(inspect.signature(type(self).__init__).parameters) - {
            "self",
            "tokenizer",
        }
        unknown = set(overrides) - accepted
        if unknown:
            raise ValueError(f"Unsupported MiMo processor overrides: {sorted(unknown)}")
        # Checkpoint geometry is not a resizable preprocessing option.
        structural = {
            "patch_size",
            "merge_size",
            "temporal_patch_size",
            "temporal_compression_ratio",
            "audio_channels",
            "audio_group_size",
            "audio_kernel_size",
            "audio_stride_size",
            "audio_avg_pooler",
        }
        for name in structural & overrides.keys():
            if overrides[name] != self._constructor_kwargs[name]:
                raise ValueError(f"MiMo checkpoint geometry cannot override {name}")
        return type(self)(self.tokenizer, **(self._constructor_kwargs | overrides))

    @classmethod
    def from_hf_config(
        cls,
        tokenizer: Any,
        hf_config: Any,
        **kwargs: Any,
    ) -> "MiMoOmniProcessor":
        """Instantiate directly from an HF model config object."""
        vc = hf_config.vision_config
        if isinstance(vc, dict):
            patch_size = vc.get("patch_size", 14)
            merge_size = vc.get("spatial_merge_size", 2)
            temporal_patch_size = vc.get("temporal_patch_size", 2)
        else:
            patch_size = getattr(vc, "patch_size", 14)
            merge_size = getattr(vc, "spatial_merge_size", 2)
            temporal_patch_size = getattr(vc, "temporal_patch_size", 2)

        pc: dict = getattr(hf_config, "processor_config", {}) or {}
        ac = getattr(hf_config, "audio_config", None)
        audio_sr: int | None = pc.get("audio_sampling_rate")
        if audio_sr is None and ac is not None:
            if isinstance(ac, dict):
                audio_sr = ac.get("sampling_rate") or ac.get("sample_rate")
            else:
                audio_sr = getattr(ac, "sampling_rate", None) or getattr(
                    ac, "sample_rate", None
                )

        rope_type = "rope"
        rs = getattr(hf_config, "rope_scaling", None)
        if rs and rs.get("type") == "default" and rs.get("mrope_section") is not None:
            rope_type = "mrope"

        unit = patch_size * merge_size
        processor = cls(
            tokenizer,
            patch_size=patch_size,
            merge_size=merge_size,
            temporal_patch_size=temporal_patch_size,
            temporal_compression_ratio=pc.get("temporal_compression_ratio", 1),
            image_min_pixels=pc.get("image_min_pixels", 4 * unit * unit),
            image_max_pixels=pc.get("image_max_pixels", 4096 * unit * unit),
            video_min_pixels=pc.get("video_min_pixels", 4 * unit * unit),
            video_max_pixels=pc.get("video_max_pixels", 4096 * unit * unit),
            video_total_max_pixels=(
                pc.get("video_total_max_pixels", 16384 * unit * unit)
            ),
            fps=pc.get("fps", 2.0),
            num_frames=pc.get("num_frames"),
            max_frames=pc.get("max_frames", 256),
            min_frames=pc.get("min_frames", 8),
            video_audio_interleave_length=pc.get("video_audio_interleave_length", 0),
            audio_sampling_rate=audio_sr or 24000,
            audio_nfft=pc.get("audio_nfft", 960),
            audio_hop_length=pc.get("audio_hop_length", 240),
            audio_window_size=pc.get("audio_window_size", 960),
            audio_fmin=pc.get("audio_fmin", 0.0),
            audio_fmax=pc.get("audio_fmax"),
            audio_n_mels=pc.get("audio_n_mels", 128),
            audio_segment_size=pc.get("audio_segment_size", 6000),
            audio_kernel_size=pc.get("audio_kernel_size", 3),
            audio_stride_size=pc.get("audio_stride_size", 2),
            audio_avg_pooler=pc.get("audio_avg_pooler", 2),
            audio_channels=pc.get("audio_channels", 8),
            audio_group_size=pc.get("audio_group_size", 4),
            audio_input_id_per_second=pc.get("audio_input_id_per_second", 25.0),
            audio_zeroemb_idx=pc.get("audio_zeroemb_idx", 4096),
            image_token_id=pc.get("image_token_id"),
            video_token_id=pc.get("video_token_id"),
            audio_token_id=pc.get("audio_token_id"),
            vision_start_token_id=pc.get("vision_start_token_id"),
            vision_end_token_id=pc.get("vision_end_token_id"),
            audio_start_token_id=pc.get("audio_start_token_id"),
            audio_end_token_id=pc.get("audio_end_token_id"),
            video_start_token_id=pc.get("video_start_token_id"),
            video_end_token_id=pc.get("video_end_token_id"),
            rope_type=rope_type,
        )
        return processor.with_overrides(kwargs)

    @property
    def image_token(self) -> str:
        """Token string used as image placeholder (for vLLM integration)."""
        return "<|image_pad|>"

    @property
    def video_token(self) -> str:
        """Token string used as video placeholder (for vLLM integration)."""
        return "<|video_pad|>"

    @property
    def image_processor(self) -> Any:
        """Minimal image-processor-like object for vLLM processing-info compat."""
        p = self.mimo_processor

        class _ImageProcessor:
            merge_size = p.merge_size
            size = {
                "shortest_edge": p._img_kw["min_pixels"],
                "longest_edge": p._img_kw["max_pixels"],
            }

        return _ImageProcessor()

    def _modality(self, token: str) -> str:
        if self._IMG_RE.fullmatch(token):
            return "image"
        if self._VID_RE.fullmatch(token):
            return "video"
        if self._AUD_RE.fullmatch(token):
            return "audio"
        return "unknown"

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Any = None,
        videos: Any = None,
        audio: Any = None,
        video_audio: Any = None,
        return_tensors: str | TensorType | None = None,
        **kwargs: Any,
    ) -> BatchFeature:
        """Process multimodal inputs into model-ready tensors.

        Args:
            text: Prompt string(s) containing multimodal placeholders
                  ``<|vision_start|><|image_pad|><|vision_end|>``,
                  ``<|vision_start|><|video_pad|><|vision_end|>``, or
                  ``<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>``.
            images: PIL.Image or list[PIL.Image].
            videos: list of ``(frames_TCHW: torch.Tensor, timestamps_T: torch.Tensor)``
                    tuples (pre-decoded).
            audio: list of ``str`` (path/url/base64), ``bytes``,
                   ``(waveform_1D, sample_rate)`` tuples, or ``np.ndarray``.
            video_audio: audio track extracted from ``videos``, in the same
                   formats accepted by ``audio``.
            return_tensors: Passed to :class:`BatchFeature`.

        Returns:
            :class:`BatchFeature` with keys:
            - ``input_ids``
            - ``pixel_values`` + ``image_grid_thw``
            - ``pixel_values_videos`` + ``video_grid_thw`` + ``second_per_grid_ts``
            - ``audio_features``

        """
        if kwargs:
            return self.with_overrides(kwargs)(
                text=text,
                images=images,
                videos=videos,
                audio=audio,
                video_audio=video_audio,
                return_tensors=return_tensors,
            )
        if isinstance(text, list):
            text = text[0] if len(text) == 1 else "\n".join(text)

        imgs: list = (
            ([images] if isinstance(images, Image.Image) else list(images))
            if images is not None
            else []
        )
        vids: list = list(videos) if videos is not None else []
        auds: list = list(audio) if audio is not None else []
        va_items: list = list(video_audio) if video_audio is not None else []

        # If audio exists but text has no audio placeholder, prepend one
        _aud_placeholder = "<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>"
        if auds and text is not None and not self._AUD_RE.search(text):
            text = _aud_placeholder + text

        # Build Content list
        contents: list[Content] = []

        if text and (imgs or vids or auds or va_items):
            parts = self._MM_RE.split(text)
            img_it = iter(imgs)
            vid_it = iter(vids)
            aud_it = iter(auds)
            va_it = iter(va_items)
            for part in parts:
                if self._MM_RE.fullmatch(part):
                    mod = self._modality(part)
                    if mod == "image":
                        with contextlib.suppress(StopIteration):
                            contents.append(
                                Content(
                                    type="image",
                                    content=ImageInput(image=next(img_it)),
                                )
                            )
                    elif mod == "video":
                        # Try regular video first, fall back to video_audio
                        vid_item = None
                        vid_type = "video"
                        with contextlib.suppress(StopIteration):
                            vid_item = next(vid_it)
                        if vid_item is None:
                            with contextlib.suppress(StopIteration):
                                vid_item = next(va_it)
                                vid_type = "video_audio"
                        if isinstance(vid_item, VideoAudioInput):
                            vid_type = "video_audio"
                        if vid_item is not None:
                            if vid_type == "video":
                                contents.append(
                                    Content(
                                        type="video",
                                        content=VideoInput(video=vid_item),
                                    )
                                )
                            else:
                                contents.append(
                                    Content(
                                        type="video_audio",
                                        content=vid_item,
                                    )
                                )
                    elif mod == "audio":
                        with contextlib.suppress(StopIteration):
                            contents.append(
                                Content(
                                    type="audio",
                                    content=AudioInput(audio=next(aud_it)),
                                )
                            )
                elif part:
                    contents.append(Content(type="text", content=part))
        elif text:
            contents.append(Content(type="text", content=text))
        else:
            for img in imgs:
                contents.append(Content(type="image", content=ImageInput(image=img)))
            for vid in vids:
                contents.append(
                    Content(
                        type="video_audio"
                        if isinstance(vid, VideoAudioInput)
                        else "video",
                        content=vid
                        if isinstance(vid, VideoAudioInput)
                        else VideoInput(video=vid),
                    )
                )
            for aud in auds:
                contents.append(Content(type="audio", content=AudioInput(audio=aud)))
            for va_item in va_items:
                contents.append(Content(type="video_audio", content=va_item))

        if not contents:
            ids = self.tokenizer(text or "", return_tensors=return_tensors)["input_ids"]
            return BatchFeature(data={"input_ids": ids}, tensor_type=return_tensors)

        sample = self.mimo_processor.process(contents, verbose=False)

        # vLLM expects input_ids to have a batch dimension [1, seq_len].
        data: dict = {"input_ids": sample.input_ids.unsqueeze(0)}

        if sample.pixel_values:
            _admit_processor_bytes(
                sum(tensor.nbytes for tensor in sample.pixel_values),
                sample.pixel_values[0].device,
                "MiMo pixel_values batching",
            )
            data["pixel_values"] = torch.cat(sample.pixel_values, dim=0)
            data["image_grid_thw"] = torch.stack(sample.image_thw_grids)

        if sample.pixel_values_videos:
            _admit_processor_bytes(
                sum(tensor.nbytes for tensor in sample.pixel_values_videos),
                sample.pixel_values_videos[0].device,
                "MiMo pixel_values_videos batching",
            )
            data["pixel_values_videos"] = torch.cat(sample.pixel_values_videos, dim=0)
            data["video_grid_thw"] = torch.stack(sample.video_thw_grids)
            if sample.second_per_grid_ts:
                data["second_per_grid_ts"] = torch.tensor(
                    sample.second_per_grid_ts, dtype=torch.float32
                )
            if sample.video_grid_timestamps:
                data["video_grid_timestamps"] = sample.video_grid_timestamps
            if sample.video_start_times:
                data["video_start_times"] = torch.tensor(
                    sample.video_start_times, dtype=torch.float32
                )
            if sample.video_audio_n_segs:
                data["video_audio_n_segs"] = torch.tensor(
                    sample.video_audio_n_segs, dtype=torch.long
                )
            # video_audio_seg_lens: 2D padded tensor (num_videos, max_T).
            # Row i has the per-group audio token lengths for video i
            # (zeros for regular videos; valid values for video_audio videos).
            n_segs_list = sample.video_audio_n_segs
            max_segs = max(n_segs_list) if n_segs_list else 0
            if max_segs > 0:
                seg_lens_2d = torch.zeros(len(n_segs_list), max_segs, dtype=torch.long)
                frame_counts_2d = torch.zeros_like(seg_lens_2d)
                flat_cursor = 0
                for vi, n in enumerate(n_segs_list):
                    if n > 0:
                        seg_lens_2d[vi, :n] = torch.tensor(
                            sample.video_audio_seg_lens[flat_cursor : flat_cursor + n],
                            dtype=torch.long,
                        )
                        frame_counts_2d[vi, :n] = torch.tensor(
                            sample.video_audio_frame_counts[
                                flat_cursor : flat_cursor + n
                            ],
                            dtype=torch.long,
                        )
                        flat_cursor += n
                data["video_audio_seg_lens"] = seg_lens_2d
                data["video_audio_frame_counts"] = frame_counts_2d
                data["video_audio_start_offsets"] = torch.tensor(
                    sample.video_audio_start_offsets,
                    dtype=torch.long,
                )

        # audio_features is a list of variable-length mel-spec tensors; pop it
        # before BatchFeature conversion to avoid "batched tensors of the same
        # length" errors, then re-attach it after.
        audio_features = None
        if sample.audio_inputs:
            audio_features = sample.audio_inputs
            if "is_audio_tokenized" in sample.extra:
                data["is_audio_tokenized"] = sample.extra["is_audio_tokenized"]
            if sample.audio_token_lens:
                data["audio_token_lens"] = torch.tensor(
                    sample.audio_token_lens, dtype=torch.long
                )

        bf = BatchFeature(data=data, tensor_type=return_tensors)
        if audio_features is not None:
            bf["audio_features"] = audio_features
        # va_audio_features: list of mel-spec tensors (one per video_audio item)
        if sample.va_audio_inputs:
            audio_iter = iter(sample.va_audio_inputs)
            bf["va_audio_features"] = [
                next(audio_iter)
                if count
                else torch.empty((0, self.mimo_processor.audio_n_mels))
                for count in sample.video_audio_n_segs
            ]
        return bf
