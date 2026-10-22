# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from time import sleep

import pytest
import torch
from PIL import Image

from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
    MultiModalSharedField,
    PlaceholderRange,
    nested_tensors_equal,
)
from vllm.multimodal.utils import (
    argsort_mm_positions,
    encode_image_url,
    group_and_batch_mm_items,
)


@pytest.mark.parametrize(
    "case",
    [
        # Single modality
        ## Internally sorted
        dict(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=0, length=2),
                    PlaceholderRange(offset=3, length=2),
                ]
            },
            expected_modality_idxs=[
                ("image", 0),
                ("image", 1),
            ],
        ),
        ## Internally unsorted
        dict(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=3, length=2),
                    PlaceholderRange(offset=0, length=2),
                ]
            },
            expected_modality_idxs=[
                ("image", 1),
                ("image", 0),
            ],
        ),
        # Two modalities
        ## Internally sorted
        dict(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=7, length=4),
                    PlaceholderRange(offset=11, length=5),
                ],
                "audio": [
                    PlaceholderRange(offset=0, length=2),
                    PlaceholderRange(offset=2, length=3),
                ],
            },
            expected_modality_idxs=[
                ("audio", 0),
                ("audio", 1),
                ("image", 0),
                ("image", 1),
            ],
        ),
        ## Interleaved, internally sorted
        dict(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=0, length=4),
                    PlaceholderRange(offset=8, length=2),
                ],
                "audio": [
                    PlaceholderRange(offset=5, length=2),
                    PlaceholderRange(offset=11, length=4),
                ],
            },
            expected_modality_idxs=[
                ("image", 0),
                ("audio", 0),
                ("image", 1),
                ("audio", 1),
            ],
        ),
        ## Interleaved, internally unsorted
        dict(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=8, length=2),
                    PlaceholderRange(offset=0, length=4),
                ],
                "audio": [
                    PlaceholderRange(offset=11, length=4),
                    PlaceholderRange(offset=5, length=2),
                ],
            },
            expected_modality_idxs=[
                ("image", 1),
                ("audio", 1),
                ("image", 0),
                ("audio", 0),
            ],
        ),
        # Three modalities
        ## Internally sorted
        dict(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=15, length=7),
                    PlaceholderRange(offset=22, length=8),
                ],
                "audio": [
                    PlaceholderRange(offset=0, length=2),
                ],
                "video": [
                    PlaceholderRange(offset=3, length=4),
                    PlaceholderRange(offset=7, length=5),
                    PlaceholderRange(offset=12, length=6),
                ],
            },
            expected_modality_idxs=[
                ("audio", 0),
                ("video", 0),
                ("video", 1),
                ("video", 2),
                ("image", 0),
                ("image", 1),
            ],
        ),
        ## Interleaved, internally sorted
        dict(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=0, length=2),
                    PlaceholderRange(offset=2, length=3),
                    PlaceholderRange(offset=20, length=4),
                ],
                "audio": [
                    PlaceholderRange(offset=5, length=2),
                ],
                "video": [
                    PlaceholderRange(offset=8, length=5),
                ],
            },
            expected_modality_idxs=[
                ("image", 0),
                ("image", 1),
                ("audio", 0),
                ("video", 0),
                ("image", 2),
            ],
        ),
        ## Interleaved, internally unsorted
        dict(
            mm_positions={
                "image": [
                    PlaceholderRange(offset=0, length=2),
                    PlaceholderRange(offset=20, length=4),
                    PlaceholderRange(offset=2, length=3),
                ],
                "audio": [
                    PlaceholderRange(offset=5, length=2),
                ],
                "video": [
                    PlaceholderRange(offset=8, length=5),
                ],
            },
            expected_modality_idxs=[
                ("image", 0),
                ("image", 2),
                ("audio", 0),
                ("video", 0),
                ("image", 1),
            ],
        ),
    ],
)
def test_argsort_mm_positions(case):
    mm_positions = case["mm_positions"]
    expected_modality_idxs = case["expected_modality_idxs"]

    modality_idxs = argsort_mm_positions(mm_positions)

    assert modality_idxs == expected_modality_idxs


def test_group_and_batch_mm_items_split_by_fieldset():
    elem = MultiModalFieldElem(
        data=torch.empty(1, dtype=torch.uint8),
        field=MultiModalBatchedField(),
    )
    item1 = MultiModalKwargsItem({"x": elem, "y": elem})
    item2 = MultiModalKwargsItem({"y": elem, "x": elem})
    item3 = MultiModalKwargsItem({"x": elem, "y": elem, "z": elem})
    item4 = MultiModalKwargsItem({"x": elem})
    item5 = MultiModalKwargsItem({"x": elem, "y": elem})

    res = group_and_batch_mm_items([item1, item2, item3, item4, item5])
    assert [num_items for num_items, _ in res] == [2, 1, 1, 1]


def test_group_and_batch_mm_items_splits_shared_data_by_dtype():
    elem1 = MultiModalFieldElem(
        data=torch.zeros(1, dtype=torch.int32),
        field=MultiModalSharedField(batch_size=1),
    )
    elem2 = MultiModalFieldElem(
        data=torch.zeros(1, dtype=torch.float32),
        field=MultiModalSharedField(batch_size=1),
    )
    elem3 = MultiModalFieldElem(
        data=[torch.zeros(1, dtype=torch.int32), torch.zeros(1, dtype=torch.float32)],
        field=MultiModalSharedField(batch_size=1),
    )

    res = group_and_batch_mm_items(
        [
            MultiModalKwargsItem({"x": elem1}),
            MultiModalKwargsItem({"x": elem2}),
            MultiModalKwargsItem({"x": elem3}),
        ]
    )

    assert [num_items for num_items, _ in res] == [1, 1, 1]


def test_encode_image_url_uncommon_format_has_valid_mimetype():
    """A format missing from mimetypes.types_map must still produce a
    well-formed type/subtype media type, not a bare 'image'."""
    # ".tga" is not registered in mimetypes.types_map, so it exercises the
    # fallback path; Pillow can still encode it.
    url = encode_image_url(Image.new("RGB", (2, 2)), format="TGA")
    mimetype = url.removeprefix("data:").split(";", 1)[0]
    assert mimetype == "image/tga"


def test_group_and_batch_mm_items_split_by_shared_data():
    elem1 = MultiModalFieldElem(
        data=torch.zeros(1, dtype=torch.uint8),
        field=MultiModalSharedField(batch_size=1),
    )
    elem2 = MultiModalFieldElem(
        data=torch.zeros(2, dtype=torch.uint8),
        field=MultiModalSharedField(batch_size=1),
    )
    item1 = MultiModalKwargsItem({"x": elem1})
    item2 = MultiModalKwargsItem({"x": elem1})
    item3 = MultiModalKwargsItem({"x": elem2})
    item4 = MultiModalKwargsItem({"x": elem1})
    item5 = MultiModalKwargsItem({"x": elem2})

    res = group_and_batch_mm_items([item1, item2, item3, item4, item5])
    assert [num_items for num_items, _ in res] == [2, 1, 1, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("num_items", [1, 3])
def test_group_and_batch_mm_items_pinned_h2d(num_items):
    items = [
        MultiModalKwargsItem(
            {
                "stacked": MultiModalFieldElem(
                    torch.randn(4, 8), MultiModalBatchedField()
                ),
                "ragged": MultiModalFieldElem(
                    torch.randn(i + 1, 3).T, MultiModalBatchedField()
                ),
                "concat": MultiModalFieldElem(
                    torch.randn(i + 2, 8),
                    MultiModalFlatField(slices=[slice(0, i + 2)]),
                ),
                "padded": MultiModalFieldElem(
                    torch.randn(2, i + 4),
                    MultiModalFlatField(slices=[slice(0, 2)]),
                ),
            }
        )
        for i in range(num_items)
    ]
    [(_, staged)] = group_and_batch_mm_items(items, device="cuda", pin_memory=True)
    [(_, direct)] = group_and_batch_mm_items(items, device="cuda")
    assert staged.keys() == direct.keys()
    for key, value in direct.items():
        assert nested_tensors_equal(staged[key], value), key


def test_xpu_mm_transfer_keeps_generic_pinned_copy_path(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.multimodal.inputs as mm_inputs

    pin_requests = []
    transfers = []
    original_new_empty = torch.Tensor.new_empty

    def fake_new_empty(tensor, *args, **kwargs):
        pin_requests.append(kwargs.pop("pin_memory", False))
        return original_new_empty(tensor, *args, **kwargs)

    def fake_to(tensor, *args, **kwargs):
        transfers.append((kwargs.get("device"), kwargs.get("non_blocking")))
        return tensor

    def unexpected_cuda_use(*args, **kwargs):
        raise AssertionError("XPU staging must not use CUDA streams or events")

    monkeypatch.setattr(torch.Tensor, "new_empty", fake_new_empty)
    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda tensor: True)
    monkeypatch.setattr(torch.cuda, "current_stream", unexpected_cuda_use)
    monkeypatch.setattr(torch.cuda, "stream", unexpected_cuda_use)
    monkeypatch.setattr(torch, "Event", unexpected_cuda_use)
    monkeypatch.setattr(mm_inputs, "_get_pinned_staging_buffer", unexpected_cuda_use)

    items = [
        MultiModalKwargsItem(
            {
                "pixel_values": MultiModalFieldElem(
                    torch.full((2, 3), value), MultiModalBatchedField()
                )
            }
        )
        for value in (1, 2)
    ]
    [(_, grouped)] = group_and_batch_mm_items(
        items, device=torch.device("xpu"), pin_memory=True
    )

    assert pin_requests == [True]
    assert transfers == [(torch.device("xpu"), True)]
    expected = torch.tensor([[[1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2]]])
    assert torch.equal(grouped["pixel_values"], expected)


def test_mm_batching_releases_staging_lease_after_stack_error(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.multimodal.inputs as mm_inputs
    from vllm.utils import torch_utils
    from vllm.utils.torch_utils import PinnedStagingBuffer

    staging = PinnedStagingBuffer(4096)
    monkeypatch.setattr(
        torch_utils,
        "_pinned_uint8_buffer",
        lambda capacity: torch.empty(capacity, dtype=torch.uint8),
    )
    monkeypatch.setattr(mm_inputs, "_pinned_staging_buffer", staging)

    def fail_stack(*args, **kwargs):
        raise RuntimeError("stack failed")

    monkeypatch.setattr(torch, "stack", fail_stack)
    items = [
        MultiModalFieldElem(torch.ones(2, 3), MultiModalBatchedField()),
        MultiModalFieldElem(torch.zeros(2, 3), MultiModalBatchedField()),
    ]

    with pytest.raises(RuntimeError, match="stack failed"):
        MultiModalBatchedField().reduce_data(
            items, device=torch.device("cuda"), pin_memory=True
        )

    assert not staging._regions


def test_mm_h2d_drains_stream_before_release_on_record_error(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.multimodal.inputs as mm_inputs
    from vllm.utils import torch_utils
    from vllm.utils.torch_utils import PinnedStagingBuffer

    class TrackingStream:
        synchronized = False

        def synchronize(self):
            self.synchronized = True

    stream = TrackingStream()
    staging = PinnedStagingBuffer(4096)
    monkeypatch.setattr(
        torch_utils,
        "_pinned_uint8_buffer",
        lambda capacity: torch.empty(capacity, dtype=torch.uint8),
    )
    staged = staging.empty((2, 3), torch.float32)
    assert staged is not None
    monkeypatch.setattr(mm_inputs, "_pinned_staging_buffer", staging)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda tensor: True)
    monkeypatch.setattr(torch.Tensor, "to", lambda tensor, **kwargs: tensor.clone())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda selected: nullcontext(selected))
    monkeypatch.setattr(
        staging,
        "record",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("record failed")),
    )

    with pytest.raises(RuntimeError, match="record failed"):
        mm_inputs._nested_tensors_h2d(
            staged, device=torch.device("cuda:1"), pin_memory=True
        )

    assert stream.synchronized
    assert not staging._regions


def test_mm_pinned_staging_singleton_is_thread_safe(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.multimodal.inputs as mm_inputs
    from vllm.utils import torch_utils

    created = []

    class FakeStagingBuffer:
        def __init__(self, capacity):
            sleep(0.01)
            self.capacity = capacity
            created.append(self)

    monkeypatch.setattr(mm_inputs, "_pinned_staging_buffer", None)
    monkeypatch.setattr(torch_utils, "PinnedStagingBuffer", FakeStagingBuffer)

    with ThreadPoolExecutor(max_workers=8) as pool:
        buffers = list(
            pool.map(lambda _: mm_inputs._get_pinned_staging_buffer(), range(8))
        )

    assert len(created) == 1
    assert all(buffer is created[0] for buffer in buffers)
