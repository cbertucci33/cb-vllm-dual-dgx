# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Barrier

import numpy as np
import pytest
import torch

from vllm.utils import torch_utils
from vllm.utils.torch_utils import (
    OMP_NUM_THREADS_SET_BY_VLLM,
    PinnedStagingBuffer,
    async_tensor_h2d,
    available_cpu_count,
    common_broadcastable_dtype,
    current_stream,
    get_kv_cache_torch_dtype,
    is_lossless_cast,
    is_quantized_kv_cache,
    set_default_torch_dtype,
    set_random_seed,
    set_torch_threads_for_runtime,
    startup_omp_num_threads,
)


def test_nvfp4_4over6_cache_dtype() -> None:
    from vllm.config.cache import CacheConfig
    from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

    cache_config = CacheConfig(cache_dtype="nvfp4_4over6")

    assert cache_config.cache_dtype == "nvfp4_4over6"
    assert get_kv_cache_torch_dtype(cache_config.cache_dtype) == torch.uint8
    assert is_quantized_kv_cache(cache_config.cache_dtype)
    assert get_kv_quant_mode(cache_config.cache_dtype) == KVQuantMode.NVFP4


def test_set_default_torch_dtype_restores_dtype_after_exception() -> None:
    original_dtype = torch.get_default_dtype()

    with (
        pytest.raises(RuntimeError, match="expected failure"),
        set_default_torch_dtype(torch.bfloat16),
    ):
        assert torch.get_default_dtype() == torch.bfloat16
        raise RuntimeError("expected failure")

    assert torch.get_default_dtype() == original_dtype


@pytest.mark.parametrize(
    ("src_dtype", "tgt_dtype", "expected_result"),
    [
        # Different precision_levels
        (torch.bool, torch.int8, True),
        (torch.bool, torch.float16, True),
        (torch.bool, torch.complex32, True),
        (torch.int64, torch.bool, False),
        (torch.int64, torch.float16, True),
        (torch.int64, torch.complex32, True),
        (torch.float64, torch.bool, False),
        (torch.float64, torch.int8, False),
        (torch.float64, torch.complex32, True),
        (torch.complex128, torch.bool, False),
        (torch.complex128, torch.int8, False),
        (torch.complex128, torch.float16, False),
        # precision_level=0
        (torch.bool, torch.bool, True),
        # precision_level=1
        (torch.int8, torch.int16, True),
        (torch.int16, torch.int8, False),
        (torch.uint8, torch.int8, False),
        (torch.int8, torch.uint8, False),
        # precision_level=2
        (torch.float16, torch.float32, True),
        (torch.float32, torch.float16, False),
        (torch.bfloat16, torch.float32, True),
        (torch.float32, torch.bfloat16, False),
        # precision_level=3
        (torch.complex32, torch.complex64, True),
        (torch.complex64, torch.complex32, False),
    ],
)
def test_is_lossless_cast(src_dtype, tgt_dtype, expected_result):
    assert is_lossless_cast(src_dtype, tgt_dtype) == expected_result


@pytest.mark.parametrize(
    ("dtypes", "expected_result"),
    [
        ([torch.bool], torch.bool),
        ([torch.bool, torch.int8], torch.int8),
        ([torch.bool, torch.int8, torch.float16], torch.float16),
        ([torch.bool, torch.int8, torch.float16, torch.complex32], torch.complex32),  # noqa: E501
    ],
)
def test_common_broadcastable_dtype(dtypes, expected_result):
    assert common_broadcastable_dtype(dtypes) == expected_result


def _test_stream_thread(main_expected_stream: torch.cuda.Stream):
    import threading

    child_stream = torch.cuda.Stream()
    thread_stream_ready = threading.Event()
    thread_can_exit = threading.Event()

    def child_thread_func():
        with torch.cuda.stream(child_stream):
            thread_stream_ready.set()
            thread_can_exit.wait(timeout=10)

    child_thread = threading.Thread(target=child_thread_func)
    child_thread.start()

    try:
        assert thread_stream_ready.wait(timeout=5), (
            "Child thread failed to enter stream context in time"
        )

        main_current_stream = current_stream()

        assert main_current_stream != child_stream, (
            "Main thread's current_stream was contaminated by child thread"
        )
        assert main_current_stream == main_expected_stream, (
            f"Main thread's stream changed unexpectedly. "
            f"Expected {main_expected_stream}, got {main_current_stream}"
        )

        thread_can_exit.set()

    finally:
        child_thread.join(timeout=5)
        if child_thread.is_alive():
            pytest.fail("Child thread failed to exit properly")


def test_current_stream_multithread():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    main_dedicated_stream = current_stream()

    assert main_dedicated_stream.cuda_stream != 0, (
        "ROCm/CUDA should create a dedicated stream, not use default stream (0x0)"
    )

    main_stream_again = current_stream()
    assert main_stream_again == main_dedicated_stream, (
        "Multiple calls to current_stream should return the same dedicated stream"
    )

    _test_stream_thread(main_dedicated_stream)


@pytest.fixture
def restore_torch_threads(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    original = torch.get_num_threads()
    yield
    torch.set_num_threads(original)


def test_startup_omp_num_threads_divides_between_local_workers():
    """Workers share the node's usable CPUs rather than each taking them all."""
    available = available_cpu_count()
    if available < 4:
        pytest.skip("needs at least 4 usable CPUs")
    assert startup_omp_num_threads(1) == available
    assert startup_omp_num_threads(2) == available // 2
    # Never zero, however many workers share the node.
    assert startup_omp_num_threads(available * 4) == 1


def test_set_torch_threads_for_runtime(restore_torch_threads):
    torch.set_num_threads(max(2, available_cpu_count()))
    set_torch_threads_for_runtime()
    assert torch.get_num_threads() == 1


def test_runtime_threads_respect_user_omp_num_threads(
    restore_torch_threads, monkeypatch: pytest.MonkeyPatch
):
    """An externally-set OMP_NUM_THREADS is the user's choice; leave it alone."""
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    torch.set_num_threads(3)
    set_torch_threads_for_runtime()
    assert torch.get_num_threads() == 3


def test_runtime_threads_override_vllm_set_omp_num_threads(
    restore_torch_threads, monkeypatch: pytest.MonkeyPatch
):
    """The value vLLM picked for worker startup is dropped once serving starts."""
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv(OMP_NUM_THREADS_SET_BY_VLLM, "1")
    torch.set_num_threads(3)
    set_torch_threads_for_runtime()
    assert torch.get_num_threads() == 1


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_async_tensor_h2d_staging(device):
    """Staging must preserve values while normalizing dtype and layout."""
    # Non-contiguous source with a dtype conversion: would previously sync.
    src = torch.arange(12, dtype=torch.int64).reshape(3, 4).T
    result = async_tensor_h2d(src, device=device, dtype=torch.int32)
    assert result.dtype == torch.int32
    assert result.is_contiguous()
    assert torch.equal(result.cpu(), src.to(torch.int32))

    # numpy and list sources still work.
    assert torch.equal(
        async_tensor_h2d(np.arange(4, dtype=np.int64), device=device),
        torch.arange(4, device=device),
    )
    assert torch.equal(
        async_tensor_h2d([1, 2, 3], device=device, dtype=torch.int32),
        torch.tensor([1, 2, 3], dtype=torch.int32, device=device),
    )


def test_set_random_seed_differs_per_data_parallel_index():
    """Unseeded requests draw from the global RNGs; DP engines sharing the engine
    seed must get distinct, reproducible streams, and index 0 keeps the seed."""

    def draws(*args):
        set_random_seed(*args)
        return np.random.randint(2**31, size=4).tolist(), torch.rand(4).tolist()

    assert draws(42, 0) == draws(42)
    assert draws(42, 1) == draws(42, 1)
    assert len({str(draws(42, i)) for i in range(4)}) == 4
    assert draws(42, 1) != draws(43, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_pinned_staging_buffer():
    capacity = 1 << 20
    staging = PinnedStagingBuffer(capacity)
    pinned_before = torch.cuda.host_memory_stats().get("allocated_bytes.current", 0)
    srcs = [torch.randn(n) for n in (1, 1000, 100_000, 300_000, 1_000_000)]
    srcs.append(torch.randn(256, 512).T)
    outs = [staging.copy_to_device(x, "cuda") for _ in range(3) for x in srcs]
    for out, x in zip(outs, srcs * 3):
        assert torch.equal(out.cpu(), x)
    pinned_after = torch.cuda.host_memory_stats()["allocated_bytes.current"]
    assert pinned_after - pinned_before <= capacity
    assert staging.empty((capacity + 1,), torch.uint8) is None


def _mock_cuda_events(monkeypatch: pytest.MonkeyPatch):
    events = []

    class FakeStream:
        def __init__(self, device):
            self.device = torch.device(device)
            self.synchronized = False

        def synchronize(self):
            self.synchronized = True

    class FakeEvent:
        def __init__(self, device):
            self.device = torch.device(device)
            self.stream = None
            self.synchronized = False
            events.append(self)

        def record(self, stream):
            self.stream = stream

        def query(self):
            return self.synchronized

        def synchronize(self):
            self.synchronized = True

    monkeypatch.setattr(torch, "Event", FakeEvent)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device=None: FakeStream(device),
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext(stream))
    monkeypatch.setattr(
        torch_utils,
        "_pinned_uint8_buffer",
        lambda capacity: torch.empty(capacity, dtype=torch.uint8),
    )
    return events


def test_pinned_staging_buffer_uses_destination_device_stream(
    monkeypatch: pytest.MonkeyPatch,
):
    events = _mock_cuda_events(monkeypatch)
    real_empty = torch.empty
    requested_devices = []

    def fake_empty(*args, **kwargs):
        device = kwargs.get("device")
        if device is not None and torch.device(device).type == "cuda":
            requested_devices.append(torch.device(device))
            kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", fake_empty)
    destination = torch.device("cuda:1")
    source = torch.arange(32, dtype=torch.int32)
    staging = PinnedStagingBuffer(4096)

    output = staging.copy_to_device(source, destination)

    assert torch.equal(output, source)
    assert requested_devices == [destination]
    assert len(events) == 1
    assert events[0].device == destination
    assert events[0].stream.device == destination


def test_pinned_staging_buffer_concurrent_leases_and_event_reuse(
    monkeypatch: pytest.MonkeyPatch,
):
    events = _mock_cuda_events(monkeypatch)
    staging = PinnedStagingBuffer(1024)
    barrier = Barrier(4)

    def reserve_and_record(index: int) -> int:
        tensor = staging.empty((128,), torch.uint8)
        assert tensor is not None
        barrier.wait()
        staging.record(tensor, torch.device(f"cuda:{index}"))
        return tensor.data_ptr()

    with ThreadPoolExecutor(max_workers=4) as pool:
        pointers = list(pool.map(reserve_and_record, range(4)))

    assert len(set(pointers)) == 4
    assert len(events) == 4
    assert {event.device for event in events} == {
        torch.device(f"cuda:{index}") for index in range(4)
    }
    assert all(event.stream.device == event.device for event in events)

    # The ring is full. Reusing its first slot must synchronize the matching
    # event before returning overlapping host storage.
    first_events = tuple(events)
    reused = staging.empty((1,), torch.uint8)
    assert reused is not None
    assert any(event.synchronized for event in first_events)


def test_pinned_staging_buffer_does_not_lease_zero_sized_regions(
    monkeypatch: pytest.MonkeyPatch,
):
    _mock_cuda_events(monkeypatch)
    staging = PinnedStagingBuffer(1024)

    assert staging.empty((0, 4), torch.float32) is None
    assert not staging._regions


def test_pinned_staging_buffer_releases_lease_after_copy_error(
    monkeypatch: pytest.MonkeyPatch,
):
    _mock_cuda_events(monkeypatch)
    real_empty = torch.empty

    def fake_empty(*args, **kwargs):
        if torch.device(kwargs.get("device", "cpu")).type == "cuda":
            kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    def fail_copy(*args, **kwargs):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(torch, "empty", fake_empty)
    monkeypatch.setattr(torch.Tensor, "copy_", fail_copy)
    staging = PinnedStagingBuffer(1024)

    with pytest.raises(RuntimeError, match="copy failed"):
        staging.copy_to_device(torch.arange(16), torch.device("cuda:1"))

    assert not staging._regions


def test_pinned_staging_buffer_drains_stream_before_release_on_record_error(
    monkeypatch: pytest.MonkeyPatch,
):
    _mock_cuda_events(monkeypatch)
    real_empty = torch.empty
    streams = []

    class TrackingStream:
        def __init__(self, device):
            self.device = torch.device(device)
            self.synchronized = False
            streams.append(self)

        def synchronize(self):
            self.synchronized = True

    def fake_empty(*args, **kwargs):
        if torch.device(kwargs.get("device", "cpu")).type == "cuda":
            kwargs["device"] = "cpu"
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", fake_empty)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device=None: TrackingStream(device),
    )
    staging = PinnedStagingBuffer(1024)
    monkeypatch.setattr(
        staging,
        "record",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("record failed")),
    )

    with pytest.raises(RuntimeError, match="record failed"):
        staging.copy_to_device(torch.arange(16), torch.device("cuda:1"))

    assert len(streams) == 1
    assert streams[0].synchronized
    assert not staging._regions
