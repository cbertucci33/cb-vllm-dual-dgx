# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for stale multiproc RPC deadlines."""

import time
from collections import deque
from concurrent.futures import Future, InvalidStateError
from contextlib import suppress
from typing import cast
from unittest.mock import patch


def _dequeue_timeout(deadline: float | None) -> float | None:
    return None if deadline is None else max(0.0, deadline - time.monotonic())


class FutureWrapper(Future):
    def __init__(
        self,
        futures_queue: deque["FutureWrapper"],
        get_response,
        aggregate=lambda x: x,
    ):
        self.futures_queue = futures_queue
        self.get_response = get_response
        self.aggregate = aggregate
        super().__init__()
        self.futures_queue.appendleft(self)

    def result(self, timeout=None):
        if timeout is not None:
            raise RuntimeError("timeout not implemented")

        while not self.done():
            future = self.futures_queue.pop()
            future._wait_for_response()
        return super().result()

    def _wait_for_response(self):
        try:
            response = self.aggregate(self.get_response())
            with suppress(InvalidStateError):
                self.set_result(response)
        except Exception as e:
            with suppress(InvalidStateError):
                self.set_exception(e)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def monotonic(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class FakeResponseMQ:
    def __init__(self, response: object = ("SUCCESS", "dummy_response")) -> None:
        self.response = response
        self.timeouts: list[float | None] = []

    def dequeue(self, timeout: float | None = None) -> object:
        assert timeout is None or timeout >= 0.0, (
            f"dequeue received negative timeout: {timeout}"
        )
        self.timeouts.append(timeout)
        return self.response


def test_future_wrapper_stale_deadline_never_passes_negative_timeout() -> None:
    clock = FakeClock(100.0)
    futures_queue: deque[FutureWrapper] = deque()
    response_mq = FakeResponseMQ()

    deadline = clock.monotonic() + 1.0

    def get_response() -> object:
        return response_mq.dequeue(timeout=_dequeue_timeout(deadline))

    future = FutureWrapper(
        futures_queue,
        get_response=get_response,
        aggregate=lambda x: x,
    )

    clock.advance(5.0)

    with patch("time.monotonic", clock.monotonic):
        future.result()

    assert response_mq.timeouts == [0.0]


def test_future_wrapper_non_expired_deadline_passes_positive_timeout() -> None:
    clock = FakeClock(100.0)
    futures_queue: deque[FutureWrapper] = deque()
    response_mq = FakeResponseMQ()

    deadline = clock.monotonic() + 5.0

    def get_response() -> object:
        return response_mq.dequeue(timeout=_dequeue_timeout(deadline))

    future = FutureWrapper(
        futures_queue,
        get_response=get_response,
        aggregate=lambda x: x,
    )

    clock.advance(2.0)

    with patch("time.monotonic", clock.monotonic):
        future.result()

    assert len(response_mq.timeouts) == 1
    timeout = response_mq.timeouts[0]
    assert timeout is not None
    assert timeout > 0.0
    assert timeout <= 5.0


def test_future_wrapper_deadline_none_passes_none() -> None:
    futures_queue: deque[FutureWrapper] = deque()
    response_mq = FakeResponseMQ()

    def get_response() -> object:
        return response_mq.dequeue(timeout=_dequeue_timeout(None))

    future = FutureWrapper(
        futures_queue,
        get_response=get_response,
        aggregate=lambda x: x,
    )

    future.result()

    assert response_mq.timeouts == [None]


def test_future_wrapper_drains_pending_before_own_get_response() -> None:
    futures_queue: deque[FutureWrapper] = deque()
    call_order: list[str] = []

    def make_get_response(label: str):
        def _get() -> object:
            call_order.append(label)
            return FakeResponseMQ().dequeue(timeout=None)

        return _get

    first = FutureWrapper(futures_queue, make_get_response("first"))
    second = FutureWrapper(futures_queue, make_get_response("second"))

    second.result()

    assert first.done()
    assert second.done()
    assert call_order == ["first", "second"]


def test_recv_timeout_ms_clamps_negative_timeout() -> None:
    def recv_timeout_ms(timeout: float | None) -> int | None:
        return None if timeout is None else max(0, int(timeout * 1000))

    assert recv_timeout_ms(None) is None
    assert recv_timeout_ms(-1.0) == 0
    assert recv_timeout_ms(-0.001) == 0
    assert recv_timeout_ms(0.0) == 0
    assert recv_timeout_ms(0.001) == 1
    assert recv_timeout_ms(2.5) == 2500


def _runtime_method(relative_path, class_name, method_name, namespace):
    """Execute the real method body without importing model/CUDA modules."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    tree = ast.parse((root / relative_path).read_text())
    owner = (
        tree
        if class_name is None
        else next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            method,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, relative_path, "exec"), namespace)
    return namespace[method_name]


def test_cancelled_writer_never_touches_shared_buffer():
    """Cancellation interrupts a writer even when readers never acknowledge."""
    from types import SimpleNamespace

    import pytest

    acquire = _runtime_method(
        "vllm/distributed/device_communicators/shm_broadcast.py",
        "MessageQueue",
        "acquire_write",
        {"time": time},
    )
    writer = SimpleNamespace(_is_writer=True, shutting_down=True)
    with pytest.raises(RuntimeError, match="cancelled"):
        next(acquire(writer))


def test_remote_poll_observes_shutdown_before_receiving():
    """Remote wait checks shutdown after poll, not just local spin cancellation."""
    from types import SimpleNamespace

    import pytest

    zmq = SimpleNamespace(POLLIN=1)

    class Socket:
        def poll(self, **kwargs):
            queue.shutting_down = True
            return True

    wait = _runtime_method(
        "vllm/distributed/device_communicators/shm_broadcast.py",
        "MessageQueue",
        "_wait_socket",
        {"time": time, "zmq": zmq, "SHM_READER_RECHECK_INTERVAL_MS": 5000},
    )
    queue = SimpleNamespace(shutting_down=False)
    with pytest.raises(RuntimeError, match="cancelled"):
        wait(queue, Socket(), zmq.POLLIN, None)


def test_remote_send_deadline_expires_without_blocking_send():
    from types import SimpleNamespace

    import pytest

    wait = _runtime_method(
        "vllm/distributed/device_communicators/shm_broadcast.py",
        "MessageQueue",
        "_wait_socket",
        {"time": time, "SHM_READER_RECHECK_INTERVAL_MS": 5000},
    )
    queue = SimpleNamespace(shutting_down=False)
    with pytest.raises(TimeoutError):
        wait(queue, object(), 2, time.monotonic() - 1)


def test_fault_cleanup_aborts_before_worker_and_continues_after_errors():
    """One failing stage cannot suppress later distributed cleanup."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    events = []
    concrete = SimpleNamespace()

    def worker_cleanup():
        events.append("worker")
        assert concrete._shutdown_due_to_failure
        raise RuntimeError("cleanup failure")

    shutdown = _runtime_method(
        "vllm/v1/executor/multiproc_executor.py",
        "WorkerProc",
        "shutdown",
        {
            "logger": Mock(),
            "abort_pending_collectives": lambda: events.append("abort"),
            "destroy_model_parallel": lambda: events.append("groups"),
            "destroy_distributed_environment": lambda: events.append("world"),
        },
    )
    worker = SimpleNamespace(
        rpc_broadcast_mq=SimpleNamespace(shutdown=lambda: events.append("rpc")),
        worker_response_mq=SimpleNamespace(shutdown=lambda: events.append("response")),
        worker=SimpleNamespace(worker=concrete, shutdown=worker_cleanup),
    )
    shutdown(worker, failed=True)
    assert events == ["rpc", "response", "abort", "worker", "groups", "world"]
    assert worker.rpc_broadcast_mq is None
    assert worker.worker_response_mq is None


def test_audio_encoder_never_requests_a_generation_cache():
    """Even a changed HF config cannot reactivate unused audio K/V caching."""
    from types import SimpleNamespace

    seen = {}

    def transformer(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(last_hidden_state="audio output")

    apply = _runtime_method(
        "vllm/model_executor/models/mimo_audio.py",
        "MimoAudioEncoder",
        "apply_input_local_transformer",
        {
            "check_startup_memory": lambda **kwargs: None,
            "record_startup_memory_peak": lambda **kwargs: None,
        },
    )
    encoder = SimpleNamespace(
        input_local_transformer=transformer,
        config=SimpleNamespace(
            input_full_attention=True,
            use_cache=True,
            input_local_intermediate_size=4096,
            input_local_attn_heads=16,
        ),
    )
    embeddings = SimpleNamespace(
        shape=(2, 4, 1024), element_size=lambda: 2, device="cpu"
    )
    assert apply(encoder, embeddings) == "audio output"
    assert seen == {
        "inputs_embeds": embeddings,
        "use_cache": False,
        "output_hidden_states": False,
        "output_attentions": False,
        "return_dict": True,
        "is_causal": False,
    }


def test_rpc_send_timeout_fails_executor_and_cleans_up():
    """An expired broadcast cannot leave its response in the next RPC stream."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    import pytest

    events = []

    class Queue:
        def enqueue(self, request, timeout):
            events.append(timeout)
            raise TimeoutError("stalled reader")

    rpc = _runtime_method(
        "vllm/v1/executor/multiproc_executor.py",
        "MultiprocExecutor",
        "collective_rpc",
        {"time": time, "logger": Mock()},
    )
    executor = SimpleNamespace(
        rpc_broadcast_mq=Queue(),
        is_failed=False,
        _startup_time_remaining=lambda: None,
        shutdown=lambda: events.append("shutdown"),
    )
    with pytest.raises(TimeoutError, match="stalled reader"):
        rpc(executor, "profile", timeout=1)
    assert 0 < events[0] <= 1
    assert events[1:] == ["shutdown"]
    assert executor.is_failed


def test_startup_deadline_is_consumed_once_and_cleared_for_serving():
    from types import SimpleNamespace

    clock = FakeClock(100)
    remaining = _runtime_method(
        "vllm/v1/executor/multiproc_executor.py",
        "MultiprocExecutor",
        "_startup_time_remaining",
        {"time": clock},
    )
    finish = _runtime_method(
        "vllm/v1/executor/multiproc_executor.py",
        "MultiprocExecutor",
        "finish_startup",
        {},
    )
    executor = SimpleNamespace(_startup_deadline=700)
    assert remaining(executor) == 600
    clock.advance(200)
    assert remaining(executor) == 400
    clock.advance(500)
    assert remaining(executor) == 0
    finish(executor)
    assert remaining(executor) is None


def test_stalled_startup_ready_pipe_times_out_and_closes():
    from types import SimpleNamespace

    import pytest

    seen: list[float | str] = []

    class Pipe:
        def close(self):
            seen.append("closed")

    def wait(pipes, timeout):
        seen.append(timeout)
        return []

    ready = _runtime_method(
        "vllm/v1/executor/multiproc_executor.py",
        "WorkerProc",
        "wait_for_ready",
        {
            "time": time,
            "multiprocessing": SimpleNamespace(connection=SimpleNamespace(wait=wait)),
        },
    )
    with pytest.raises(TimeoutError, match="engine-ready timeout"):
        ready([SimpleNamespace(ready_pipe=Pipe())], timeout=2)
    assert 0 < cast(float, seen[0]) <= 2
    assert seen[1:] == ["closed"]


def test_video_intermediate_admission_precedes_float_and_resize():
    from types import SimpleNamespace

    import pytest

    seen = []

    def reject(nbytes, device, label):
        seen.append((nbytes, device))
        raise MemoryError("joint budget exceeded")

    transform = _runtime_method(
        "vllm/transformers_utils/processors/mimo_v2_omni.py",
        None,
        "_transform_batch",
        {
            "_smart_resize": lambda h, w, *args: (h, w),
            "_admit_processor_bytes": reject,
        },
    )
    frames = SimpleNamespace(
        shape=(2, 3, 32, 32),
        numel=lambda: 6144,
        nbytes=6144,
        device="cpu",
    )
    with pytest.raises(MemoryError, match="joint budget"):
        transform(frames, 32, 32, 8192)
    assert seen == [(73728, "cpu")]


def test_h2d_joint_admission_precedes_destination_allocation():
    import sys
    from types import SimpleNamespace
    from unittest.mock import patch

    import pytest

    seen = []

    def reject(**kwargs):
        seen.append(kwargs)
        raise MemoryError("joint budget exceeded")

    torch = SimpleNamespace(device=lambda device: SimpleNamespace(type="cuda"))
    copy = _runtime_method(
        "vllm/utils/torch_utils.py",
        "PinnedStagingBuffer",
        "copy_to_device",
        {"torch": torch},
    )
    tensor = SimpleNamespace(nbytes=100, is_contiguous=lambda: False)
    with (
        patch.dict(
            sys.modules,
            {"vllm.utils.mem_utils": SimpleNamespace(check_startup_memory=reject)},
        ),
        pytest.raises(MemoryError, match="joint budget"),
    ):
        copy(object(), tensor, "cuda:0")
    assert seen[0]["device_bytes"] == 100
    assert seen[0]["host_bytes"] == 100


def test_legacy_fault_cleanup_skips_sync_but_graceful_still_syncs():
    from types import SimpleNamespace
    from unittest.mock import Mock

    for failed in (False, True):
        calls: list[str] = []
        torch = SimpleNamespace(
            accelerator=SimpleNamespace(
                synchronize=lambda calls=calls: calls.append("sync"),
                empty_cache=lambda calls=calls: calls.append("empty"),
            )
        )
        cleanup = _runtime_method(
            "vllm/v1/worker/gpu_model_runner.py",
            "GPUModelRunner",
            "_cleanup_profiling_kv_cache",
            {
                "torch": torch,
                "gc": SimpleNamespace(collect=lambda: None),
                "logger": Mock(),
            },
        )
        runner = SimpleNamespace(
            _shutdown_due_to_failure=failed,
            kv_caches=[object()],
            attn_groups=[object()],
            kv_cache_config=object(),
            cache_config=SimpleNamespace(num_gpu_blocks=1),
            compilation_config=SimpleNamespace(static_forward_context={}),
        )
        cleanup(runner)
        assert calls == (["empty"] if failed else ["sync", "empty"])
        assert runner.kv_caches == []
        assert runner.attn_groups == []
        assert not hasattr(runner, "kv_cache_config")


def test_audio_local_config_cannot_allocate_a_default_token_vocabulary():
    """Execute the constructor's actual config expression, not a copied config."""
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[3]
    tree = ast.parse((root / "vllm/model_executor/models/mimo_audio.py").read_text())
    encoder = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MimoAudioEncoder"
    )
    constructor = next(
        node
        for node in encoder.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    config_call = next(
        node.value
        for node in constructor.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "input_local_config"
            for target in node.targets
        )
    )
    fields = {
        node.attr: 1
        for node in ast.walk(config_call)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "config"
    }
    config = eval(
        compile(ast.Expression(config_call), "mimo_audio.py", "eval"),
        {"config": SimpleNamespace(**fields), "Qwen2Config": lambda **kwargs: kwargs},
    )
    assert config["vocab_size"] == 0
    assert config["pad_token_id"] is None
    assert config["use_cache"] is False


def test_legacy_sync_error_latches_fault_and_still_releases_cache():
    from types import SimpleNamespace
    from unittest.mock import Mock

    import pytest

    def fail_sync():
        raise RuntimeError("original sync failure")

    torch = SimpleNamespace(
        accelerator=SimpleNamespace(
            synchronize=fail_sync,
            empty_cache=lambda: None,
        )
    )
    cleanup = _runtime_method(
        "vllm/v1/worker/gpu_model_runner.py",
        "GPUModelRunner",
        "_cleanup_profiling_kv_cache",
        {"torch": torch, "gc": SimpleNamespace(collect=lambda: None), "logger": Mock()},
    )
    runner = SimpleNamespace(
        kv_caches=[object()],
        attn_groups=[object()],
        cache_config=SimpleNamespace(num_gpu_blocks=1),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    with pytest.raises(RuntimeError, match="original sync failure"):
        cleanup(runner)
    assert runner._shutdown_due_to_failure
    assert runner.kv_caches == []
    assert runner.attn_groups == []


def _rpc_response_error_case(rank_responses):
    from types import SimpleNamespace
    from unittest.mock import Mock

    class Queue:
        def __init__(self, responses=()):
            self.responses = deque(responses)

        def enqueue(self, request, timeout=None):
            pass

        def dequeue(self, timeout=None):
            response = self.responses.popleft()
            if isinstance(response, Exception):
                raise response
            return response

    rpc = _runtime_method(
        "vllm/v1/executor/multiproc_executor.py",
        "MultiprocExecutor",
        "collective_rpc",
        {
            "time": time,
            "logger": Mock(),
            "FutureWrapper": FutureWrapper,
            "WorkerProc": SimpleNamespace(
                ResponseStatus=SimpleNamespace(SUCCESS=0, FAILURE=1)
            ),
        },
    )
    executor = SimpleNamespace(
        rpc_broadcast_mq=Queue(),
        response_mqs=[Queue(responses) for responses in rank_responses],
        is_failed=False,
        _startup_time_remaining=lambda: None,
        futures_queue=deque(),
        shutdown=Mock(),
    )
    return lambda: rpc(executor, "profile"), executor


def test_worker_error_drains_all_replies_before_executor_reuse():
    """A recoverable utility error cannot shift replies into the next RPC."""
    from unittest import TestCase

    rpc, executor = _rpc_response_error_case(
        [
            [(1, "root failure"), (0, "new rank0")],
            [(0, "old rank1"), (0, "new rank1")],
        ]
    )
    with TestCase().assertRaisesRegex(RuntimeError, "root failure"):
        rpc()
    assert rpc() == ["new rank0", "new rank1"]
    assert all(not queue.responses for queue in executor.response_mqs)
    assert not executor.is_failed
    executor.shutdown.assert_not_called()


def test_worker_error_then_drain_timeout_preserves_root_and_fails_closed():
    from unittest import TestCase

    rpc, executor = _rpc_response_error_case(
        [
            [(1, "root failure")],
            [TimeoutError("remaining rank stalled")],
        ]
    )
    with TestCase().assertRaisesRegex(RuntimeError, "root failure") as caught:
        rpc()
    assert isinstance(caught.exception.__cause__, TimeoutError)
    assert executor.is_failed
    executor.shutdown.assert_called_once()


def _worker_rpc_fixture(rank, fail):
    import traceback
    from functools import partial
    from types import SimpleNamespace
    from unittest.mock import Mock

    fn = _runtime_method(
        "vllm/v1/executor/multiproc_executor.py",
        "WorkerProc",
        "_execute_worker_rpc",
        {"traceback": traceback, "logger": Mock(), "partial": partial},
    )
    outputs: list[str] = []
    root = RuntimeError("non-reply worker root failure")

    def call():
        if fail:
            raise root
        return "output"

    worker = SimpleNamespace(
        rank=rank,
        worker=SimpleNamespace(execute_model=call, sample_tokens=call),
        handle_output=outputs.append,
    )
    return fn, worker, outputs, root


def test_non_reply_worker_exception_is_not_swallowed():
    import pytest

    for rank in (0, 1):
        for method in ("execute_model", "sample_tokens"):
            fn, worker, outputs, root = _worker_rpc_fixture(rank, True)
            with pytest.raises(
                RuntimeError, match="non-reply worker root failure"
            ) as e:
                fn(worker, (method, (), {}, 1 - rank))
            assert e.value is root
            assert outputs == []  # no unsolicited reply to corrupt a later RPC
            assert "Traceback" in root.__notes__[0]


def test_selected_reply_worker_error_keeps_rpc_failure_transport():
    for rank in (0, 1):
        for output_rank in (rank, None):
            fn, worker, outputs, root = _worker_rpc_fixture(rank, True)
            fn(worker, ("execute_model", (), {}, output_rank))
            assert outputs == [root]


def test_successful_unique_reply_rpc_keeps_nonreply_silent():
    for rank in (0, 1):
        fn, worker, outputs, root = _worker_rpc_fixture(rank, False)
        fn(worker, ("sample_tokens", (), {}, 0))
        assert outputs == (["output"] if rank == 0 else [])
