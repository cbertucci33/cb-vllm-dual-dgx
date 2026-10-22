# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import __future__

import ast
import hashlib
import math
import os
import sys
import time
import unittest
import warnings
from collections.abc import Iterable
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast

repo = Path(__file__).resolve().parents[3]
dependency = (
    Path(os.environ["INSTANTTENSOR_SOURCE_DIR"])
    if "INSTANTTENSOR_SOURCE_DIR" in os.environ
    else None
)


def extract(
    path: Path, name: str, namespace: dict[str, Any], cls: str | None = None
) -> Any:
    tree = ast.parse(path.read_text())
    scope = (
        tree.body
        if cls is None
        else next(
            n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
        )
    )
    node = next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[name]


class Source:
    def __init__(self, fail_close=False, fail_read=False):
        self.position = 0
        self.closed = 0
        self.fail_close = fail_close
        self.fail_read = fail_read

    def __iter__(self):
        return self

    def __next__(self):
        if self.fail_read:
            raise ValueError("read error")
        if self.position == 2:
            raise StopIteration
        self.position += 1
        return "weight" + str(self.position), object()

    def close(self):
        self.closed += 1
        if self.fail_close:
            raise RuntimeError("close error")


class IteratorCleanup(unittest.TestCase):
    def setUp(self):
        self.sources: list[Source] = []
        ns: dict[str, Any] = {
            "contextmanager": contextmanager,
            "sys": sys,
            "time": time,
            "cast": cast,
            "Iterable": Iterable,
            "refresh_derived_weights": extract(
                repo / "vllm/model_executor/model_loader/derived_weights.py",
                "refresh_derived_weights",
                {},
            ),
        }
        exec(
            (repo / "vllm/model_executor/model_loader/iterator_utils.py").read_text(),
            ns,
        )
        ns.update(
            pt_weights_iterator=lambda *_: self.new_source(),
            logger=SimpleNamespace(info_once=lambda *_: None),
            DefaultModelLoader=SimpleNamespace(
                Source=lambda *a, **kw: SimpleNamespace(
                    model_or_path=a[0], revision=a[1], subfolder=None, **kw
                )
            ),
        )
        file = repo / "vllm/model_executor/model_loader/default_loader.py"
        for name in ["_get_weights_iterator", "get_all_weights", "load_weights"]:
            ns[name] = extract(file, name, ns, "DefaultModelLoader")
        prepared = (
            ("folder", ["file"], False, None)
            if "index_file = (" in file.read_text()
            else ("folder", ["file"], False)
        )
        self.loader = SimpleNamespace(
            load_config=SimpleNamespace(
                model_loader_extra_config={},
                load_format="pt",
                use_tqdm_on_load=False,
                pt_load_map_location="cpu",
            ),
            counter_before_loading_weights=0.0,
            enable_weights_track=False,
            _encoder_only_lm_prefixes=None,
            _prepare_weights=lambda *_: prepared,
            _init_ep_weight_filter=lambda *_: None,
            _init_mm_encoder_only_weight_filter=lambda *_: None,
        )
        for name in ["_get_weights_iterator", "get_all_weights", "load_weights"]:
            setattr(self.loader, name, ns[name].__get__(self.loader))
        self.config = SimpleNamespace(model="primary", revision=None, quantization=None)

    def new_source(self):
        source = Source(
            getattr(self, "fail_close", False), getattr(self, "fail_read", False)
        )
        self.sources.append(source)
        return source

    def run_consumer(self, consumer, **attrs):
        model = SimpleNamespace(load_weights=consumer, modules=lambda: (), **attrs)
        return self.loader.load_weights(model, self.config)

    def test_retained_consumer_traceback(self):
        primary = ValueError("parameter error")

        def consumer(weights):
            next(weights)
            raise primary

        try:
            self.run_consumer(consumer)
        except ValueError as error:
            retained = error.__traceback__
            self.assertIs(error, primary)
        self.assertIsNotNone(retained)
        self.assertEqual([s.closed for s in self.sources], [1])

    def test_consumer_error_preserved_when_close_fails(self):
        self.fail_close = True
        primary = ValueError("parameter error")

        def consumer(weights):
            next(weights)
            raise primary

        with self.assertRaises(ValueError) as result:
            self.run_consumer(consumer)
        self.assertIs(result.exception, primary)
        self.assertIn("close error", str(getattr(primary, "__notes__", [])))
        self.assertEqual(self.sources[0].closed, 1)

    def test_early_return_closes_source(self):
        self.run_consumer(lambda weights: next(weights))
        self.assertEqual(self.sources[0].closed, 1)

    def test_prefix_close_propagates(self):
        source = SimpleNamespace(
            model_or_path="p",
            revision=None,
            subfolder=None,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
            prefix="prefix.",
        )
        iterator = self.loader._get_weights_iterator(source)
        self.assertEqual(next(iterator)[0], "prefix.weight1")
        iterator.close()
        self.assertEqual(self.sources[0].closed, 1)

    def test_normal_primary_and_secondary(self):
        second = SimpleNamespace(
            model_or_path="secondary",
            revision=None,
            subfolder=None,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
            prefix="secondary.",
        )
        loaded: list[str] = []
        self.run_consumer(
            lambda weights: loaded.extend(name for name, _ in weights),
            secondary_weights=[second],
        )
        self.assertEqual(
            loaded, ["weight1", "weight2", "secondary.weight1", "secondary.weight2"]
        )
        self.assertEqual([s.closed for s in self.sources], [1, 1])

    def test_secondary_failure_closes_current_source(self):
        second = SimpleNamespace(
            model_or_path="secondary",
            revision=None,
            subfolder=None,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
            prefix="secondary.",
        )

        def consumer(weights):
            next(weights)
            next(weights)
            next(weights)
            raise ValueError("secondary parameter")

        with self.assertRaisesRegex(ValueError, "secondary parameter"):
            self.run_consumer(consumer, secondary_weights=[second])
        self.assertEqual([s.closed for s in self.sources], [1, 1])

    def test_close_error_not_suppressed_after_success(self):
        self.fail_close = True
        with self.assertRaisesRegex(RuntimeError, "close error"):
            self.run_consumer(lambda weights: next(weights))

    def test_read_error_preserved(self):
        self.fail_close = self.fail_read = True
        with self.assertRaisesRegex(ValueError, "read error"):
            self.run_consumer(lambda weights: next(weights))
        self.assertEqual(self.sources[0].closed, 1)


class ReaderCleanup(unittest.TestCase):
    @unittest.skipUnless(
        dependency is not None or __name__ == "__main__",
        "Set INSTANTTENSOR_SOURCE_DIR to the patched source",
    )
    def test_exit_failure_matrix(self):
        # Execute the patched dependency method without initializing CUDA.
        assert dependency is not None
        file = dependency / "instanttensor/_impl.py"
        for body_failure in [False, True]:
            for sync_failure in [False, True]:
                for close_failure in [False, True]:
                    with self.subTest(
                        body=body_failure, sync=sync_failure, close=close_failure
                    ):
                        calls: list[str] = []
                        body = ValueError("body") if body_failure else None
                        sync = RuntimeError("sync") if sync_failure else None
                        close = RuntimeError("native close") if close_failure else None

                        def synchronize(calls=calls, sync=sync):
                            calls.append("sync")
                            if sync:
                                raise sync

                        def native_close(handle, calls=calls, close=close):
                            self.assertEqual(handle, 17)
                            calls.append("close")
                            if close:
                                raise close

                        stream = SimpleNamespace(synchronize=synchronize)
                        ns = {
                            "torch": SimpleNamespace(
                                cuda=SimpleNamespace(
                                    current_stream=lambda stream=stream: stream
                                )
                            ),
                            "_C": SimpleNamespace(close=native_close),
                            "time": time,
                            "env_debug": lambda: False,
                        }
                        fn = extract(file, "__exit__", ns, "safe_open")
                        reader = SimpleNamespace(
                            loader_handle=17,
                            _invalidated=False,
                            init_time=0.0,
                            sync_time=1.0,
                            meta_read_time=2.0,
                            open_time=3.0,
                            enter_time=4.0,
                        )
                        expected = body or sync or close
                        if expected is not None and expected is not body:
                            with self.assertRaises(BaseException) as result:
                                fn(reader, type(body) if body else None, body, None)
                            self.assertIs(result.exception, expected)
                        else:
                            fn(reader, type(body) if body else None, body, None)
                        self.assertEqual(calls, ["sync", "close"])
                        self.assertTrue(reader._invalidated)
                        self.assertIsNone(reader.loader_handle)
                        fn(reader, None, None, None)
                        self.assertEqual(calls, ["sync", "close"])
                        if body and sync:
                            self.assertIn("sync", str(getattr(body, "__notes__", [])))
                        if expected and close and expected is not close:
                            self.assertIn(
                                "native close", str(getattr(expected, "__notes__", []))
                            )


@unittest.skipUnless(
    dependency is not None or __name__ == "__main__", "Set INSTANTTENSOR_SOURCE_DIR"
)
class NativeStagingAdmission(unittest.TestCase):
    """Execute dependency admission logic without importing Torch or CUDA."""

    class Backend(Enum):
        URING = 1
        AIO = 2
        URING_BUFFERED = 3
        AIO_BUFFERED = 4
        CUFILE = 5
        MMAP = 6

    class Flag:
        def __init__(self, value):
            self.value = value

        def fill_(self, value):
            self.value = value
            return self

        def item(self):
            return self.value

    def namespace(self):
        self.native_calls: list[tuple[Any, ...]] = []

        def native_open(*args: Any) -> int:
            self.native_calls.append(args)
            return 17

        return {
            "time": time,
            "os": SimpleNamespace(
                sysconf=lambda _: 4096,
                environ={},
                stat=lambda _: SimpleNamespace(st_size=7 * 1024**3),
            ),
            "sys": sys,
            "math": math,
            "warnings": warnings,
            "Backend": self.Backend,
            "_C": SimpleNamespace(open=native_open),
            "debug_log": lambda *a: None,
            "parse_backend_candidates": lambda b: [b],
            "file_in_memory": lambda _: False,
            "select_backend": lambda candidates: candidates[0],
            "cpu_count": lambda: 64,
            "MAX_IO_DEPTH": 1024,
            "MAX_CHUNK_SIZE": 1024**3,
            "_host_available_bytes": lambda: 80 * 1024**3,
            "required_buffer_size_for_io": (
                lambda chunk, depth, world: chunk * depth * world
            ),
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(
                    get_device_properties=lambda _: SimpleNamespace(is_integrated=True)
                ),
                tensor=lambda values, **kwargs: self.Flag(values[0]),
            ),
            "dist": SimpleNamespace(
                ReduceOp=SimpleNamespace(MIN="min", MAX="max"),
                distributed=False,
            ),
        }

    def reader(self, **overrides):
        values = {
            "filename": ["target.safetensors"],
            "device": "cuda:0",
            "device_idx": 0,
            "process_group": None,
            "world_size": 2,
            "buffer_size": 4 * 1024**3,
            "chunk_size": 8 * 1024**2,
            "io_depth": 256,
            "concurrency": 0,
            "backend": self.Backend.URING,
            "tensor_offsets": [],
            "_admission_status": None,
            "startup_deadline": None,
            "allocation_admission": None,
            "allocation_budget": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def method(self, name, ns):
        assert dependency is not None
        return extract(dependency / "instanttensor/_impl.py", name, ns, "safe_open")

    def test_rejection_preserves_error_and_never_opens_native_loader(self):
        ns = self.namespace()
        error = MemoryError("staging exceeds recovered physical budget")

        def reject(**kwargs):
            raise error

        with self.assertRaises(MemoryError) as result:
            self.method("_open", ns)(self.reader(allocation_admission=reject))
        self.assertIs(result.exception, error)
        self.assertEqual(self.native_calls, [])

    def test_joint_device_and_pinned_request_includes_native_padding(self):
        ns = self.namespace()
        admitted = []
        reader = self.reader(
            allocation_admission=lambda **kwargs: admitted.append(kwargs)
        )
        self.method("_open", ns)(reader)
        self.assertEqual(
            admitted,
            [
                {
                    "device_bytes": 4 * 1024**3 + 3 * (16 + 4096 + 8192),
                    "host_bytes": 2 * 1024**3,
                }
            ],
        )
        self.assertEqual(reader.loader_handle, 17)
        self.assertEqual(len(self.native_calls), 1)

    def test_mmap_internal_registration_counts_all_file_maps(self):
        ns = self.namespace()
        ns["os"].environ["INSTANTTENSOR_USE_INTERNAL_MEMORY_REGISTER"] = "1"
        admitted = []
        reader = self.reader(
            backend=self.Backend.MMAP,
            filename=["a", "b"],
            allocation_admission=lambda **kwargs: admitted.append(kwargs),
        )
        self.method("_open", ns)(reader)
        self.assertEqual(admitted[0]["host_bytes"], 14 * 1024**3)

    def determine(self, remaining):
        ns = self.namespace()
        reader = self.reader(allocation_budget=lambda: remaining)
        config = SimpleNamespace(
            buffer_size=None,
            chunk_size=None,
            concurrency=None,
            io_depth=None,
            max_free_mem_usage=None,
            backend=self.Backend.URING,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.method("_determine_io_params", ns)(reader, config)
        return reader

    def test_joint_budget_limits_depth_not_device_ring_alone(self):
        # TP2: 16MiB device + 8MiB pinned per operation, not 16MiB alone.
        padding = 3 * (16 + 4096 + 8192)
        reader = self.determine(3 * 24 * 1024**2 + padding)
        self.assertEqual(reader.io_depth, 3)

    def test_unconfigured_guard_preserves_dependency_default(self):
        self.assertEqual(self.determine(None).io_depth, 256)

    def test_peer_rejection_prevents_this_ranks_native_open(self):
        ns = self.namespace()
        ns["dist"].all_reduce = lambda flag, **kwargs: flag.fill_(1)
        reader = self.reader(
            _admission_status=self.Flag(0),
            process_group=SimpleNamespace(),
            allocation_admission=lambda **kwargs: None,
        )
        with self.assertRaisesRegex(RuntimeError, "Another rank rejected"):
            self.method("_open", ns)(reader)
        self.assertEqual(self.native_calls, [])

    def test_native_open_receives_remaining_budget_not_a_reset_timeout(self):
        ns = self.namespace()
        ns["time"] = SimpleNamespace(perf_counter=lambda: 10, monotonic=lambda: 63)
        self.method("_open", ns)(self.reader(startup_deadline=100))
        self.assertEqual(self.native_calls[0][-1], 37)

    def test_expired_startup_budget_does_not_open_native_loader(self):
        ns = self.namespace()
        ns["time"] = SimpleNamespace(perf_counter=lambda: 10, monotonic=lambda: 101)
        with self.assertRaisesRegex(TimeoutError, "deadline expired"):
            self.method("_open", ns)(self.reader(startup_deadline=100))
        self.assertEqual(self.native_calls, [])

    def test_accepted_constructor_handle_prevents_second_open(self):
        ns = self.namespace()
        reader = self.reader(loader_handle=17)
        reader._open = lambda: self.fail("native loader reopened")
        self.assertIs(self.method("__enter__", ns)(reader), reader)


class NativeInputIdentity(unittest.TestCase):
    def test_valid_manifest_then_changed_patch_is_rejected(self):
        with TemporaryDirectory() as directory:
            runtime_root = Path(directory)
            inputs = {}
            for relative in (
                "build/build_instanttensor.py",
                "build/patches/instanttensor-0.2.1-cleanup.patch",
                "build/patches/instanttensor-0.2.1-startup-safety.patch",
                "build/patches/exllamav3-startup-safety.patch",
                "build/patches/sparkinfer-trellis-swiglu-limit.patch",
                "build/patches/sparkinfer-trellis-barrier-order.patch",
                "build/patches/flashinfer-sm120-memory-order.patch",
                "build/patches/transformers-glm5next-temporal-budget.patch",
            ):
                path = runtime_root / relative.removeprefix("build/")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative)
                inputs[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            verify = extract(
                repo / "build/verify_runtime.py",
                "verify_native_patch_inputs",
                {"Path": Path, "hashlib": hashlib},
            )
            verify({"input_sha256": inputs}, runtime_root)
            for relative in inputs:
                with self.subTest(relative=relative):
                    changed = runtime_root / relative.removeprefix("build/")
                    original = changed.read_bytes()
                    changed.write_text("wrong patched source")
                    with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                        verify({"input_sha256": inputs}, runtime_root)
                    changed.write_bytes(original)

    def test_missing_native_patch_identity_is_rejected(self):
        verify = extract(
            repo / "build/verify_runtime.py",
            "verify_native_patch_inputs",
            {"Path": Path, "hashlib": hashlib},
        )
        with self.assertRaisesRegex(RuntimeError, "lacks provenance"):
            verify({"input_sha256": {}}, Path("/unused"))


if __name__ == "__main__":
    repo = Path(sys.argv.pop(1))
    dependency = Path(sys.argv.pop(1))
    unittest.main(verbosity=2)
