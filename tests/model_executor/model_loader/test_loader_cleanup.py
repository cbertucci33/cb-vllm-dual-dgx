# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import __future__

import ast
import os
import sys
import time
import unittest
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
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
        model = SimpleNamespace(load_weights=consumer, **attrs)
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


if __name__ == "__main__":
    repo = Path(sys.argv.pop(1))
    dependency = Path(sys.argv.pop(1))
    unittest.main(verbosity=2)
