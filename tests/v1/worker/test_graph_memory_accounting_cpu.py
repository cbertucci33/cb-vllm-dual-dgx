# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute the worker's memory-accounting methods without CUDA or model imports."""

import ast
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[3]
GIB = 1 << 30


def worker_methods():
    path = ROOT / "vllm/v1/worker/gpu_worker.py"
    tree = ast.parse(path.read_text())
    worker = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Worker"
    )
    names = {"_determine_available_memory", "compile_or_warm_up_model"}
    methods = [
        n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert len(methods) == len(names)
    for method in methods:
        method.decorator_list = []
    namespace = {"CompilationTimes": lambda **kwargs: kwargs}
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace


class GraphMemoryAccountingTests(unittest.TestCase):
    def test_profile_and_final_recommendation_count_each_allocation_once(self):
        for estimate_enabled in (False, True):
            for estimated_gib, actual_gib in ((2, 2), (2, 3), (0, 2), (2, 0)):
                with self.subTest(
                    enabled=estimate_enabled, estimate=estimated_gib, actual=actual_gib
                ):
                    ns = worker_methods()
                    noop = lambda *args, **kwargs: None
                    profile = types.SimpleNamespace(
                        total_consumed=80 * GIB,
                        transient_peak_headroom=4 * GIB,
                        non_kv_cache_memory=84 * GIB,
                        after_profile=types.SimpleNamespace(free_memory=40 * GIB),
                    )
                    ns.update(
                        maybe_apply_startup_plan=noop,
                        check_startup_memory=noop,
                        memory_profiling=lambda *args, _profile=profile, **kwargs: (
                            nullcontext(_profile)
                        ),
                        maybe_rocm_profiling_fallback=lambda _: None,
                        reserve_mm_ipc_gpu_memory=lambda value, *args: value,
                        envs=types.SimpleNamespace(
                            VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=estimate_enabled
                        ),
                        current_platform=types.SimpleNamespace(
                            is_cuda_alike=lambda: True, is_xpu=lambda: False
                        ),
                        CUDAGraphMode=types.SimpleNamespace(NONE=0),
                        CompilationMode=types.SimpleNamespace(NONE=0, VLLM_COMPILE=1),
                        format_gib=lambda value: value / GIB,
                        GiB_bytes=GIB,
                        logger=MagicMock(),
                        kernel_warmup=noop,
                        set_random_seed=noop,
                        freeze_gc_heap=noop,
                        maybe_attach_gc_debug_callback=noop,
                        enable_gpu_sync_check=noop,
                        clear_startup_memory_guard=noop,
                        maybe_save_startup_plan=MagicMock(),
                    )
                    state = MagicMock()
                    state.cache_config.kv_cache_memory_bytes = None
                    state.cache_config.gpu_memory_utilization = 0.9
                    state.init_snapshot.free_memory = 120 * GIB
                    state.init_snapshot.total_memory = 128 * GIB
                    state.requested_memory = 110 * GIB
                    state.model_runner.model_memory_usage = 75 * GIB
                    state.model_runner.profile_cudagraph_memory.return_value = (
                        estimated_gib * GIB
                    )
                    state._scoped_allocator_max_split.return_value = nullcontext()
                    state.vllm_config.compilation_config.cudagraph_mode = 1
                    state.vllm_config.compilation_config.mode = 0
                    state.vllm_config.kernel_config.enable_jit_warmup = False
                    state.compilation_config.mode = 0
                    state.compilation_config.compilation_time = 0
                    state.compilation_config.encoder_compilation_time = 0
                    state.use_v2_model_runner = True
                    state.model_config.enforce_eager = actual_gib == 0
                    state._get_cudagraph_capture_context.return_value = nullcontext()
                    state.model_runner.capture_model.return_value = actual_gib * GIB

                    allocation = ns["_determine_available_memory"](state)
                    applied_estimate = estimated_gib * GIB if estimate_enabled else 0
                    self.assertEqual(allocation, 26 * GIB - applied_estimate)
                    ns["compile_or_warm_up_model"](state)
                    expected = (26 - actual_gib) * GIB - 150 * (1 << 20)
                    ns["maybe_save_startup_plan"].assert_called_once_with(
                        state, expected
                    )
                    self.assertEqual(state.peak_activation_memory, 4 * GIB)


if __name__ == "__main__":
    unittest.main(verbosity=2)
