# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions without importing vLLM models, Torch or native libraries."""

import ast
import dataclasses
import gc
import importlib.util
import sys
import types
import unittest
import weakref
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[2]


def load_policy():
    name = "_vllm_startup_memory_cpu_policy"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "vllm/utils/startup_memory.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


POLICY = load_policy()


def extracted_scope_functions():
    path = ROOT / "vllm/model_executor/layers/quantization/exl3.py"
    tree = ast.parse(path.read_text())
    names = {"_RuntimeScopeOwner", "_release_runtime_scope", "_runtime_scope_id"}
    nodes: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    namespace: dict[str, Any] = {
        "Any": object,
        "weakref": weakref,
        "suppress": __import__("contextlib").suppress,
        "_NEXT_RUNTIME_SCOPE_ID": 0,
        "_RANK_SLICED_RUNTIMES": {},
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class StartupAdmissionTests(unittest.TestCase):
    def test_shared_host_and_device_are_admitted_together(self):
        budget = POLICY.StartupMemoryBudget(1000, 850, True, 1000)
        with self.assertRaises(MemoryError):
            budget.check(
                device_bytes=400,
                host_bytes=300,
                available=800,
                host_available=800,
                label="combined native staging",
            )
        budget.check(
            device_bytes=400,
            host_bytes=200,
            available=800,
            host_available=800,
            label="combined native staging",
        )
        self.assertEqual(budget.reserve_bytes, 150)

    def test_guard_rejects_before_device_allocation_even_after_weights_fit(self):
        budget = POLICY.StartupMemoryBudget(1000, 850, True, 1000)
        with self.assertRaises(MemoryError):
            budget.check(
                device_bytes=200,
                host_bytes=0,
                available=300,
                host_available=300,
                label="arena after weights",
            )

    def test_released_native_temporary_is_reserved_in_inference_peak(self):
        budget = POLICY.StartupMemoryBudget(1000, 850, True, 1000)
        budget.check(
            device_bytes=200,
            host_bytes=100,
            available=600,
            host_available=600,
            label="temporary native staging",
        )
        # The declared native temporary has disappeared at the final snapshot.
        self.assertEqual(budget.transient_headroom(600), 300)

    def test_persistent_native_memory_is_not_double_counted(self):
        budget = POLICY.StartupMemoryBudget(1000, 850, True, 1000)
        budget.check(
            device_bytes=200,
            host_bytes=0,
            available=600,
            host_available=600,
            label="persistent arena",
        )
        self.assertEqual(budget.transient_headroom(400), 0)

    def test_explicit_kv_overrides_utilization_but_not_physical_availability(self):
        budget = POLICY.StartupMemoryBudget(1000, 850, True, 1000)
        budget.check(
            device_bytes=200,
            host_bytes=0,
            available=250,
            host_available=250,
            label="explicit KV",
            respect_reserve=False,
        )
        with self.assertRaises(MemoryError):
            budget.check(
                device_bytes=251,
                host_bytes=0,
                available=250,
                host_available=250,
                label="explicit KV",
                respect_reserve=False,
            )

    def test_discrete_device_and_host_do_not_share_a_counter(self):
        budget = POLICY.StartupMemoryBudget(1000, 850, False, 600)
        budget.check(
            device_bytes=200,
            host_bytes=300,
            available=400,
            host_available=400,
            label="discrete staging",
        )
        with self.assertRaises(MemoryError):
            budget.check(
                device_bytes=200,
                host_bytes=500,
                available=400,
                host_available=400,
                label="discrete staging",
            )

    def test_negative_or_unavailable_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            POLICY.StartupMemoryBudget(1000, 1001, True, 1000)
        budget = POLICY.StartupMemoryBudget(1000, 850, True, 1000)
        with self.assertRaises(ValueError):
            budget.check(
                device_bytes=-1,
                host_bytes=0,
                available=900,
                host_available=900,
                label="invalid allocation",
            )


class StartupIdentityTests(unittest.TestCase):
    def test_utilization_and_multimodal_demand_change_memory_identity(self):
        @dataclasses.dataclass
        class Config:
            gpu_memory_utilization: float
            limit_per_prompt: dict
            skip_mm_profiling: bool = False
            num_gpu_blocks: int = 0

        first = Config(0.85, {"video": {"count": 1, "num_frames": 32}})
        original = POLICY.memory_config_identity(first)
        first.gpu_memory_utilization = 0.90
        self.assertNotEqual(original, POLICY.memory_config_identity(first))
        first.gpu_memory_utilization = 0.85
        first.limit_per_prompt["video"]["num_frames"] = 64
        self.assertNotEqual(original, POLICY.memory_config_identity(first))
        first.limit_per_prompt["video"]["num_frames"] = 32
        first.skip_mm_profiling = True
        self.assertNotEqual(original, POLICY.memory_config_identity(first))

    def test_derived_block_counts_do_not_change_requested_identity(self):
        @dataclasses.dataclass
        class Config:
            gpu_memory_utilization: float = 0.85
            num_gpu_blocks: int = 0

        config = Config()
        before = POLICY.memory_config_identity(config)
        config.num_gpu_blocks = 100
        self.assertEqual(before, POLICY.memory_config_identity(config))

    def test_private_dynamic_memory_demand_is_not_omitted(self):
        @dataclasses.dataclass
        class Config:
            gpu_memory_utilization: float = 0.85

        config = Config()
        cast(Any, config)._api_process_count = 1
        original = POLICY.memory_config_identity(config)
        cast(Any, config)._api_process_count = 2
        self.assertNotEqual(original, POLICY.memory_config_identity(config))

    def test_mapping_key_type_cannot_alias_memory_configuration(self):
        self.assertNotEqual(
            POLICY.memory_config_identity({1: 800}),
            POLICY.memory_config_identity({"1": 800}),
        )

    def test_opaque_identity_is_rejected_not_repr_hashed(self):
        with self.assertRaises(TypeError):
            POLICY.memory_config_identity(object())


class RuntimeLifetimeTests(unittest.TestCase):
    def test_owner_collection_evicts_only_its_shared_arenas(self):
        namespace = extracted_scope_functions()
        target = types.SimpleNamespace()
        draft = types.SimpleNamespace()
        scope = namespace["_runtime_scope_id"](target)
        other_scope = namespace["_runtime_scope_id"](draft)
        self.assertEqual(scope, namespace["_runtime_scope_id"](target))
        cache = namespace["_RANK_SLICED_RUNTIMES"]
        cache[((scope, False), "decode")] = object()
        cache[((scope, False), "prefill")] = object()
        cache[((other_scope, True), "prefill")] = object()
        del target
        gc.collect()
        self.assertEqual(list(cache), [((other_scope, True), "prefill")])

    def test_explicit_scope_eviction_is_idempotent(self):
        namespace = extracted_scope_functions()
        config = types.SimpleNamespace()
        scope = namespace["_runtime_scope_id"](config)
        cache = namespace["_RANK_SLICED_RUNTIMES"]
        cache[((scope, False), "prefill")] = object()
        namespace["_release_runtime_scope"](scope)
        namespace["_release_runtime_scope"](scope)
        self.assertEqual(cache, {})


class FakeDevice:
    def __init__(self, value):
        text = str(value)
        self.type = text.split(":")[0]
        self.index = int(text.split(":")[1]) if ":" in text else None

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"


def extracted_guard_functions():
    path = ROOT / "vllm/utils/mem_utils.py"
    tree = ast.parse(path.read_text())
    names = {
        "configure_startup_memory_guard",
        "restore_startup_allocator_limit",
        "clear_startup_memory_guard",
        "_startup_memory_state",
        "check_startup_memory",
        "record_startup_memory_peak",
        "startup_memory_guard_remaining_bytes",
        "startup_memory_peak_headroom",
    }
    nodes: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    cuda = types.SimpleNamespace(
        get_per_process_memory_fraction=lambda device: 1.0,
        set_per_process_memory_fraction=lambda fraction, device: None,
    )
    namespace: dict[str, Any] = {
        "torch": types.SimpleNamespace(device=FakeDevice, cuda=cuda),
        "psutil": types.SimpleNamespace(
            virtual_memory=lambda: types.SimpleNamespace(available=800)
        ),
        "current_platform": types.SimpleNamespace(
            current_device=lambda: "cuda:0",
            is_integrated_gpu=lambda index: True,
        ),
        "MemorySnapshot": object,
        "StartupMemoryBudget": POLICY.StartupMemoryBudget,
        "_STARTUP_MEMORY_BUDGETS": {},
        "_STARTUP_ALLOCATOR_FRACTIONS": {},
        "sys": sys,
        "logger": types.SimpleNamespace(warning=lambda *args: None),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class StartupCompletionTests(unittest.TestCase):
    """Execute the worker completion boundary with the actual CPU guard."""

    def run_warmup(self, capture_error=None):
        namespace = extracted_guard_functions()
        namespace["_STARTUP_MEMORY_BUDGETS"][0] = POLICY.StartupMemoryBudget(
            1000000, 850000, True, 1000000
        )
        namespace["psutil"].virtual_memory = lambda: types.SimpleNamespace(
            available=100000
        )
        path = ROOT / "vllm/v1/worker/gpu_worker.py"
        tree = ast.parse(path.read_text())
        worker = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Worker"
        )
        method = next(
            node
            for node in worker.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "compile_or_warm_up_model"
        )
        method.decorator_list = []
        noop = lambda *args, **kwargs: None
        namespace.update(
            CompilationTimes=lambda **kwargs: kwargs,
            CompilationMode=types.SimpleNamespace(NONE=0, VLLM_COMPILE=1),
            kernel_warmup=noop,
            set_random_seed=noop,
            freeze_gc_heap=noop,
            maybe_attach_gc_debug_callback=noop,
            enable_gpu_sync_check=noop,
        )
        state = MagicMock()
        state.vllm_config.compilation_config.mode = 0
        state.compilation_config.mode = 0
        state.compilation_config.compilation_time = 1
        state.compilation_config.encoder_compilation_time = 2
        state.vllm_config.kernel_config.enable_jit_warmup = False
        state.use_v2_model_runner = True
        state.model_config.enforce_eager = False
        state.cache_config.kv_cache_memory_bytes = 7300000000
        state.device = FakeDevice("cuda:0")
        state._get_cudagraph_capture_context.return_value = nullcontext()
        del state.cudagraph_memory_estimate

        def capture():
            # The reserve remains enforced through graph capture, even when
            # explicit KV allocation has already consumed it.
            with self.assertRaises(MemoryError):
                namespace["check_startup_memory"](
                    device_bytes=75264, label="startup", device=state.device
                )
            if capture_error is not None:
                raise capture_error
            return 0

        state.model_runner.capture_model.side_effect = capture
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        return namespace, state

    def test_completed_startup_does_not_reject_runtime_image_transfer(self):
        namespace, state = self.run_warmup()
        result = namespace["compile_or_warm_up_model"](state)
        self.assertEqual(result, {"language_model": 1, "encoder": 2})
        namespace["check_startup_memory"](
            device_bytes=75264, label="runtime image", device=state.device
        )
        self.assertEqual(namespace["_STARTUP_MEMORY_BUDGETS"], {})
        # Runtime host admission is still physical-capacity checked.
        with self.assertRaises(MemoryError):
            namespace["check_startup_memory"](
                host_bytes=100001, label="runtime host", device=state.device
            )

    def test_failed_warmup_does_not_clear_startup_reserve(self):
        namespace, state = self.run_warmup(RuntimeError("capture failure"))
        with self.assertRaisesRegex(RuntimeError, "capture failure"):
            namespace["compile_or_warm_up_model"](state)
        self.assertIn(0, namespace["_STARTUP_MEMORY_BUDGETS"])


class RuntimeAdapterTests(unittest.TestCase):
    def test_unconfigured_frontend_never_queries_cuda(self):
        namespace = extracted_guard_functions()

        def forbidden():
            self.fail("Frontend guard tried to create/query a CUDA context")

        namespace["current_platform"].current_device = forbidden
        self.assertIsNone(namespace["_startup_memory_state"]())
        namespace["check_startup_memory"](host_bytes=700, label="frontend")
        namespace["clear_startup_memory_guard"]()

    def test_frontend_declared_host_allocation_is_not_silently_unchecked(self):
        namespace = extracted_guard_functions()

        def forbidden():
            self.fail("Frontend host check initialized CUDA")

        namespace["current_platform"].current_device = forbidden
        with self.assertRaises(MemoryError):
            namespace["check_startup_memory"](host_bytes=801, label="frontend")
        with self.assertRaises(MemoryError):
            namespace["check_startup_memory"](
                device_bytes=600, host_bytes=201, label="CPU tensor", device="cpu"
            )
        namespace["check_startup_memory"](
            device_bytes=600, host_bytes=200, label="CPU tensor", device="cpu"
        )
        with self.assertRaises(ValueError):
            namespace["check_startup_memory"](host_bytes=-1, label="invalid frontend")

    def test_cpu_allocation_uses_worker_shared_memory_budget(self):
        namespace = extracted_guard_functions()
        namespace["_STARTUP_MEMORY_BUDGETS"][0] = POLICY.StartupMemoryBudget(
            1000, 850, True, 1000
        )
        self.assertEqual(namespace["startup_memory_guard_remaining_bytes"]("cpu"), 650)
        with self.assertRaises(MemoryError):
            namespace["check_startup_memory"](
                host_bytes=651, label="CPU media", device="cpu"
            )

    def test_cpu_budget_ownership_ambiguity_is_not_ignored(self):
        namespace = extracted_guard_functions()
        for index in (0, 1):
            namespace["_STARTUP_MEMORY_BUDGETS"][index] = POLICY.StartupMemoryBudget(
                1000, 850, True, 1000
            )
        with self.assertRaises(ValueError):
            namespace["check_startup_memory"](
                host_bytes=1, label="ambiguous CPU media", device="cpu"
            )

    def test_allocator_install_failure_rolls_back_without_installing_budget(self):
        namespace = extracted_guard_functions()
        calls: list[float] = []

        def setter(fraction, device):
            calls.append(fraction)
            if len(calls) == 1:
                raise RuntimeError("allocator install failed")

        namespace["torch"].cuda.set_per_process_memory_fraction = setter
        snapshot = types.SimpleNamespace(
            device_=FakeDevice("cuda:0"),
            free_memory=1000,
            total_memory=1000,
            torch_memory=0,
        )
        with self.assertRaisesRegex(RuntimeError, "allocator install failed"):
            namespace["configure_startup_memory_guard"](snapshot, 850)
        self.assertEqual(calls, [0.85, 1.0])
        self.assertEqual(namespace["_STARTUP_MEMORY_BUDGETS"], {})
        self.assertEqual(namespace["_STARTUP_ALLOCATOR_FRACTIONS"], {})

    def test_allocator_restore_preserves_original_failure(self):
        namespace = extracted_guard_functions()
        namespace["_STARTUP_ALLOCATOR_FRACTIONS"][0] = 1.0

        def setter(fraction, device):
            raise RuntimeError("secondary restore failure")

        namespace["torch"].cuda.set_per_process_memory_fraction = setter
        root = MemoryError("original native allocation failure")
        try:
            raise root
        except MemoryError:
            namespace["restore_startup_allocator_limit"]("cuda:0")
        self.assertIn(
            "secondary restore failure",
            cast(list[str], getattr(root, "__notes__", []))[0],
        )

    def test_clear_removes_stopped_worker_policy(self):
        namespace = extracted_guard_functions()
        namespace["_STARTUP_MEMORY_BUDGETS"][0] = POLICY.StartupMemoryBudget(
            1000, 850, True, 1000
        )
        namespace["_STARTUP_ALLOCATOR_FRACTIONS"][0] = 1.0
        namespace["clear_startup_memory_guard"]("cuda:0")
        self.assertEqual(namespace["_STARTUP_MEMORY_BUDGETS"], {})
        self.assertEqual(namespace["_STARTUP_ALLOCATOR_FRACTIONS"], {})


class PartialFailureTests(unittest.TestCase):
    def test_partial_scope_setter_rolls_back_both_attributes(self):
        namespace = extracted_scope_functions()

        class Restricted:
            def __init__(self, denied):
                self.denied = denied

            def __setattr__(self, name, value):
                object.__setattr__(self, name, value)
                if name == getattr(self, "denied", None):
                    raise AttributeError("partial setter failed after assigning")

        for name in ("_exl3_runtime_owner", "_exl3_runtime_scope_id"):
            owner = Restricted(name)
            with self.assertRaises(TypeError):
                namespace["_runtime_scope_id"](owner)
            self.assertFalse(hasattr(owner, "_exl3_runtime_owner"))
            self.assertFalse(hasattr(owner, "_exl3_runtime_scope_id"))

    def test_existing_scope_without_owner_is_rejected(self):
        namespace = extracted_scope_functions()
        with self.assertRaisesRegex(RuntimeError, "no live resource owner"):
            namespace["_runtime_scope_id"](
                types.SimpleNamespace(_exl3_runtime_scope_id=42)
            )

    def test_failed_restore_retains_previous_policy_for_later_cleanup(self):
        namespace = extracted_guard_functions()
        namespace["_STARTUP_ALLOCATOR_FRACTIONS"][0] = 0.97
        namespace["_STARTUP_MEMORY_BUDGETS"][0] = POLICY.StartupMemoryBudget(
            1000, 850, True, 1000
        )
        calls: list[float] = []

        def setter(fraction, device):
            calls.append(fraction)
            if len(calls) == 1:
                raise RuntimeError("restore failed")

        namespace["torch"].cuda.set_per_process_memory_fraction = setter
        with self.assertRaises(RuntimeError):
            namespace["clear_startup_memory_guard"]("cuda:0")
        self.assertEqual(namespace["_STARTUP_MEMORY_BUDGETS"], {})
        self.assertEqual(namespace["_STARTUP_ALLOCATOR_FRACTIONS"], {0: 0.97})
        namespace["clear_startup_memory_guard"]("cuda:0")
        self.assertEqual(calls, [0.97, 0.97])
        self.assertEqual(namespace["_STARTUP_ALLOCATOR_FRACTIONS"], {})

    def test_failed_install_and_rollback_retain_previous_allocator_policy(self):
        namespace = extracted_guard_functions()

        def setter(fraction, device):
            raise RuntimeError("allocator API failed")

        namespace["torch"].cuda.set_per_process_memory_fraction = setter
        snapshot = types.SimpleNamespace(
            device_=FakeDevice("cuda:0"),
            free_memory=1000,
            total_memory=1000,
            torch_memory=0,
        )
        with self.assertRaises(RuntimeError) as failure:
            namespace["configure_startup_memory_guard"](snapshot, 850)
        self.assertEqual(namespace["_STARTUP_ALLOCATOR_FRACTIONS"], {0: 1.0})
        self.assertIn(
            "allocator API failed",
            cast(list[str], getattr(failure.exception, "__notes__", []))[0],
        )


class MemoryPlanFingerprintTests(unittest.TestCase):
    def test_actual_fingerprint_rejects_compile_hash_aliasing(self):
        import hashlib
        import json
        import os
        from importlib import metadata

        path = ROOT / "vllm/v1/worker/startup_plan.py"
        tree = ast.parse(path.read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "compute_plan_fingerprint"
        )
        namespace: dict[str, Any] = {
            "__file__": str(path),
            "VllmConfig": object,
            "PLAN_SCHEMA_VERSION": 2,
            "memory_config_identity": POLICY.memory_config_identity,
            "hashlib": hashlib,
            "json": json,
            "os": os,
            "Path": Path,
            "metadata": metadata,
            "torch": types.SimpleNamespace(
                __version__="CPU-contract", version=types.SimpleNamespace(cuda="")
            ),
            "current_platform": types.SimpleNamespace(
                get_device_capability=lambda: None,
                get_device_name=lambda: "CPU-contract",
                get_device_total_memory=lambda: 1000,
            ),
        }
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
            namespace,
        )

        @dataclasses.dataclass
        class Cache:
            gpu_memory_utilization: float = 0.85

        @dataclasses.dataclass
        class Model:
            model: str = "/nonexistent-cpu-contract-model"
            multimodal_config: dict = dataclasses.field(
                default_factory=lambda: {"video": {"num_frames": 32}}
            )

        config = types.SimpleNamespace(
            model_config=Model(),
            cache_config=Cache(),
            scheduler_config=None,
            parallel_config=None,
            load_config=None,
            speculative_config=None,
            compilation_config=None,
            compute_hash=lambda: "unchanged-compilation-key",
        )
        fake_package = types.ModuleType("vllm")
        cast(Any, fake_package).__version__ = "CPU-contract"
        old_package = sys.modules.get("vllm")
        sys.modules["vllm"] = fake_package
        try:
            fingerprint = namespace["compute_plan_fingerprint"]
            original = fingerprint(config, 0, 2)
            config.cache_config.gpu_memory_utilization = 0.90
            self.assertNotEqual(original, fingerprint(config, 0, 2))
            config.cache_config.gpu_memory_utilization = 0.85
            config.model_config.multimodal_config["video"]["num_frames"] = 64
            self.assertNotEqual(original, fingerprint(config, 0, 2))
        finally:
            if old_package is None:
                del sys.modules["vllm"]
            else:
                sys.modules["vllm"] = old_package


class FailureContainmentTests(unittest.TestCase):
    def test_zero_allocation_phase_cannot_enter_below_existing_reserve(self):
        budget = POLICY.StartupMemoryBudget(1000, 850, True, 1000)
        with self.assertRaises(MemoryError):
            budget.check(
                device_bytes=0,
                host_bytes=0,
                available=149,
                host_available=149,
                label="enter profile",
            )

    def test_profile_cleanup_runs_and_preserves_original_forward_exception(self):
        import contextlib
        from collections.abc import Generator

        path = ROOT / "vllm/utils/mem_utils.py"
        tree = ast.parse(path.read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "memory_profiling"
        )
        calls: list[str] = []
        warnings = []

        class Result:
            def __init__(self, **kwargs):
                self.before_profile = types.SimpleNamespace(measure=lambda: None)
                self.after_profile = types.SimpleNamespace(measure=self.fail)

            def fail(self):
                raise RuntimeError("secondary measurement failure")

        namespace: dict[str, Any] = {
            "contextlib": contextlib,
            "Generator": Generator,
            "MemorySnapshot": object,
            "MemoryProfilingResult": Result,
            "gc": types.SimpleNamespace(collect=lambda: calls.append("collect")),
            "torch": types.SimpleNamespace(
                accelerator=types.SimpleNamespace(
                    empty_cache=lambda: calls.append("empty"),
                    reset_peak_memory_stats=lambda device: None,
                )
            ),
            "_startup_memory_state": lambda device: None,
            "logger": types.SimpleNamespace(
                warning=lambda *args: warnings.append(args)
            ),
        }
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        with (
            self.assertRaisesRegex(MemoryError, "forward failure"),
            namespace["memory_profiling"](
                types.SimpleNamespace(device_="CPU-contract")
            ),
        ):
            raise MemoryError("forward failure")
        self.assertEqual(calls, ["collect", "empty", "collect", "empty"])
        self.assertEqual(len(warnings), 1)

    def test_model_unload_releases_one_shared_owner_once(self):
        spec = importlib.util.spec_from_file_location(
            "_model_lifecycle_cpu", ROOT / "vllm/utils/model_lifecycle.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        calls: list[str] = []
        owner = object()
        method = types.SimpleNamespace(
            quant_config=owner,
            release_runtime_resources=lambda: calls.append("release"),
        )
        model = types.SimpleNamespace(
            modules=lambda: iter(
                [
                    types.SimpleNamespace(quant_method=method),
                    types.SimpleNamespace(quant_method=method),
                ]
            )
        )
        module.release_model_runtime_resources(model)
        self.assertEqual(calls, ["release"])

    def test_unsupported_plan_identity_never_reads_or_applies_stale_cache(self):
        path = ROOT / "vllm/v1/worker/startup_plan.py"
        tree = ast.parse(path.read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "maybe_apply_startup_plan"
        )
        calls: list[str] = []

        def unsupported(*args):
            raise TypeError("opaque runtime identity")

        namespace: dict[str, Any] = {
            "envs": types.SimpleNamespace(VLLM_ENABLE_STARTUP_PLAN=True),
            "compute_plan_fingerprint": unsupported,
            "_load_plan": lambda *args: calls.append("loaded stale plan"),
            "logger": types.SimpleNamespace(
                warning=lambda *args: calls.append("warning")
            ),
        }
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        worker = types.SimpleNamespace(
            cache_config=types.SimpleNamespace(kv_cache_memory_bytes=None),
            vllm_config=object(),
            rank=0,
            parallel_config=types.SimpleNamespace(world_size=2),
        )
        namespace["maybe_apply_startup_plan"](worker)
        self.assertEqual(calls, ["warning"])
        self.assertIsNone(worker.cache_config.kv_cache_memory_bytes)


class ShutdownOrderingTests(unittest.TestCase):
    def test_faulted_optional_connector_cannot_prevent_arena_owner_eviction(self):
        path = ROOT / "vllm/v1/worker/gpu/model_runner.py"
        tree = ast.parse(path.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner"
        )
        node = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "shutdown"
        )
        for failure in (False, True):
            calls: list[str] = []

            def release(model, calls=calls):
                calls.append(model)

            def close():
                raise RuntimeError("connector cleanup failed")

            namespace: dict[str, Any] = {
                "torch": types.SimpleNamespace(
                    accelerator=types.SimpleNamespace(
                        synchronize=lambda calls=calls: calls.append("synchronize")
                    )
                ),
                "release_model_runtime_resources": release,
                "set_offloader": lambda value: None,
                "free_before_shutdown": lambda config: None,
                "clear_layer_kv_caches": lambda modules: None,
                "gc": types.SimpleNamespace(collect=lambda: None),
            }
            namespace["torch"].accelerator.empty_cache = lambda: None
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
                namespace,
            )
            runner = types.SimpleNamespace(
                _shutdown_due_to_failure=failure,
                model="target",
                speculator=types.SimpleNamespace(model="draft"),
                aux_output_connector=types.SimpleNamespace(close=close),
            )
            with self.assertRaisesRegex(RuntimeError, "connector cleanup failed"):
                namespace["shutdown"](runner)
            expected = ["target", "draft"]
            if not failure:
                expected.insert(0, "synchronize")
            self.assertEqual(calls, expected)

    def test_multiple_cleanup_failures_do_not_skip_later_owner_release(self):
        path = ROOT / "vllm/v1/worker/gpu/model_runner.py"
        tree = ast.parse(path.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner"
        )
        node = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "shutdown"
        )
        calls: list[str] = []

        def close():
            raise RuntimeError("first connector failure")

        def offloader(value):
            raise RuntimeError("second offloader failure")

        model = types.SimpleNamespace(modules=lambda: iter(()))
        runner = types.SimpleNamespace(
            _shutdown_due_to_failure=True,
            model=model,
            speculator=types.SimpleNamespace(model=model),
            aux_output_connector=types.SimpleNamespace(close=close),
            kv_caches=["backing"],
            attn_groups=["groups"],
            kv_cache_config=object(),
            execute_model_state=object(),
            model_state=types.SimpleNamespace(supports_mm_inputs=True),
            encoder_cache=types.SimpleNamespace(
                reset_encoder_cache=lambda: calls.append("encoder cache")
            ),
            vllm_config=object(),
        )
        namespace: dict[str, Any] = {
            "torch": types.SimpleNamespace(
                accelerator=types.SimpleNamespace(
                    synchronize=lambda: self.fail("fault cleanup synchronized"),
                    empty_cache=lambda: calls.append("allocator cache"),
                )
            ),
            "release_model_runtime_resources": lambda owner: calls.append("owners"),
            "set_offloader": offloader,
            "free_before_shutdown": lambda config: calls.append("offloader buffers"),
            "clear_layer_kv_caches": lambda modules: calls.append("layer caches"),
            "gc": types.SimpleNamespace(collect=lambda: calls.append("collect")),
            "logger": types.SimpleNamespace(debug=lambda *args: None),
        }
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        with self.assertRaisesRegex(RuntimeError, "first connector failure") as raised:
            namespace["shutdown"](runner)
        self.assertFalse(hasattr(runner, "model"))
        self.assertFalse(hasattr(runner, "model_state"))
        self.assertFalse(hasattr(runner, "kv_cache_config"))
        self.assertEqual(runner.kv_caches, [])
        self.assertEqual(runner.attn_groups, [])
        self.assertIsNone(runner.speculator)
        self.assertIsNone(runner.encoder_cache)
        self.assertIsNone(runner.execute_model_state)
        self.assertEqual(calls[-2:], ["collect", "allocator cache"])
        self.assertTrue(
            any(
                "second offloader failure" in note
                for note in cast(list[str], getattr(raised.exception, "__notes__", []))
            )
        )

    def test_restore_in_propagating_finally_preserves_root_exception(self):
        namespace = extracted_guard_functions()
        namespace["_STARTUP_ALLOCATOR_FRACTIONS"][0] = 1.0

        def fail_restore(fraction, device):
            raise RuntimeError("secondary restore failure")

        namespace["torch"].cuda.set_per_process_memory_fraction = fail_restore

        def wrapper():
            try:
                raise MemoryError("primary profiling failure")
            finally:
                namespace["restore_startup_allocator_limit"]("cuda:0")

        with self.assertRaisesRegex(MemoryError, "primary profiling failure"):
            wrapper()
        self.assertEqual(namespace["_STARTUP_ALLOCATOR_FRACTIONS"], {0: 1.0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
