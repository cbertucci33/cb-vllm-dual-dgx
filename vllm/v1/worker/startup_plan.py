# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persist and reuse the memory-profiling result across engine boots.

On startup, vLLM measures how much GPU memory the KV cache can use and
computes the ``--kv-cache-memory`` value that reproduces that allocation.
For a fixed (model, config, hardware, library) combination the result is
deterministic, yet it is re-measured on every boot.

When ``VLLM_ENABLE_STARTUP_PLAN=1``, each worker persists that value under
``{VLLM_CACHE_ROOT}/startup_plan/`` (regenerable derived state, alongside
the torch.compile cache), keyed by a fingerprint of everything the value
depends on, and later boots apply it automatically -- skipping the
automatic KV sizing and the CUDA-graph memory estimation pass --
if and only if the memory-contract fingerprint matches and the device has
at least as much free memory as when the plan was recorded. The full-shape
forward and prospective allocation admission still run; cached KV sizing
does not bypass startup safety. On an identity mismatch the worker profiles
normally.
"""

import hashlib
import json
import os
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.startup_memory import memory_config_identity

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

PLAN_SCHEMA_VERSION = 2


def compute_plan_fingerprint(
    vllm_config: VllmConfig, rank: int, world_size: int
) -> str:
    """Hash everything the profiled KV-cache memory value depends on.

    ``VllmConfig.compute_hash()`` covers the vLLM version and the model,
    cache, parallel, and compilation configs, but deliberately contains no
    device identity (``DeviceConfig.compute_hash`` is empty), so device
    name, total memory, compute capability, and the torch/CUDA build are
    added here. The vLLM version is also pinned as an explicit factor so
    version invalidation holds no matter how ``compute_hash`` evolves.
    Rank is included because per-rank memory use differs under TP/PP.
    Complete memory configuration, runtime source, installed native inputs,
    driver identity and memory-affecting environment extend the compile key.
    Unknown configuration encodings are rejected rather than repr-hashed.
    """
    # Imported here (as VllmConfig.compute_hash does) to avoid a cycle with
    # the top-level vllm package.
    from vllm import __version__ as vllm_version

    capability = current_platform.get_device_capability()
    memory_configs = {
        name: memory_config_identity(getattr(vllm_config, name, None))
        for name in (
            "model_config",
            "cache_config",
            "scheduler_config",
            "parallel_config",
            "load_config",
            "speculative_config",
            "compilation_config",
        )
    }
    quant_config = getattr(vllm_config, "quant_config", None)
    if quant_config is not None:
        memory_configs["quant_config"] = {
            "implementation": (
                type(quant_config).__module__ + "." + type(quant_config).__qualname__
            ),
            "state": memory_config_identity(
                {
                    name: value
                    for name, value in vars(quant_config).items()
                    if name
                    not in {
                        "_exl3_runtime_scope_id",
                        "_exl3_runtime_owner",
                    }
                }
            ),
        }
    libraries: dict[str, object] = {}
    for name in (
        "instanttensor",
        "sparkinfer",
        "exllamav3-ext-vllm",
        "transformers",
        "nvidia-cutlass-dsl",
    ):
        try:
            distribution = metadata.distribution(name)
            record = distribution.read_text("RECORD")
            files = []
            for item in sorted(distribution.files or (), key=str):
                if str(item).endswith((".py", ".so", ".a", ".json")):
                    path = Path(str(distribution.locate_file(item)))
                    if path.is_file():
                        digest = hashlib.sha256()
                        with path.open("rb") as handle:
                            for chunk in iter(lambda: handle.read(1 << 20), b""):
                                digest.update(chunk)
                        files.append((str(item), digest.hexdigest()))
            if not files and record is None:
                raise ValueError(f"Unverified installed library identity: {name}")
            libraries[name] = {
                "version": distribution.version,
                "record": hashlib.sha256((record or "").encode()).hexdigest(),
                "files": files,
            }
        except metadata.PackageNotFoundError:
            libraries[name] = None
    source_digest = hashlib.sha256()
    source_root = Path(__file__).resolve().parents[2]
    for path in sorted(source_root.rglob("*.py")):
        source_digest.update(str(path.relative_to(source_root)).encode())
        source_digest.update(path.read_bytes())
    artifacts = {}
    model_configs = [vllm_config.model_config]
    speculative = vllm_config.speculative_config
    if speculative is not None:
        model_configs.append(speculative.draft_model_config)
    for model_config in model_configs:
        if model_config is None:
            continue
        root = Path(model_config.model)
        if root.is_dir():
            artifact_files = []
            for artifact_file in sorted(root.iterdir()):
                if artifact_file.is_file() and (
                    artifact_file.name.endswith(".json")
                    or artifact_file.name.endswith(".safetensors")
                ):
                    stat = artifact_file.stat()
                    artifact_files.append(
                        (artifact_file.name, stat.st_size, stat.st_mtime_ns)
                    )
            artifacts[str(root.resolve())] = artifact_files
    driver_path = Path("/proc/driver/nvidia/version")
    driver = driver_path.read_text() if driver_path.exists() else None
    memory_environment = {
        name: value
        for name, value in sorted(os.environ.items())
        if name.startswith(
            (
                "VLLM_",
                "INSTANTTENSOR_",
                "SPARKINFER_",
                "CUDA_",
                "PYTORCH_",
                "OMP_",
                "MKL_",
            )
        )
    }
    factors = {
        "schema": PLAN_SCHEMA_VERSION,
        "vllm": vllm_version,
        "vllm_config": vllm_config.compute_hash(),
        "memory_configs": memory_configs,
        "memory_environment": memory_environment,
        "libraries": libraries,
        "runtime_python_source": source_digest.hexdigest(),
        "local_artifact_inventory": artifacts,
        "driver": driver,
        "device_name": current_platform.get_device_name(),
        "device_total_memory": current_platform.get_device_total_memory(),
        "device_capability": str(capability) if capability else "",
        "torch": torch.__version__,
        "torch_git": getattr(torch.version, "git_version", None),
        "cuda": torch.version.cuda or "",
        "rank": rank,
        "world_size": world_size,
    }
    fingerprint_digest = hashlib.sha256(
        json.dumps(factors, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return fingerprint_digest[:16]


def _plan_path(fingerprint: str) -> str:
    """Plans are regenerable derived state, so they live under the standard
    vLLM cache root (like the torch.compile cache) and relocate with
    ``VLLM_CACHE_ROOT`` instead of needing a location knob of their own."""
    # VLLM_CACHE_ROOT is already user-expanded by envs.py.
    return os.path.join(
        envs.VLLM_CACHE_ROOT, "startup_plan", f"startup_plan_{fingerprint}.json"
    )


def _load_plan(fingerprint: str) -> dict | None:
    """Load a plan for this fingerprint; None if absent or unreadable."""
    path = _plan_path(fingerprint)
    try:
        with open(path) as f:
            plan = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Ignoring unreadable startup plan %s: %s", path, e)
        return None
    if not isinstance(plan, dict):
        logger.warning("Ignoring startup plan with invalid top-level type: %s", path)
        return None
    if (
        plan.get("schema") != PLAN_SCHEMA_VERSION
        or plan.get("fingerprint") != fingerprint
    ):
        return None
    return plan


def _applicable_kv_cache_memory_bytes(
    plan: dict, current_free_memory: int
) -> int | None:
    """The apply-time OOM-safety gate.

    The recorded value is only valid if the device has at least as much
    free memory now as when the plan was measured (co-tenants, leaked
    allocations, or MIG changes all reduce it). Outside that envelope,
    return None and let the caller re-profile.
    """
    kv_bytes = plan.get("kv_cache_memory_bytes")
    baseline = plan.get("free_memory_baseline")
    if type(kv_bytes) is not int or type(baseline) is not int:
        return None
    if kv_bytes <= 0 or baseline <= 0 or kv_bytes > baseline:
        return None
    if current_free_memory < baseline:
        logger.info(
            "Startup plan not applied: current free memory (%.2f GiB) is "
            "below the recorded baseline (%.2f GiB); falling back to full "
            "memory profiling.",
            current_free_memory / (1 << 30),
            baseline / (1 << 30),
        )
        return None
    return kv_bytes


def maybe_apply_startup_plan(worker: "Worker") -> None:
    """If enabled and ``--kv-cache-memory`` was not set explicitly, apply a
    persisted plan by setting ``worker.cache_config.kv_cache_memory_bytes``.
    No-op unless ``VLLM_ENABLE_STARTUP_PLAN=1``."""
    if (
        not envs.VLLM_ENABLE_STARTUP_PLAN
        or worker.cache_config.kv_cache_memory_bytes is not None
    ):
        return
    try:
        fingerprint = compute_plan_fingerprint(
            worker.vllm_config, worker.rank, worker.parallel_config.world_size
        )
    except (TypeError, ValueError, OSError, RecursionError) as exc:
        logger.warning("Startup plan identity is unverified; not reused/saved: %s", exc)
        return
    plan = _load_plan(fingerprint)
    if plan is None:
        return
    current_free_memory = worker.init_snapshot.free_memory
    kv_bytes = _applicable_kv_cache_memory_bytes(plan, current_free_memory)
    if kv_bytes is None:
        return
    logger.info(
        "Applying persisted startup plan (fingerprint %s): "
        "kv_cache_memory_bytes=%d (%.2f GiB), recorded free-memory "
        "baseline %.2f GiB, current %.2f GiB. Memory profiling will "
        "be skipped.",
        fingerprint,
        kv_bytes,
        kv_bytes / (1 << 30),
        plan["free_memory_baseline"] / (1 << 30),
        current_free_memory / (1 << 30),
    )
    worker.cache_config.kv_cache_memory_bytes = kv_bytes


def maybe_save_startup_plan(worker: "Worker", kv_cache_memory_bytes: int) -> None:
    """Atomically persist this boot's profiling result for future boots.
    No-op unless ``VLLM_ENABLE_STARTUP_PLAN=1``; failures are logged,
    never raised."""
    if not envs.VLLM_ENABLE_STARTUP_PLAN:
        return
    try:
        fingerprint = compute_plan_fingerprint(
            worker.vllm_config, worker.rank, worker.parallel_config.world_size
        )
    except (TypeError, ValueError, OSError, RecursionError) as exc:
        logger.warning("Startup plan identity is unverified; not reused/saved: %s", exc)
        return
    path = _plan_path(fingerprint)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "schema": PLAN_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "kv_cache_memory_bytes": int(kv_cache_memory_bytes),
            "free_memory_baseline": int(worker.init_snapshot.free_memory),
        }
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
        logger.info("Saved startup plan to %s", path)
    except OSError as e:
        logger.warning("Failed to save startup plan to %s: %s", path, e)
