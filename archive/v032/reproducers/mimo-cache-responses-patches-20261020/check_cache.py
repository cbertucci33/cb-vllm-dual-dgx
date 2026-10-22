from pathlib import Path
import json
import vllm.platforms
from vllm.platforms.cpu import CpuPlatform
vllm.platforms._current_platform = CpuPlatform()
import pytest, torch
rc = pytest.main([
    "/source/tests/v1/core/test_prefix_caching.py::test_dflash_replay_boundary_is_reachable_through_local_prefix_cache",
    "/source/tests/v1/core/test_prefix_caching.py::test_hybrid_local_kv_retention_latest_only_reuses_replay_boundary",
    "/source/tests/v1/core/test_prefix_caching.py::test_hybrid_local_kv_retention_interval_survives_recycling",
    "-q", "--noconftest", "-p", "no:cacheprovider",
])
print(json.dumps({"exit_code": int(rc), "cuda_initialized": torch.cuda.is_initialized(), "memory_peak_bytes": int(Path("/sys/fs/cgroup/memory.peak").read_text())}))
assert not torch.cuda.is_initialized()
raise SystemExit(rc)
