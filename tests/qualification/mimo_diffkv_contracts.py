#!/usr/bin/env python3
"""Run exact MiMo FP8 DiffKV contracts without the pytest fixture stack."""

from __future__ import annotations

from pytest import MonkeyPatch

from tests.kernels.attention.test_triton_unified_attention_diffkv import (
    test_mimo_diffkv_mixed_metadata_partition_contract,
    test_mimo_fp8_diffkv_cache_read_and_partition_contract,
    test_mimo_fp8_diffkv_cache_write_contract,
    test_mimo_fp8_diffkv_cuda_graph_replay,
    test_mimo_fp8_diffkv_route_parity,
    test_mimo_static_fp8_query_quantization,
    test_triton_unified_attn_diffkv_mimo_fp8_whole_verify_3d,
)
from vllm.config import VllmConfig, set_current_vllm_config

ROUTE_CASES = [
    ([5], [18], (-1, -1), 0, False, False),
    ([63], [294], (-1, -1), 0, False, False),
    ([1], [1], (-1, -1), 0, False, False),
    ([1], [2011], (-1, -1), 0, False, False),
    ([1], [33], (-1, -1), 64, False, True),
    ([1], [8193], (-1, -1), 64, False, True),
    ([4], [8195], (127, 0), 64, True, False),
]

WHOLE_VERIFY_CASES = [
    ([2], [8193], 2),
    ([4], [8195], 4),
    ([4], [62287], 4),
    ([2, 4], [8193, 777], 4),
    ([4], [799999], 4),
]


def _with_monkeypatch(function, *args) -> None:
    monkeypatch = MonkeyPatch()
    try:
        function(*args, monkeypatch)
    finally:
        monkeypatch.undo()


def main() -> None:
    config = VllmConfig()
    with set_current_vllm_config(config):
        test_mimo_static_fp8_query_quantization(config)
    print("PASS fp8-static-query")

    test_mimo_fp8_diffkv_cache_write_contract()
    print("PASS fp8-asymmetric-cache-write")

    _with_monkeypatch(test_mimo_fp8_diffkv_cache_read_and_partition_contract)
    print("PASS fp8-packed-cache-read-partition-consumption")

    _with_monkeypatch(test_mimo_diffkv_mixed_metadata_partition_contract)
    print("PASS mixed-decode-prefill-metadata-partition")

    test_mimo_fp8_diffkv_cuda_graph_replay()
    print("PASS fp8-3d-cuda-graph-capture-replay")

    for case in ROUTE_CASES:
        _with_monkeypatch(test_mimo_fp8_diffkv_route_parity, *case)
        print(f"PASS route {case}")

    for case in WHOLE_VERIFY_CASES:
        _with_monkeypatch(
            test_triton_unified_attn_diffkv_mimo_fp8_whole_verify_3d, *case
        )
        print(f"PASS whole-verify {case}")


if __name__ == "__main__":
    main()
