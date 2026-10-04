#!/usr/bin/env python3
"""Standalone MiMo DFlash-v1 K=4 runner contracts.

This file deliberately depends only on packages in the production image. It is
mounted read-only into a disposable qualification container and is never copied
into the runtime image.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    prepare_dflash_inputs,
    synthesize_draft_ring_block_tables,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

K = 4
NUM_QUERY_PER_REQ = K + 1
BLOCK_SIZE = 16


def _assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: actual={actual!r}, expected={expected!r}")


def _run_prepare(
    *,
    target_positions: list[int],
    block_table_values: list[int],
    num_sampled: int = 1,
    num_rejected: int = 2,
    cp_rank: int = 0,
    cp_size: int = 1,
    cp_interleave: int = 1,
) -> SimpleNamespace:
    device = torch.device("cuda")
    max_num_reqs = 4
    max_num_tokens = 32
    input_buffers = SimpleNamespace(
        input_ids=torch.full((max_num_tokens,), -1, dtype=torch.int32, device=device),
        positions=torch.full((max_num_tokens,), -1, dtype=torch.int64, device=device),
        query_start_loc=torch.full(
            (max_num_reqs + 1,), -1, dtype=torch.int32, device=device
        ),
        seq_lens=torch.full((max_num_reqs,), -1, dtype=torch.int32, device=device),
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_scheduled_tokens=np.array([len(target_positions)], dtype=np.int32),
        positions=torch.tensor(target_positions, dtype=torch.int64, device=device),
        query_start_loc=torch.tensor(
            [0, len(target_positions)], dtype=torch.int32, device=device
        ),
        idx_mapping=torch.tensor([2], dtype=torch.int32, device=device),
    )
    query_slot_mapping = torch.full(
        (max_num_tokens,), -2, dtype=torch.int64, device=device
    )
    context_positions = torch.full(
        (max_num_tokens,), -1, dtype=torch.int64, device=device
    )
    context_slot_mapping = torch.full(
        (max_num_tokens,), -2, dtype=torch.int64, device=device
    )
    sample_indices = torch.full(
        (max_num_reqs * K,), -1, dtype=torch.int64, device=device
    )
    sample_pos = torch.full_like(sample_indices, -1)
    sample_idx_mapping = torch.full(
        sample_indices.shape, -1, dtype=torch.int32, device=device
    )
    temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
    seeds = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
    input_temperature = torch.tensor(
        [0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=device
    )
    input_seeds = torch.tensor([0, 0, 17, 0], dtype=torch.int64, device=device)
    last_sampled = torch.tensor([0, 0, 99, 0], dtype=torch.int64, device=device)
    next_prefill_tokens = torch.tensor([0, 0, 77, 0], dtype=torch.int64, device=device)
    block_table = torch.tensor([block_table_values], dtype=torch.int32, device=device)

    prepare_dflash_inputs(
        input_buffers,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch,
        torch.tensor([num_sampled], dtype=torch.int32, device=device),
        torch.tensor([num_rejected], dtype=torch.int32, device=device),
        last_sampled,
        next_prefill_tokens,
        input_temperature,
        input_seeds,
        block_table,
        4,
        cp_rank,
        cp_size,
        cp_interleave,
        123,
        NUM_QUERY_PER_REQ,
        K,
        max_num_reqs,
        max_num_tokens,
        128,
        sample_from_anchor=False,
    )
    torch.cuda.synchronize()
    return SimpleNamespace(
        input_buffers=input_buffers,
        query_slot_mapping=query_slot_mapping.cpu(),
        context_positions=context_positions.cpu(),
        context_slot_mapping=context_slot_mapping.cpu(),
        sample_indices=sample_indices.cpu(),
        sample_pos=sample_pos.cpu(),
        sample_idx_mapping=sample_idx_mapping.cpu(),
        temperature=temperature.cpu(),
        seeds=seeds.cpu(),
    )


def contract_prepare_inputs() -> None:
    out = _run_prepare(
        target_positions=[10, 11, 12, 13],
        block_table_values=[0, 0, 7, 8, 9, 10, 11, 12],
    )
    _assert_equal(out.context_positions[:4].tolist(), [10, 11, 0, 0], "context")
    _assert_equal(
        out.context_slot_mapping[:4].tolist(),
        [30, 31, PAD_SLOT_ID, PAD_SLOT_ID],
        "rejected-context-slots",
    )
    _assert_equal(
        out.input_buffers.input_ids[:5].cpu().tolist(),
        [99, 123, 123, 123, 123],
        "query-inputs",
    )
    _assert_equal(
        out.input_buffers.positions[:5].cpu().tolist(),
        [12, 13, 14, 15, 16],
        "query-positions",
    )
    _assert_equal(out.query_slot_mapping[:5].tolist(), [32, 33, 34, 35, 36], "slots")
    _assert_equal(out.sample_indices[:4].tolist(), [1, 2, 3, 4], "sample-indices")
    _assert_equal(out.sample_pos[:4].tolist(), [13, 14, 15, 16], "sample-pos")
    _assert_equal(out.sample_idx_mapping[:4].tolist(), [2, 2, 2, 2], "sample-map")

    prefill = _run_prepare(
        target_positions=[20, 21, 22, 23],
        block_table_values=[0, 0, 0, 12, 13, 14, 15, 16],
        num_sampled=0,
        num_rejected=0,
    )
    _assert_equal(prefill.input_buffers.input_ids[0].item(), 77, "prefill-bonus")


def contract_private_ring() -> None:
    ring_size = 4
    block_table = torch.full((4, 8), -1, dtype=torch.int32, device="cuda")
    idx_mapping = torch.tensor([0, 2, 1, 3], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([0, 3, 15, 20], dtype=torch.int32, device="cuda")
    synthesize_draft_ring_block_tables(
        block_table,
        idx_mapping,
        seq_lens,
        block_size=4,
        ring_base=1,
        ring_size=ring_size,
        num_query_per_req=NUM_QUERY_PER_REQ,
    )
    torch.cuda.synchronize()
    actual = block_table.cpu().tolist()
    for row, (state_idx, seq_len) in enumerate(
        zip(idx_mapping.cpu().tolist(), seq_lens.cpu().tolist())
    ):
        count = (seq_len + NUM_QUERY_PER_REQ + 3) // 4
        base = 1 + state_idx * ring_size
        expected = [
            base + offset % ring_size if offset < count else 0 for offset in range(8)
        ]
        _assert_equal(actual[row], expected, f"ring-row-{row}")


def contract_draft_probability_order() -> None:
    vocab_size = 7
    probs = torch.arange(3 * K * vocab_size, dtype=torch.float32).view(3, K, vocab_size)
    probs = torch.softmax(probs, dim=-1)
    runner = object.__new__(GPUModelRunner)
    runner._draft_probs = probs
    runner._draft_prob_req_ids = ["request-b", "request-a", "request-c"]
    runner.input_batch = SimpleNamespace(
        req_ids=["request-a", "request-b", "request-c"]
    )
    metadata = SimpleNamespace(num_draft_tokens=[4, 2, 0])
    actual = runner._get_spec_decode_draft_probs(metadata)
    expected = torch.cat((probs[1, :4], probs[0, :2]), dim=0).contiguous()
    assert actual is not None and actual.is_contiguous()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(-1), torch.ones(actual.shape[0]))


def _rejection_inputs(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
) -> dict:
    num_trials = target_logits.shape[0]
    device = target_logits.device
    target_rows = (
        target_logits[:, None, :]
        .expand(-1, K + 1, -1)
        .reshape(num_trials * (K + 1), -1)
        .contiguous()
    )
    draft_rows = draft_logits[:, None, :].expand(-1, K, -1).contiguous()
    sampled = torch.zeros(num_trials, K + 1, dtype=torch.int64, device=device)
    sampled[:, 1:] = draft_tokens
    return {
        "target_logits": target_rows,
        "draft_logits": draft_rows,
        "draft_sampled": sampled.reshape(-1),
        "cu_num_logits": torch.arange(num_trials + 1, dtype=torch.int32, device=device)
        * (K + 1),
        "pos": torch.arange(num_trials * (K + 1), dtype=torch.int32, device=device),
        "idx_mapping": torch.arange(num_trials, dtype=torch.int32, device=device),
        "expanded_idx_mapping": torch.arange(
            num_trials, dtype=torch.int32, device=device
        ).repeat_interleave(K + 1),
        "expanded_local_pos": torch.arange(
            K + 1, dtype=torch.int32, device=device
        ).repeat(num_trials),
        "temperature": torch.ones(num_trials, dtype=torch.float32, device=device),
        "seed": torch.arange(num_trials, dtype=torch.int64, device=device),
    }


def contract_probabilistic_rejection() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    num_trials = 100_000
    vocab_size = 16
    target_1d = torch.randn(vocab_size, device=device)
    draft_1d = -target_1d
    target_logits = target_1d.expand(num_trials, -1).contiguous()
    draft_logits = draft_1d.expand(num_trials, -1).contiguous()
    draft_probs = torch.softmax(draft_1d, dim=-1)
    draft_tokens = torch.multinomial(
        draft_probs.expand(num_trials, -1), K, replacement=True
    )
    inputs = _rejection_inputs(target_logits, draft_logits, draft_tokens)
    sampled, num_sampled = rejection_sample(**inputs, num_speculative_steps=K)
    target_probs = torch.softmax(target_1d, dim=-1)
    for pos in range(K + 1):
        valid = num_sampled >= pos + 1
        tokens = sampled[valid, pos]
        observed = torch.bincount(tokens, minlength=vocab_size).float()
        expected = target_probs * tokens.numel()
        chi2 = ((observed - expected) ** 2 / expected).sum().item()
        threshold = (vocab_size - 1) + 10 * (2 * (vocab_size - 1)) ** 0.5
        if chi2 >= threshold:
            raise AssertionError(
                f"probabilistic-position-{pos}: chi2={chi2}, threshold={threshold}"
            )

    target = torch.full((1, 8), float("-inf"), device=device)
    draft = torch.full((1, 8), float("-inf"), device=device)
    target[0, 6] = 0
    draft[0, 1] = 0
    forced = _rejection_inputs(
        target,
        draft,
        torch.ones((1, K), dtype=torch.int64, device=device),
    )
    replacement, replacement_count = rejection_sample(**forced, num_speculative_steps=K)
    _assert_equal(replacement_count.item(), 1, "forced-rejection-count")
    _assert_equal(replacement[0, 0].item(), 6, "forced-replacement-token")


def contract_k4_scheduler(target: str, draft: str) -> None:
    target_config = ModelConfig(
        model=target,
        runner="generate",
        max_model_len=100,
        trust_remote_code=True,
    )
    speculative_config = SpeculativeConfig(
        target_model_config=target_config,
        target_parallel_config=ParallelConfig(),
        model=draft,
        method="dflash",
        num_speculative_tokens=K,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        max_model_len=100,
        is_encoder_decoder=target_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=target_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        speculative_config=speculative_config,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = 8
    scheduler = Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    _assert_equal(scheduler.num_lookahead_tokens, K + 1, "lookahead")

    init_none_hash(sha256)
    params = SamplingParams(max_tokens=16)
    params.update_from_generation_config({}, 50256)
    request = Request(
        request_id="k4",
        prompt_token_ids=[0] * BLOCK_SIZE,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    scheduler.add_request(request)
    output = scheduler.schedule()
    _assert_equal(output.num_scheduled_tokens["k4"], BLOCK_SIZE, "prefill-tokens")
    block_ids = output.scheduled_new_reqs[0].block_ids[0]
    if len(block_ids) != 2:
        raise AssertionError(f"lookahead-block-count: {len(block_ids)}")
    for position in range(BLOCK_SIZE, BLOCK_SIZE + K + 1):
        if position // BLOCK_SIZE >= len(block_ids):
            raise AssertionError(f"unallocated-lookahead-position: {position}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    contract_prepare_inputs()
    print("PASS prepare-inputs-k4")
    contract_private_ring()
    print("PASS private-ring-cold-partial-continuation-wrap-isolation")
    contract_draft_probability_order()
    print("PASS draft-probability-order-normalization-k4")
    contract_probabilistic_rejection()
    print("PASS probabilistic-rejection-replacement-k4")
    contract_k4_scheduler(args.target, args.draft)
    print("PASS scheduler-lookahead-k4")


if __name__ == "__main__":
    main()
