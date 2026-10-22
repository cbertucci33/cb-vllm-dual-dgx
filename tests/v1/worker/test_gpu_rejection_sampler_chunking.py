# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import MethodType, SimpleNamespace
from typing import get_args

import numpy as np
import pytest
import torch

from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.sample.output import SamplingMaskTensors
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    RejectionSampler,
    _iter_request_chunks,
)


def _make_input_batch(
    cu_num_logits_np: np.ndarray,
    idx_mapping_np: np.ndarray,
    device: torch.device,
) -> SimpleNamespace:
    """Minimal InputBatch fields read by RejectionSampler._verify_in_chunks."""
    num_reqs = len(idx_mapping_np)
    num_logits_per_req = np.diff(cu_num_logits_np)
    return SimpleNamespace(
        num_reqs=num_reqs,
        cu_num_logits_np=cu_num_logits_np,
        cu_num_logits=torch.from_numpy(cu_num_logits_np).to(device),
        idx_mapping_np=idx_mapping_np,
        idx_mapping=torch.from_numpy(idx_mapping_np).to(device),
        expanded_idx_mapping=torch.from_numpy(
            np.repeat(idx_mapping_np, num_logits_per_req)
        ).to(device),
        expanded_local_pos=torch.from_numpy(
            np.concatenate(
                [np.arange(count, dtype=np.int32) for count in num_logits_per_req]
            )
        ).to(device),
        seq_lens_cpu_upper_bound=torch.full((num_reqs,), 64, dtype=torch.int32),
    )


def _make_rejection_sampler(
    sampler: SimpleNamespace, num_speculative_steps: int
) -> RejectionSampler:
    """RejectionSampler for standard, fixed-boundary, unwatermarked verification."""
    rejection_sampler = object.__new__(RejectionSampler)
    rejection_sampler.sampler = sampler
    rejection_sampler.num_speculative_steps = num_speculative_steps
    rejection_sampler.enable_adaptive_verification = False
    rejection_sampler.synthetic_conditional_rates = None
    rejection_sampler.use_block_verification = False
    rejection_sampler.watermark_key = None
    return rejection_sampler


def test_iter_request_chunks_preserves_request_boundaries():
    cu_num_logits = np.array([0, 3, 4, 11, 13], dtype=np.int32)

    assert list(_iter_request_chunks(cu_num_logits, max_chunk_logits=5)) == [
        (0, 2),
        (2, 3),
        (3, 4),
    ]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("logprobs_mode", get_args(LogprobsMode))
def test_chunked_scores_match_full_batch(logprobs_mode: str):
    device = torch.device("cuda")
    cu_num_logits_np = np.array([0, 3, 4, 8, 10], dtype=np.int32)
    num_logits_per_req = np.diff(cu_num_logits_np)
    idx_mapping_np = np.array([7, 2, 9, 1], dtype=np.int32)
    input_batch = _make_input_batch(cu_num_logits_np, idx_mapping_np, device)
    rejection_sampler = _make_rejection_sampler(
        SimpleNamespace(logprobs_mode=logprobs_mode, return_sampling_mask=False),
        num_speculative_steps=3,
    )

    def fake_verify(
        self,
        logits,
        _draft_logits,
        _draft_sampled,
        _pos,
        cu_num_logits,
        idx_mapping,
        *_mappings,
    ):
        num_sampled = torch.diff(cu_num_logits).to(torch.int32)
        sampled = (
            idx_mapping.to(torch.int64).unsqueeze(1) + torch.arange(4, device=device)
        ) % logits.shape[1]
        return logits.float() + 1, sampled, num_sampled

    rejection_sampler._verify = MethodType(fake_verify, rejection_sampler)
    logits = torch.arange(170, dtype=torch.float32, device=device).view(10, 17)

    sampled, num_sampled, chunked_logprobs, sampling_mask_tensors = (
        rejection_sampler._verify_in_chunks(
            logits,
            input_batch,
            draft_logits=None,
            draft_sampled=torch.arange(10, device=device),
            pos=torch.arange(10, device=device),
            max_chunk_logits=5,
            max_num_logprobs=2,
        )
    )
    score_logits = logits + 1 if logprobs_mode in PROCESSED_LOGPROBS_MODES else logits
    full_logprobs = rejection_sampler._get_logprobs_tensors(
        sampled,
        num_sampled,
        score_logits,
        input_batch.cu_num_logits,
        input_batch.cu_num_logits_np,
        max_num_logprobs=2,
    )

    assert sampled[:, 0].tolist() == idx_mapping_np.tolist()
    assert num_sampled.tolist() == num_logits_per_req.tolist()
    assert chunked_logprobs is not None
    assert full_logprobs is not None
    assert torch.equal(
        chunked_logprobs.logprob_token_ids,
        full_logprobs.logprob_token_ids,
    )
    assert torch.equal(chunked_logprobs.logprobs, full_logprobs.logprobs)
    assert torch.equal(
        chunked_logprobs.selected_token_ranks,
        full_logprobs.selected_token_ranks,
    )
    assert (
        chunked_logprobs.cu_num_generated_tokens
        == full_logprobs.cu_num_generated_tokens
    )
    assert sampling_mask_tensors is None


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_replay_on_off_preserves_rejection_sampling_and_rng(monkeypatch):
    device = torch.device("cuda")
    num_reqs = 3
    num_speculative_steps = 3
    rows_per_request = num_speculative_steps + 1
    vocab_size = 8
    cu_num_logits_np = np.arange(0, 13, rows_per_request, dtype=np.int32)
    idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
    input_batch = _make_input_batch(cu_num_logits_np, idx_mapping_np, device)
    processed_logits = torch.zeros((12, vocab_size), device=device)
    processed_logits[0, 0] = -float("inf")
    processed_logits[rows_per_request + 1, 1] = -float("inf")
    draft_logits = torch.zeros(
        (num_reqs, num_speculative_steps, vocab_size), device=device
    )
    draft_sampled = torch.zeros(
        (num_reqs, rows_per_request), dtype=torch.int64, device=device
    )
    draft_sampled[:, 1:] = torch.tensor([0, 1, 2], device=device)
    draft_sampled = draft_sampled.flatten()
    sampler = SimpleNamespace(
        logprobs_mode="processed_logprobs",
        return_sampling_mask=False,
        sampling_states=SimpleNamespace(
            top_k=SimpleNamespace(np=np.full(num_reqs, vocab_size)),
            temperature=SimpleNamespace(gpu=torch.ones(num_reqs, device=device)),
            seeds=SimpleNamespace(
                gpu=torch.arange(num_reqs, dtype=torch.int64, device=device)
            ),
        ),
        use_fp64_gumbel=False,
        apply_sampling_params=lambda logits, *_: logits,
    )
    rejection_sampler = _make_rejection_sampler(sampler, num_speculative_steps)
    pos = torch.arange(12, dtype=torch.int32, device=device)

    pack_calls: list[tuple[list[int], int, int]] = []
    pack = SamplingMaskTensors.from_logits.__func__

    def track_pack(
        cls,
        logits,
        cu_num_logits,
        num_sampled_tokens,
        max_num_kept,
        rows_per_request=1,
    ):
        pack_calls.append(
            (cu_num_logits.cpu().tolist(), max_num_kept, rows_per_request)
        )
        return pack(
            cls,
            logits,
            cu_num_logits,
            num_sampled_tokens,
            max_num_kept,
            rows_per_request,
        )

    monkeypatch.setattr(
        SamplingMaskTensors,
        "from_logits",
        classmethod(track_pack),
    )

    rng_before = torch.cuda.get_rng_state()
    off_sampled, off_counts, off_logprobs, off_masks = (
        rejection_sampler._verify_in_chunks(
            processed_logits,
            input_batch,
            draft_logits,
            draft_sampled,
            pos,
            max_chunk_logits=8,
            max_num_logprobs=0,
        )
    )
    rng_after_off = torch.cuda.get_rng_state()
    assert pack_calls == []

    sampler.return_sampling_mask = True
    on_sampled, on_counts, on_logprobs, on_masks = rejection_sampler._verify_in_chunks(
        processed_logits,
        input_batch,
        draft_logits,
        draft_sampled,
        pos,
        max_chunk_logits=8,
        max_num_logprobs=0,
    )
    rng_after_on = torch.cuda.get_rng_state()

    assert pack_calls == [
        ([0, 4, 8], vocab_size, rows_per_request),
        ([0, 4], vocab_size, rows_per_request),
    ]
    assert off_masks is None and on_masks is not None
    assert on_masks.rows_per_request == rows_per_request
    assert on_masks.token_ids.shape == (num_reqs * rows_per_request, vocab_size)
    assert on_masks.packed_mask.shape[0] == num_reqs * rows_per_request
    assert off_counts.tolist() == [1, 2, 4]
    assert torch.equal(off_counts, on_counts)
    counts = off_counts.cpu().tolist()
    for req_idx, count in enumerate(counts):
        assert torch.equal(off_sampled[req_idx, :count], on_sampled[req_idx, :count])

    mapped_rows = [
        int(input_batch.cu_num_logits_np[req_idx]) + slot_idx
        for req_idx, count in enumerate(counts)
        for slot_idx in range(count)
    ]
    assert off_logprobs is not None and on_logprobs is not None
    assert torch.equal(
        off_logprobs.logprob_token_ids[mapped_rows],
        on_logprobs.logprob_token_ids[mapped_rows],
    )
    assert torch.equal(
        off_logprobs.logprobs[mapped_rows], on_logprobs.logprobs[mapped_rows]
    )
    assert torch.equal(
        off_logprobs.selected_token_ranks[mapped_rows],
        on_logprobs.selected_token_ranks[mapped_rows],
    )
    assert torch.equal(rng_before, rng_after_off)
    assert torch.equal(rng_after_off, rng_after_on)

    masks = (
        on_masks.to_cpu_nonblocking().tolists(on_counts.cpu().numpy()).to_nested_list()
    )
    expected_masks = [
        torch.isfinite(processed_logits[row]).nonzero().flatten().tolist()
        for row in mapped_rows
    ]
    emitted = [
        int(on_sampled[req_idx, slot_idx])
        for req_idx, count in enumerate(on_counts.cpu().tolist())
        for slot_idx in range(count)
    ]
    expected_logprobs = torch.stack(
        [
            torch.log_softmax(processed_logits[row], dim=-1)[token_id]
            for row, token_id in zip(mapped_rows, emitted)
        ]
    )

    assert masks == expected_masks
    assert all(token_id in mask for token_id, mask in zip(emitted, masks))
    assert torch.equal(on_logprobs.logprobs[mapped_rows, 0], expected_logprobs)


@pytest.mark.skipif(
    not current_platform.is_cuda() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="Requires CUDA or Triton CPU interpreter",
)
@pytest.mark.parametrize("logprobs_mode", get_args(LogprobsMode))
@pytest.mark.parametrize("num_drafts", range(1, 8))
@pytest.mark.parametrize("mixed_topk", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("through_runner", [False, True])
def test_speculative_custom_logprob_ids_preserve_request_slots(
    monkeypatch, logprobs_mode, num_drafts, mixed_topk, chunked, through_runner
):
    """Speculation must return requested IDs even without ordinary logprobs."""
    from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates

    device = torch.device("cuda" if current_platform.is_cuda() else "cpu")
    counts = np.array([num_drafts + 1, 1, num_drafts + 1], dtype=np.int32)
    cu_np = np.concatenate(
        [np.array([0], dtype=np.int32), counts.cumsum(dtype=np.int32)]
    )
    slots = np.array([5, 1, 3], dtype=np.int32)
    nrows = int(cu_np[-1])
    expanded = np.repeat(slots, counts)
    local_pos = np.concatenate([np.arange(n, dtype=np.int32) for n in counts])
    batch = SimpleNamespace(
        num_reqs=3,
        num_draft_tokens=nrows - 3,
        cu_num_logits_np=cu_np,
        cu_num_logits=torch.tensor(cu_np, device=device),
        idx_mapping_np=slots,
        idx_mapping=torch.tensor(slots, device=device),
        expanded_idx_mapping=torch.tensor(expanded, device=device),
        expanded_local_pos=torch.tensor(local_pos, device=device),
        input_ids=torch.arange(nrows, device=device, dtype=torch.int64) % 11,
        positions=torch.tensor(10 + local_pos, device=device, dtype=torch.int64),
        logits_indices=torch.arange(nrows, device=device, dtype=torch.int64),
        seq_lens_cpu_upper_bound=torch.tensor(10 + counts),
        seq_lens=torch.tensor(10 + counts, device=device),
    )
    state = object.__new__(SamplingStates)
    state.num_logprobs = np.full(6, NO_LOGPROBS, dtype=np.int32)
    if mixed_topk:
        state.num_logprobs[1] = 3
    custom = object.__new__(LogprobTokenIdsState)
    custom_counts = np.array([0, 0, 0, 1, 0, 2], dtype=np.int32)
    custom.num_token_ids = SimpleNamespace(
        np=custom_counts, gpu=torch.tensor(custom_counts, device=device)
    )
    custom_ids = torch.zeros((6, 2), device=device, dtype=torch.int32)
    custom_ids[5] = torch.tensor([2, 7], device=device)
    custom_ids[3, 0] = 4
    custom.token_ids = SimpleNamespace(gpu=custom_ids)
    sampler = object.__new__(Sampler)
    sampler.sampling_states = state
    sampler.logprob_token_ids_state = custom
    sampler.logprobs_mode = logprobs_mode
    sampler.compute_nans = False
    sampler.req_states = SimpleNamespace(
        prefill_len=SimpleNamespace(gpu=torch.ones(6, dtype=torch.int32, device=device))
    )
    rejection = object.__new__(RejectionSampler)
    rejection.sampler = sampler
    rejection.enable_adaptive_verification = False
    rejection.num_speculative_steps = num_drafts

    def verify(self, logits, _q, _drafts, _pos, cu, idx, *_rest):
        # Control acceptance only; execute production chunking and score kernels.
        accepted = torch.diff(cu).to(torch.int32)
        accepted = torch.where(idx == 5, accepted - 1, accepted)
        sampled = (
            idx.to(torch.int64)[:, None] + torch.arange(num_drafts + 1, device=device)
        ) % logits.shape[1]
        processed = logits.float() * 0.7
        return processed, sampled, accepted

    rejection._verify = MethodType(verify, rejection)
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler.get_max_chunk_logits",
        lambda _vocab: num_drafts + 1 if chunked else nrows,
    )
    logits = (
        torch.arange(nrows * 11, dtype=torch.float32, device=device).reshape(nrows, 11)
        % 19
    ) / 3
    if through_runner:
        from vllm.v1.worker.gpu import model_runner
        from vllm.v1.worker.gpu.sample import batch_shard

        metadata = SimpleNamespace(
            num_local_reqs=3,
            max_num_reqs_per_rank=3,
            max_num_logits_per_req=num_drafts + 1,
            gathered_src_indices=torch.arange(3, dtype=torch.int32, device=device),
        )
        runner = SimpleNamespace(
            sampler=sampler,
            rejection_sampler=rejection,
            speculator=SimpleNamespace(draft_logits=None),
            device=device,
            vocab_size=11,
            model=SimpleNamespace(compute_logits_local=lambda x: x),
            batch_sharder=SimpleNamespace(
                shard_sampler_inputs=lambda b, g: (b, b.logits_indices, g, metadata)
            ),
        )
        monkeypatch.setattr(model_runner, "all_to_all_logits", lambda x, _: x)
        # Execute production pack/unpack; only the transport is local here.
        monkeypatch.setattr(
            batch_shard, "tensor_model_parallel_all_gather", lambda x, dim=0: x
        )
        result, _, _ = model_runner.GPUModelRunner.sample(runner, logits, batch, None)
    else:
        result = rejection(logits, batch)
    scores = result.logprobs_tensors
    assert scores is not None
    assert scores.logprob_token_ids.shape == (nrows, 1 + (3 if mixed_topk else 2))
    assert scores.cu_num_generated_tokens == cu_np.tolist()
    score_logits = logits * 0.7 if logprobs_mode in PROCESSED_LOGPROBS_MODES else logits
    expected_scores = (
        score_logits
        if logprobs_mode.endswith("logits")
        else score_logits.log_softmax(-1)
    )
    for req, slot in enumerate(slots):
        for offset in range(int(result.num_sampled[req])):
            row = int(cu_np[req]) + offset
            selected = int(result.sampled_token_ids[req, offset])
            requested = {5: [2, 7], 3: [4]}.get(int(slot))
            if requested is None:
                requested = (
                    torch.topk(score_logits[row], 3).indices.tolist()
                    if mixed_topk
                    else []
                )
            expected_ids = [selected, *requested]
            assert (
                scores.logprob_token_ids[row, : len(expected_ids)].tolist()
                == expected_ids
            )
            torch.testing.assert_close(
                scores.logprobs[row, : len(expected_ids)],
                expected_scores[row, expected_ids],
            )
            assert torch.isneginf(scores.logprobs[row, len(expected_ids) :]).all()
            assert int(scores.selected_token_ranks[row]) == int(
                (score_logits[row] >= score_logits[row, selected]).sum()
            )
    assert result.num_sampled.tolist() == [num_drafts, 1, num_drafts + 1]
    assert result.num_rejected.tolist() == [1, 0, 0]
