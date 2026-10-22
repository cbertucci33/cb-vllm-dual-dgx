# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the compact tensor-parallel candidate exchange."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers import logits_processor


def _processor():
    processor = object.__new__(logits_processor.LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.scale = 1.0
    processor.soft_cap = None
    return processor


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("ties", [False, True])
def test_pair_exchange_matches_two_gather_order(monkeypatch, dtype, ties):
    shards = [
        torch.tensor([[1, 5, 4, 100], [2, 7, 1, 100]], dtype=dtype),
        torch.tensor([[6, 3, 2, 100], [8, 5, 4, 100]], dtype=dtype),
    ]
    if ties:
        shards[1][0, 0] = 5
        shards[1][1, 0] = 7
    k = 2
    topks = []
    for rank, shard in enumerate(shards):
        masked = shard.clone()
        masked[:, -1] = -torch.inf
        values, ids = masked.topk(k, dim=-1)
        topks.append((values, ids + rank * 3))
    all_scores = torch.cat([item[0] for item in topks], dim=-1)
    all_ids = torch.cat([item[1] for item in topks], dim=-1)
    expected_values, positions = all_scores.topk(k, dim=-1)
    expected_ids = all_ids.gather(-1, positions)
    packed = [
        torch.stack((scores.float(), ids.float()), dim=-1).flatten(-2)
        for scores, ids in topks
    ]

    for rank in range(2):
        calls: list[torch.Tensor] = []

        def gather(value, dim=-1, *, rank=rank, calls=calls):
            assert dim == -1
            torch.testing.assert_close(value, packed[rank])
            calls.append(value)
            return torch.cat(packed, dim=-1)

        monkeypatch.setattr(
            logits_processor, "tensor_model_parallel_all_gather", gather
        )
        head = SimpleNamespace(
            tp_size=2,
            num_embeddings_padded=8,
            shard_indices=SimpleNamespace(
                num_org_vocab_padding=1, org_vocab_start_index=rank * 3
            ),
        )

        def checked_topk(scores, count):
            assert scores.dtype == dtype
            assert scores.is_contiguous()
            return torch.topk(scores, count, dim=-1)

        monkeypatch.setattr(logits_processor, "_topk", checked_topk)
        ids, values = _processor().get_top_k_tokens(
            head, torch.empty(2, 0), k, local_logits=shards[rank].clone()
        )
        assert len(calls) == 1
        assert ids.dtype == torch.int64 and values.dtype == torch.float32
        torch.testing.assert_close(ids, expected_ids)
        torch.testing.assert_close(values, expected_values.float())


@pytest.mark.parametrize(
    "dtype,padded_size",
    [(torch.float64, 8), (torch.float32, 2**24 + 2)],
)
def test_nonrepresentable_domain_keeps_two_gathers(monkeypatch, dtype, padded_size):
    local = torch.tensor([[1.25, 4.5, 2.75]], dtype=dtype)
    remote = torch.tensor([[5.5, 3.5]], dtype=dtype)
    expected_local_values, expected_local_ids = local.topk(2)
    remote_ids = torch.tensor([[2**24 + 1, 2**24]], dtype=torch.int64)
    gathered = (
        torch.cat((expected_local_values, remote), -1),
        torch.cat((expected_local_ids, remote_ids), -1),
    )
    calls: list[torch.Tensor] = []

    def gather(value, dim=-1):
        assert dim == -1
        expected = (expected_local_values, expected_local_ids)[len(calls)]
        torch.testing.assert_close(value, expected)
        calls.append(value)
        return gathered[len(calls) - 1]

    monkeypatch.setattr(logits_processor, "tensor_model_parallel_all_gather", gather)
    head = SimpleNamespace(
        tp_size=2,
        num_embeddings_padded=padded_size,
        shard_indices=SimpleNamespace(num_org_vocab_padding=0, org_vocab_start_index=0),
    )
    ids, values = _processor().get_top_k_tokens(
        head, torch.empty(1, 0), 2, local_logits=local.clone()
    )
    assert len(calls) == 2
    assert ids.tolist() == [[2**24 + 1, 1]]
    assert values.dtype == torch.float32
    torch.testing.assert_close(values, torch.tensor([[5.5, 4.5]]))


def test_single_rank_has_no_candidate_collective(monkeypatch):
    monkeypatch.setattr(
        logits_processor,
        "tensor_model_parallel_all_gather",
        lambda *args, **kwargs: pytest.fail("TP1 must not gather"),
    )
    head = SimpleNamespace(
        tp_size=1,
        num_embeddings_padded=4,
        shard_indices=SimpleNamespace(num_org_vocab_padding=1, org_vocab_start_index=0),
    )
    ids, values = _processor().get_top_k_tokens(
        head,
        torch.empty(1, 0),
        2,
        local_logits=torch.tensor([[1.0, 3.0, 2.0, 100.0]]),
    )
    assert ids.tolist() == [[1, 2]]
    assert values.tolist() == [[3.0, 2.0]]


def test_logprob_consumer_keeps_partition_and_one_candidate_exchange(monkeypatch):
    shards = (
        torch.tensor([[1.0, 3.0, 2.0, 100.0]]),
        torch.tensor([[6.0, 5.0, 0.0, 100.0]]),
    )
    valid = [shard[:, :3] for shard in shards]
    local_topk = [shard.topk(2) for shard in valid]
    pairs = [
        torch.stack((topk.values, (topk.indices + rank * 3).float()), dim=-1).flatten(
            -2
        )
        for rank, topk in enumerate(local_topk)
    ]
    partition = torch.cat([shard.logsumexp(-1, keepdim=True) for shard in valid], -1)
    expected = torch.cat(valid, -1).log_softmax(-1).topk(2)

    for rank in range(2):
        calls: list[torch.Tensor] = []

        def gather(value, dim=-1, *, rank=rank, calls=calls):
            assert dim == -1
            if not calls:
                torch.testing.assert_close(
                    value, valid[rank].logsumexp(-1, keepdim=True)
                )
                result = partition
            else:
                torch.testing.assert_close(value, pairs[rank])
                result = torch.cat(pairs, -1)
            calls.append(value)
            return result

        monkeypatch.setattr(
            logits_processor, "tensor_model_parallel_all_gather", gather
        )
        head = SimpleNamespace(
            tp_size=2,
            num_embeddings_padded=8,
            shard_indices=SimpleNamespace(
                num_org_vocab_padding=1, org_vocab_start_index=rank * 3
            ),
        )
        ids, values = _processor().get_top_k_tokens(
            head,
            torch.empty(1, 0),
            2,
            local_logits=shards[rank].clone(),
            return_log_probs=True,
        )
        assert len(calls) == 2
        torch.testing.assert_close(ids, expected.indices)
        torch.testing.assert_close(values, expected.values)


def test_largest_exact_fp32_token_id_uses_one_exchange(monkeypatch):
    local = torch.tensor([[1.0, 2.0]])
    remote = torch.tensor([[3.0, float(2**24)]])
    calls: list[torch.Tensor] = []

    def gather(value, dim=-1):
        assert dim == -1
        calls.append(value)
        return torch.cat((value, remote), dim=-1)

    monkeypatch.setattr(logits_processor, "tensor_model_parallel_all_gather", gather)
    head = SimpleNamespace(
        tp_size=2,
        num_embeddings_padded=2**24 + 1,
        shard_indices=SimpleNamespace(num_org_vocab_padding=0, org_vocab_start_index=0),
    )
    ids, values = _processor().get_top_k_tokens(
        head, torch.empty(1, 0), 1, local_logits=local
    )
    assert len(calls) == 1
    assert ids.tolist() == [[2**24]]
    assert values.tolist() == [[3.0]]
