# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU interpreter contracts for complete target/proposal vocabulary support."""

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample


@pytest.mark.parametrize("k", range(1, 8))
@pytest.mark.parametrize("block", [False, True])
def test_target_only_token_is_not_removed_by_proposal_width(k, block):
    # A probability-one target token outside q's support must be returned at
    # every possible rejection position, including an all-accepted bonus.
    for reject_at in range(k + 1):
        p = torch.full((k + 1, 6), -1000.0)
        p[:, 0] = 0
        p[reject_at, 0] = -1000
        p[reject_at, 5] = 0
        q = torch.full((1, k, 4), -1000.0)
        q[:, :, 0] = 0
        sampled, count = rejection_sample(
            p,
            q,
            torch.zeros(k + 1, dtype=torch.int64),
            torch.tensor([0, k + 1], dtype=torch.int32),
            torch.arange(799990, 799991 + k, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(k + 1, dtype=torch.int32),
            torch.arange(k + 1, dtype=torch.int32),
            torch.tensor([0.8]),
            torch.tensor([1234], dtype=torch.int64),
            k,
            use_block_verification=block,
        )
        assert count.tolist() == [reject_at + 1]
        assert sampled[0, :reject_at].tolist() == [0] * reject_at
        assert sampled[0, reject_at].item() == 5
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("k", range(1, 8))
@pytest.mark.parametrize("block", [False, True])
@pytest.mark.parametrize("widths", [(4, 6), (6, 4), (6, 6), (8201, 4), (4, 8201)])
def test_zero_mass_padding_keeps_same_target_law(k, block, widths):
    # p=q=[1/8, 3/8, 1/2, 0, ...]. Width/extra zero-mass blocks may differ.
    # All proposed tokens must be accepted, whatever their probability.
    pv, qv = widths
    p = torch.full((k + 1, pv), -float("inf"))
    q = torch.full((1, k, qv), -float("inf"))
    law = torch.tensor([0.125, 0.375, 0.5]).log()
    p[:, :3] = law
    q[:, :, :3] = law
    drafts = torch.tensor([0] + [i % 3 for i in range(k)], dtype=torch.int64)
    sampled, count = rejection_sample(
        p,
        q,
        drafts,
        torch.tensor([0, k + 1], dtype=torch.int32),
        torch.arange(900, 901 + k, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(k + 1, dtype=torch.int32),
        torch.arange(k + 1, dtype=torch.int32),
        torch.tensor([1.0]),
        torch.tensor([98765], dtype=torch.int64),
        k,
        use_block_verification=block,
    )
    assert count.tolist() == [k + 1]
    assert torch.equal(sampled[0, :k], drafts[1:])
    assert sampled[0, k].item() in (0, 1, 2)


@pytest.mark.parametrize("k", range(1, 8))
@pytest.mark.parametrize("block", [False, True])
def test_proposal_only_token_is_rejected_without_target_oob(k, block):
    p = torch.full((k + 1, 4), -1000.0)
    p[:, 3] = 0
    q = torch.full((1, k, 6), -1000.0)
    q[:, :, 5] = 0
    sampled, count = rejection_sample(
        p,
        q,
        torch.full((k + 1,), 5, dtype=torch.int64),
        torch.tensor([0, k + 1], dtype=torch.int32),
        torch.arange(k + 1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(k + 1, dtype=torch.int32),
        torch.arange(k + 1, dtype=torch.int32),
        torch.tensor([0.7]),
        torch.tensor([8765], dtype=torch.int64),
        k,
        use_block_verification=block,
    )
    assert count.tolist() == [1]
    assert sampled[0, 0].item() == 3
