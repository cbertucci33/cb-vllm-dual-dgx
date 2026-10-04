# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import gather_draft_sampled


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_unproposed_prefill_draft_slots_are_rejected():
    device = torch.device("cuda")
    input_ids = torch.tensor([101, 202, 303], device=device)
    positions = torch.tensor([9, 10, 11], device=device)
    logits_indices = torch.tensor([0, 1, 2], device=device)
    expanded_idx_mapping = torch.zeros(3, dtype=torch.int32, device=device)
    expanded_local_pos = torch.arange(3, dtype=torch.int32, device=device)

    draft_sampled, pos = gather_draft_sampled(
        input_ids,
        positions,
        logits_indices,
        expanded_idx_mapping,
        expanded_local_pos,
        torch.tensor([10], dtype=torch.int32, device=device),
    )
    assert draft_sampled.tolist() == [101, -1, -1]
    assert torch.equal(pos, positions)

    draft_sampled, pos = gather_draft_sampled(
        input_ids,
        positions,
        logits_indices,
        expanded_idx_mapping,
        expanded_local_pos,
        torch.tensor([9], dtype=torch.int32, device=device),
    )
    assert torch.equal(draft_sampled, input_ids)
    assert torch.equal(pos, positions)
