# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU interpreter checks for exact GLM sparse pool expansion and tail writes."""

import pytest
import torch

from vllm.models.glm5next.nvidia.ops.kpool_compress import (
    expand_pools_and_append_tail,
)


@pytest.mark.parametrize("k", range(1, 8))
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_expand_pools_into_existing_strided_output(k, dtype):
    rows = k + 1
    # Exercise the exact 512-pool/2048-token geometry and 800K positions.
    pools = torch.arange(512, dtype=dtype).repeat(rows, 1) + 199400
    pools[:, -2:] = -1
    lengths = torch.arange(799993, 799993 + rows, dtype=torch.int32)
    storage = torch.full((rows + 2, 2059), -777, dtype=torch.int32)
    destination = storage[1:-1, 3:2054]
    before = pools.clone()
    returned = expand_pools_and_append_tail(pools, lengths, 4, out=destination)
    assert returned is destination
    assert torch.equal(pools, before)
    oracle = torch.full((rows, 2051), -1, dtype=torch.int32)
    for row in range(rows):
        for column, pool in enumerate(pools[row].tolist()):
            if pool >= 0:
                oracle[row, column * 4 : column * 4 + 4] = torch.arange(
                    pool * 4, pool * 4 + 4
                )
        tail_start = int(lengths[row]) // 4 * 4
        count = int(lengths[row]) - tail_start
        oracle[row, 2048 : 2048 + count] = torch.arange(tail_start, tail_start + count)
    assert torch.equal(destination, oracle)
    assert torch.all(storage[[0, -1]] == -777)
    assert torch.all(storage[:, :3] == -777)
    assert torch.all(storage[:, 2054:] == -777)
    allocated = expand_pools_and_append_tail(pools, lengths, 4)
    assert torch.equal(allocated, oracle)
    assert not torch.cuda.is_initialized()
