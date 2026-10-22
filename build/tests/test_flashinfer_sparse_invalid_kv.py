# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import torch
from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla as run

torch.manual_seed(123)
values = (torch.randn(26, 512) * 0.3).to(torch.float8_e4m3fn)
rows = torch.cat((values.view(torch.uint8), torch.ones(26, 4).view(torch.uint8)), dim=1)
indices = torch.full((26, 2176), -1, dtype=torch.int32)
for i in range(26):
    indices[i, : i + 1] = torch.arange(i + 1)
d = {
    "q": torch.randn(26, 32, 512).bfloat16() * 0.3,
    "rows": rows,
    "indices": indices,
    "row_ids": torch.arange(26),
    "scale": 0.0625,
}
workspace = torch.empty(64 * 1024 * 1024, device="cuda", dtype=torch.uint8)
checks = []
for rope in [0, 64]:
    for n in [1, 8, 26]:
        q = d["q"][:n].cuda()
        if rope:
            q = torch.cat(
                (q, torch.zeros(n, 32, rope, device="cuda", dtype=q.dtype)), dim=-1
            )
        q = q.unsqueeze(1)
        for pattern in ["causal", "holes", "empty"]:
            ids = d["indices"][:n].clone()
            valid = ids >= 0
            ids[valid] -= d["row_ids"][0] - 1
            if pattern == "holes":
                ids[:, 1::2] = -1
            if pattern == "empty":
                ids.fill_(-1)
            ids = ids.cuda().unsqueeze(1)
            outputs = []
            for fill in [0, 127]:
                cache = torch.full(
                    (2, 64, 528 + rope * 2), fill, device="cuda", dtype=torch.uint8
                )
                cache.reshape(-1, 528 + rope * 2)[1:27, :528] = d["rows"].cuda()
                if rope:
                    cache.reshape(-1, 656)[1:27, 528:] = 0
                out = run(
                    query=q,
                    kv_cache=cache.unsqueeze(1),
                    workspace_buffer=workspace,
                    qk_nope_head_dim=256 if not rope else 128,
                    kv_lora_rank=512,
                    qk_rope_head_dim=rope,
                    block_tables=ids,
                    seq_lens=None,
                    max_seq_len=2176,
                    bmm1_scale=d["scale"],
                    bmm2_scale=1.0,
                    sparse_mla_top_k=2176,
                    kv_scale_format="arbitrary_fp32" if not rope else "pow2_fp32",
                )
                assert out.isfinite().all(), (rope, n, pattern, fill)
                outputs.append(out.clone())
            assert torch.equal(*outputs), (
                rope,
                n,
                pattern,
                "unused rows change result",
            )
            if pattern == "empty":
                assert torch.count_nonzero(outputs[0]) == 0
            checks.append([rope, n, pattern])
print(
    json.dumps(
        {
            "passed": len(checks),
            "cases": checks,
        }
    )
)
