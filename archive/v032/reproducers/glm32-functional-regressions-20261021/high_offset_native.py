import json
import torch
from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla as run

d = torch.load('/evidence/mla-internal-199.pt', map_location='cpu', weights_only=False)
ids = d['indices'].cuda().unsqueeze(1)
q = d['q'].cuda().unsqueeze(1)
row_ids = d['row_ids'].cuda()
blocks = (int(row_ids.max()) + 64) // 64
cache = torch.empty((blocks, 1, 64, 528), device='cuda', dtype=torch.uint8)
workspace = torch.empty(64 * 1024 * 1024, device='cuda', dtype=torch.uint8)
results = []
for fill in (0, 127):
    cache.fill_(fill)
    cache.view(-1, 528)[row_ids] = d['rows'].cuda()
    out = run(
        query=q,
        kv_cache=cache,
        workspace_buffer=workspace,
        qk_nope_head_dim=256,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        block_tables=ids,
        seq_lens=None,
        max_seq_len=2176,
        bmm1_scale=d['scale'],
        bmm2_scale=1.0,
        sparse_mla_top_k=2176,
        kv_scale_format='arbitrary_fp32',
    )
    torch.cuda.synchronize()
    results.append(out.clone())
    print(json.dumps({'fill': fill, 'finite': bool(out.isfinite().all()),
                      'nan_count': int(torch.isnan(out).sum()),
                      'max_abs': float(torch.nan_to_num(out).abs().max())}), flush=True)
print(json.dumps({'equal': bool(torch.equal(*results)),
                  'blocks': blocks, 'cache_bytes': cache.numel(),
                  'peak_gpu_bytes': torch.cuda.max_memory_allocated()}))
