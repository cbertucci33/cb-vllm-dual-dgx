import json
import torch
from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla as run

d=torch.load('/evidence/mla-internal-199.pt',map_location='cpu',weights_only=False)
q=d['q'].cuda().unsqueeze(1)
source_ids=d['indices'].cuda()
source_base=int(d['row_ids'][0])
rows=d['rows'].cuda()
workspace=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8)
blocks=(source_base+26+63)//64
cache=torch.empty((blocks,1,64,528),device='cuda',dtype=torch.uint8)
result=[]
for base in [1,64,4096,65536,262144,1048576,source_base]:
    cache.fill_(127)
    pos=torch.arange(base,base+26,device='cuda',dtype=torch.int64)
    cache.view(-1,528)[pos]=rows
    eq=bool(torch.equal(cache.view(-1,528)[pos],rows))
    ids=source_ids.clone()
    valid=ids>=0
    ids[valid]+=base-source_base
    out=run(query=q,kv_cache=cache,workspace_buffer=workspace,qk_nope_head_dim=256,kv_lora_rank=512,qk_rope_head_dim=0,block_tables=ids.unsqueeze(1),seq_lens=None,max_seq_len=2176,bmm1_scale=d['scale'],bmm2_scale=1.,sparse_mla_top_k=2176,kv_scale_format='arbitrary_fp32')
    torch.cuda.synchronize()
    item={'base':base,'rows_equal':eq,'finite':bool(out.isfinite().all()),'nans':int(torch.isnan(out).sum())}
    print(json.dumps(item),flush=True)
    result.append(item)
print(json.dumps({'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'results':result}))
