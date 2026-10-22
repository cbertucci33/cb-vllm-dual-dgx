import torch,json
from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla as run
d=torch.load('/glm-diagnostics/mla-internal-199.pt',weights_only=False)
q=d['q'].cuda().unsqueeze(1); ids=d['indices'].clone();valid=ids>=0;ids[valid]-=d['row_ids'][0]; ids=ids.cuda().unsqueeze(1)
workspace=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8)
for fill in [0,127]:
 cache=torch.full((2,64,528),fill,device='cuda',dtype=torch.uint8)
 # Leave row0 invalid/unwritten, as the real allocator does.
 cache.reshape(-1,528)[1:27]=d['rows'].cuda()
 ix=torch.where(ids>=0,ids+1,ids)
 out=run(query=q,kv_cache=cache.unsqueeze(1),workspace_buffer=workspace,qk_nope_head_dim=256,kv_lora_rank=512,qk_rope_head_dim=0,block_tables=ix,seq_lens=None,max_seq_len=2176,bmm1_scale=d['scale'],bmm2_scale=1.,sparse_mla_top_k=2176,kv_scale_format='arbitrary_fp32')
 print(json.dumps({'fill':fill,'finite':out.isfinite().all().item(),'nan':out.isnan().sum().item()}),flush=True)
print('peak',torch.cuda.max_memory_allocated())
