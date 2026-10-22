import json,torch
from vllm import _custom_ops as ops
from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
print('START',flush=True)
torch.manual_seed(123)
dev='cuda'; rows=128; heads=32; dim=512; topk=2176
latent=torch.randn(rows,dim,device=dev,dtype=torch.bfloat16)*0.3
cache=torch.zeros(2,64,528,device=dev,dtype=torch.uint8)
slots=torch.arange(rows,device=dev,dtype=torch.int64)
ops.concat_and_cache_mla(latent,torch.empty(rows,0,device=dev,dtype=torch.bfloat16),cache,slots,'fp8_ds_mla',torch.ones(1,device=dev))
p=cache.reshape(rows,528)
scales=p[:,512:528].contiguous().view(torch.float32)
kv=(p[:,:512].contiguous().view(torch.float8_e4m3fn).float().reshape(rows,4,128)*scales.unsqueeze(-1)).reshape(rows,512)
print(json.dumps({'writer_max_error':(kv-latent.float()).abs().max().item(),'scales_finite':scales.isfinite().all().item()}),flush=True)
workspace=torch.empty(64*1024*1024,device=dev,dtype=torch.uint8)
for n in [1,8,64]:
 q=torch.randn(n,1,heads,dim,device=dev,dtype=torch.bfloat16)*0.3
 idx=torch.full((n,1,topk),-1,device=dev,dtype=torch.int32)
 idx[:,:,:rows]=torch.arange(rows,device=dev,dtype=torch.int32)
 out=trtllm_batch_decode_with_kv_cache_mla(query=q,kv_cache=cache.unsqueeze(1),workspace_buffer=workspace,qk_nope_head_dim=512,kv_lora_rank=512,qk_rope_head_dim=0,block_tables=idx,seq_lens=None,max_seq_len=topk,bmm1_scale=dim**-0.5,bmm2_scale=1.,sparse_mla_top_k=topk,kv_scale_format='arbitrary_fp32')
 ref=torch.softmax(q[:,0].float()@kv.T*dim**-0.5,dim=-1)@kv
 err=out[:,0].float()-ref
 print(json.dumps({'tokens':n,'finite':out.isfinite().all().item(),'max_error':err.abs().max().item(),'rms_error':err.square().mean().sqrt().item(),'ref_rms':ref.square().mean().sqrt().item()}),flush=True)
print('END',torch.cuda.max_memory_allocated(),flush=True)
