exec(open('/opt/project/evidence/vllm-v032-migration-20261018/source/reconcile_models.py').read().split("p='vllm/model_executor/model_loader/default_loader.py'")[0])
def bynumber(p,choices):
 i=0
 def f(m):
  nonlocal i
  c=choices[i];i+=1
  return m.group(1) if c=='ours' else m.group(2) if c=='theirs' else m.group(1)+'\n\n'+m.group(2)
 s=re.sub(r'^<<<<<<< ours\n(.*?)^=======\n(.*?)^>>>>>>> theirs\n',f,get(p),flags=re.M|re.S)
 assert i==len(choices),(p,i,len(choices));put(p,s)
bynumber('tests/model_executor/test_flashinfer_autotune_warmup.py',['theirs']*6+['ours'])
bynumber('tests/models/glm5next/test_kda_recurrent.py',['both','theirs'])
p='tests/models/glm5next/test_kda_state_dtype.py';put(p,blob('HEAD',p))
p='tests/models/glm5next/test_sparse_indexer_topk_dispatch.py';s=blob('HEAD',p);l=blob('9a3af353e8',p);a=l.index('@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])');s+='\n\n'+l[a:];s=s.replace('import torch\n','import torch\nfrom vllm.utils.torch_utils import set_default_torch_dtype\n',1);put(p,s)
for p in ['tests/models/multimodal/processing/test_glm5next.py','tests/multimodal/test_video.py','tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py','tests/v1/attention/test_sparse_mla_backends.py','tests/v1/attention/test_sparse_mla_kv_cache_layout.py','tests/v1/core/test_kv_cache_utils.py','tests/v1/worker/test_attn_utils.py','tests/v1/worker/test_utils.py']:
 put(p,resolve(p,'theirs'))
for p in ['tests/test_config.py','tests/v1/attention/test_gdn_metadata_builder.py','tests/v1/core/test_single_type_kv_cache_manager.py','tests/v1/kv_connector/unit/test_nixl_desc_geometry.py']:
 put(p,resolve(p,'ours'))
for p in ['tests/utils_/test_torch_utils.py','tests/v1/worker/test_gpu_model_runner_v2.py','tests/v1/worker/test_gpu_rejection_sampler_chunking.py']:
 put(p,combine(p))
p='tests/v1/attention/test_indexer_tp_row_shard.py';put(p,blob('HEAD',p))
p='tests/v1/attention/test_flashmla_nope_sm90_backend_selection.py';put(p,blob('HEAD',p))
bynumber('tests/v1/core/test_scheduler.py',['ours','ours','theirs','theirs','theirs','both'])
bynumber('tests/v1/worker/test_gpu_model_runner_prefill_tail_classification.py',['ours','ours','ours','theirs','theirs','theirs','theirs','theirs'])
bynumber('tests/v1/worker/test_gpu_target_dependent_ar_speculator.py',['ours','theirs','theirs'])
p='docs/design/mm_processing.md';put(p,blob('HEAD',p))
print('Non-parser/native test conflicts resolved retaining independent regression cases and .32 fixtures.')
