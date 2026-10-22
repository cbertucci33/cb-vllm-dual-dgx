exec(open('/opt/project/evidence/vllm-v032-migration-20261018/source/reconcile_models.py').read().split("p='vllm/model_executor/model_loader/default_loader.py'")[0])
for p in ['vllm/v1/core/sched/scheduler.py','vllm/v1/engine/core.py','vllm/v1/worker/gpu/block_table.py','vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py']:
 put(p,resolve(p,'ours'))
p='vllm/v1/core/single_type_kv_cache_manager.py';put(p,resolve(p,'ours'))
p='vllm/v1/worker/gpu/attn_utils.py';s=get(p)
s=re.sub(r'^<<<<<<< ours\n(.*?)^=======\n(.*?)^>>>>>>> theirs\n',lambda m:m.group(2) if 'is_draft_layer' in m.group(2) else m.group(1),s,flags=re.M|re.S);put(p,s)
p='vllm/v1/worker/gpu/model_runner.py';put(p,combine(p))
p='vllm/v1/worker/gpu/spec_decode/rejection_sampler.py';s=get(p)
s=re.sub(r'^<<<<<<< ours\n(.*?)^=======\n(.*?)^>>>>>>> theirs\n',lambda m:'        max_per_req_token_ids: int = 0,\n'+m.group(1) if 'SamplingMaskTensors' in m.group(1) else m.group(2),s,flags=re.M|re.S);put(p,s)
p='vllm/v1/worker/gpu/spec_decode/dflash/speculator.py';s=resolve(p,'theirs')
s=s.replace('precompute_context_kv=self._prepare_graph_forward,','prepare_forward=self._prepare_graph_forward,')
s=s.replace('    def _prepare_graph_forward(self, num_reqs: int) -> None:', '    def _prepare_graph_forward(\n        self, num_reqs: int, attn_metadata: dict[str, Any] | None\n    ) -> None:')
s=s.replace('        self._precompute_context_kv(0, num_context)\n','        self._precompute_context_kv(0, num_context)\n        if attn_metadata is not None and self.supports_decode_metadata_update:\n            self._update_draft_decode_metadata(attn_metadata, num_reqs)\n',1)
u=blob('HEAD',p);a=u.index('        is_full_cudagraph = batch_desc.cg_mode == CUDAGraphMode.FULL', u.index('    def propose('));b=u.index('        # DFlash processes all speculative tokens',a)
a2=s.index('        # Rebuild the draft attention metadata even when replaying');b2=s.index('        # DFlash processes all speculative tokens',a2);s=s[:a2]+u[a:b]+s[b2:];put(p,s)
p='vllm/v1/worker/gpu_model_runner.py';s=get(p)
def runner(m):
 a,b=m.groups()
 if 'from vllm.model_executor.model_loader.reload.inplace' in a:return a+'from vllm.model_executor.model_loader.derived_weights import refresh_derived_weights\n'
 if 'is_draft_layer' in b:return b
 if 'sync_error = None' in b:return b.replace('            self.cache_config.num_gpu_blocks = None\n','            self.cache_config.num_gpu_blocks = None\n            self._init_block_sizes = []\n')
 return a
s=re.sub(r'^<<<<<<< ours\n(.*?)^=======\n(.*?)^>>>>>>> theirs\n',runner,s,flags=re.M|re.S)
needle='        self.reset_lora_state()\n        self.reset_encoder_cache()'
assert needle in s
s=s.replace(needle,'        if not is_checkpoint_format:\n            refresh_derived_weights(self.get_model())\n        draft_model = self.get_draft_model()\n        if draft_model is not None:\n            refresh_derived_weights(draft_model)\n'+needle,1)
put(p,s)
p='vllm/v1/worker/gpu_worker.py';s=get(p)
def worker(m):
 a,b=m.groups()
 if 'refresh_derived_weights' in b:return a+b
 if 'configure_startup_memory_guard' in b:return a+'            if current_platform.is_cuda():\n                configure_startup_memory_guard(init_snapshot, self.requested_memory)\n'
 return b.replace('            ) as explicit_profile:', '            ) as explicit_profile, set_current_vllm_config(self.vllm_config):')
s=re.sub(r'^<<<<<<< ours\n(.*?)^=======\n(.*?)^>>>>>>> theirs\n',worker,s,flags=re.M|re.S);put(p,s)
p='vllm/v1/worker/utils.py';put(p,resolve(p,'theirs'))
print('Runtime conflict resolutions written; DFlash context graph now uses .32 callback signature; not staged.')
