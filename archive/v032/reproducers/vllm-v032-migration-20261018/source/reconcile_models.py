from pathlib import Path
import subprocess,re
root=Path('/opt/project/builds/glm-mimo-main-fixes-20261017')
def blob(rev,p):return subprocess.check_output(['git','show',f'{rev}:{p}'],cwd=root,text=True)
def get(p):return(root/p).read_text()
def put(p,s):
 assert '<<<<<<<' not in s,p
 (root/p).write_text(s)
def resolve(p,side):
 s=get(p)
 return re.sub(r'^<<<<<<< ours\n(.*?)^=======\n(.*?)^>>>>>>> theirs\n',lambda m:m.group(1 if side=='ours' else 2),s,flags=re.M|re.S)
def combine(p):return re.sub(r'^<<<<<<< ours\n(.*?)^=======\n(.*?)^>>>>>>> theirs\n',lambda m:m.group(1)+m.group(2),get(p),flags=re.M|re.S)
p='vllm/model_executor/model_loader/default_loader.py';s=resolve(p,'ours');s=s.replace('\n    def track_weights_loading(', '\n        refresh_derived_weights(model)\n\n    def track_weights_loading(',1);put(p,s)
p='vllm/model_executor/model_loader/weight_utils.py';put(p,combine(p))
p='vllm/model_executor/models/mimo_v2.py';put(p,resolve(p,'theirs'))
p='vllm/model_executor/models/mimo_v2_omni.py';s=get(p)
# Preserve upstream embedding metadata and local video/audio hashing together.
s=re.sub(r'^<<<<<<< ours\n(.*?)^=======\n(.*?)^>>>>>>> theirs\n',lambda m:m.group(1)+'\n\n'+m.group(2) if 'class MiMoV2OmniMultiModalDataParser' in m.group(1) else m.group(1).replace('return MiMoV2OmniMultiModalDataParser(', 'return MiMoDataParser('),s,flags=re.M|re.S)
s=s.replace('class MiMoDataParser(MultiModalDataParser):','class MiMoDataParser(MiMoV2OmniMultiModalDataParser):');put(p,s)
p='vllm/models/glm5next/common/model.py';s=resolve(p,'ours')
# Retain explicit unsupported-config guards, but use upstream aux-state timing.
old=blob('9a3af353e8',p)
validation=old[old.index('def _validate_supported_config('):old.index('\n\nclass Glm5NextMLP')]
s=s[:s.index('def _validate_supported_config(')]+validation+s[s.index('\n\nclass Glm5NextMLP'):]
u=blob('HEAD',p);a=u.index('    def _aux_hidden_state(');b=u.index('    def load_weights(',a)
a2=s.index('    def _aux_hidden_state(');b2=s.index('    def load_weights(',a2);s=s[:a2]+u[a:b]+s[b2:];put(p,s)
p='vllm/models/glm5next/common/multimodal.py';s=resolve(p,'theirs')
s=s.replace('from collections.abc import Mapping\n','')
s=s.replace('import math\n','import math\nfrom collections.abc import Mapping\n',1)
s=s.replace('from vllm.multimodal.parse import ImageSize\n','from vllm.multimodal.parse import ImageSize, MultiModalDataItems\n')
u=blob('HEAD',p);a=u.index('    def _hf_processor_applies_updates(');b=u.index('\n    def ',a+5) if '\n    def ' in u[a+5:] else len(u)
s=s.rstrip()+'\n\n'+u[a:b] if '    def _hf_processor_applies_updates(' not in s else s
put(p,s)
p='vllm/model_executor/warmup/flashinfer_autotune_cache.py';put(p,combine(p))
p='vllm/model_executor/warmup/kernel_warmup.py';s=resolve(p,'theirs')
# Replace duplicated merged helper block with local exact-coverage+bounded-BF16 helpers.
l=blob('9a3af353e8',p);a=l.index('def _flashinfer_autotune_token_counts(');b=l.index('def flashinfer_autotune(',a)
x=s.index('def _flashinfer_autotune_token_counts(');y=s.index('def flashinfer_autotune(',x);s=s[:x]+l[a:b]+s[y:]
s=s.replace('def flashinfer_autotune(runner: "GPUModelRunner") -> None:', 'def flashinfer_autotune(runner: "GPUModelRunner", *, skip_attn: bool = False) -> None:')
s=s.replace('            skip_attn = skip_attn or hisparse_enabled\n','')
s=s.replace('_run_flashinfer_autotune_dummy_runs(runner, skip_attn=skip_attn)', '_run_flashinfer_autotune_dummy_runs(runner, skip_attn=skip_attn or hisparse_enabled)')
needle='    finally:\n        set_autotune_process_group(None)'
assert needle in s
s=s.replace(needle, '        with torch.inference_mode():\n            _run_flashinfer_bf16_autotune_dummy_run(\n                runner, skip_ops=skip_ops, skip_attn=skip_attn or hisparse_enabled\n            )\n'+needle)
if '_FLASHINFER_BF16_AUTOTUNE_MAX_TOKENS =' not in s:
 line=next(x for x in l.splitlines() if x.startswith('_FLASHINFER_BF16_AUTOTUNE_MAX_TOKENS ='))
 s=s.replace('def _flashinfer_autotune_token_counts(',line+'\n\n\ndef _flashinfer_autotune_token_counts(',1)
put(p,s)
p='vllm/utils/flashinfer.py';put(p,resolve(p,'theirs'))
p='vllm/multimodal/video.py';put(p,resolve(p,'theirs'))
p='vllm/v1/kv_cache_interface.py';put(p,resolve(p,'theirs'))
p='vllm/v1/core/kv_cache_coordinator.py';put(p,combine(p))
p='vllm/v1/core/kv_cache_utils.py';put(p,resolve(p,'theirs'))
print('Model/loader/warmup/cache combined resolutions written; not staged.')
