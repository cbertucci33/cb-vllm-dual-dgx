import os,json,dataclasses,traceback
import torch

def forbidden(*args,**kwargs):
    raise RuntimeError('CPU_ONLY_HARNESS: device initialization forbidden')
torch.cuda._lazy_init=forbidden
import vllm.platforms as platforms
from vllm.platforms.cuda import CudaPlatformBase
from vllm.platforms.interface import DeviceCapability
class MetadataCuda(CudaPlatformBase):
    @classmethod
    def get_device_capability(cls,device_id=0): return DeviceCapability(12,1)
    @classmethod
    def get_device_name(cls,device_id=0): return 'NVIDIA GB10'
    @classmethod
    def num_compute_units(cls,device_id=0): return 48
    @classmethod
    def device_count(cls): return 1
    @classmethod
    def is_fully_connected(cls,device_ids): return True
platforms._current_platform=MetadataCuda()
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
args=EngineArgs(model='/models/mimo-exl3',served_model_name=['mimo-v2.6-flash-mopd-heretic-exl3'],trust_remote_code=True,quantization='exl3',dtype='bfloat16',tensor_parallel_size=2,gpu_memory_utilization=.85,kv_cache_memory_bytes=5294899200,max_model_len=600000,max_num_seqs=2,block_size=16,mm_processor_cache_gb=.5,load_format='instanttensor',max_num_batched_tokens=4096,speculative_config={'model':'/models/mimo-dflash','method':'dflash','num_speculative_tokens':7,'draft_tensor_parallel_size':2,'kv_cache_dtype':'fp8_e4m3','draft_sample_method':'probabilistic'},kv_cache_dtype='fp8_e4m3',attention_backend='TRITON_ATTN_DIFFKV',linear_backend='b12x',enable_prefix_caching=True,enable_chunked_prefill=True,reasoning_parser='mimo',override_generation_config={'temperature':1.0,'top_p':.95},async_scheduling=False,distributed_executor_backend='mp',nnodes=2,node_rank=1,master_addr='10.100.40.1',master_port=29525)
c=args.create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER,headless=True)
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import get_eagle3_aux_layers_from_config
out={'provenance':'CPU-only config with GB10 metadata stand-in; no CUDA context, no model construction','cuda_initialized':torch.cuda.is_initialized(),'aux_layers':get_eagle3_aux_layers_from_config(c.speculative_config),'target_head_dtype':str(c.model_config.head_dtype)}
for name in ['compilation_config','scheduler_config','cache_config','speculative_config','parallel_config']:
    out[name]=dataclasses.asdict(getattr(c,name))
print('MATERIALIZED_JSON='+json.dumps(out,default=str))
