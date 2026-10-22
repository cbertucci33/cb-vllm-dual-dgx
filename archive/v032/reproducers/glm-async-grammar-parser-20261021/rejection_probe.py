import torch
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import _rejection_kernel
cases=0
for block in [False,True]:
 for temp in [0.,1.]:
  for cutoff in range(8):
   sampled=torch.full((1,8),-9,dtype=torch.int64)
   lengths=torch.zeros(1,dtype=torch.int64)
   zeros=torch.zeros((8,1))
   sums=torch.full((8,1),32.)
   target=torch.zeros((8,32))
   drafts=torch.zeros((2,7,32))
   draft_ids=torch.tensor([0]+[0]*cutoff+[-1]*(7-cutoff),dtype=torch.int64)
   _rejection_kernel[(1,)](
    sampled_ptr=sampled,sampled_stride=8,rejected_steps_ptr=lengths,
    target_rejected_logsumexp_ptr=torch.zeros(1),draft_rejected_logsumexp_ptr=torch.zeros(1),
    target_logits_ptr=target,target_logits_stride=32,
    target_local_argmax_ptr=torch.zeros((8,1),dtype=torch.int64),target_local_argmax_stride=1,
    target_local_max_ptr=zeros,target_local_max_stride=1,
    target_local_sumexp_ptr=sums,target_local_sumexp_stride=1,
    draft_sampled_ptr=draft_ids,draft_logits_ptr=drafts,draft_logits_stride_0=224,draft_logits_stride_1=32,
    target_vocab_size=32,draft_vocab_size=32,
    draft_local_max_ptr=zeros,draft_local_max_stride=1,draft_local_sumexp_ptr=sums,draft_local_sumexp_stride=1,
    cu_num_logits_ptr=torch.tensor([0,8],dtype=torch.int32),idx_mapping_ptr=torch.tensor([1],dtype=torch.int32),
    temp_ptr=torch.tensor([temp,temp]),seed_ptr=torch.tensor([17,31],dtype=torch.int64),
    pos_ptr=torch.arange(8,dtype=torch.int64),synthetic_conditional_rates_ptr=torch.ones(7),
    cumulative_log_p_ptr=torch.zeros(8),local_residual_mass_ptr=zeros,local_residual_mass_stride=1,
    vocab_num_blocks=1,PADDED_VOCAB_NUM_BLOCKS=1,HAS_DRAFT_LOGITS=True,SYNTHETIC_MODE=False,USE_BLOCK_VERIFICATION=block)
   assert lengths.item()==cutoff,(block,temp,cutoff,lengths)
   cases+=1
print('ACTUAL_REJECTION_KERNEL_CPU_INTERPRETER_CASES',cases,'CUDA',torch.cuda.is_initialized(),flush=True)
