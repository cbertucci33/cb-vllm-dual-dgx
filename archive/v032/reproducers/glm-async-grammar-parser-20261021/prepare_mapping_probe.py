from pathlib import Path
d=Path('/opt/project/evidence/glm-async-grammar-parser-20261021')
s=(d/'mask_probe.py').read_text().split('cases=0')[0]
s+="""
from vllm.v1.worker.gpu import structured_outputs as so
# Exercise the actual Triton masking kernel in CPU interpreter mode.
a_req=newreq('A');commit(a_req,t.encode(xml.rsplit('</arg_value>',1)[0],add_special_tokens=False))
b_req=newreq('B');commit(b_req,t.encode(xml,add_special_tokens=False))
masks=torch.from_numpy(manager.grammar_bitmask({'A':a_req,'B':b_req},['B','A'],{}).copy())
# Only position0 is initialized when drafts empty; deliberately fill remaining
# masks with distinct patterns to test mapping and trimmed-row bounds.
for i in range(1,8):masks[i].fill_(-1 if i%2 else 0)
for i in range(9,16):masks[i].fill_(-1 if i%2 else 0)
counts=[4,8]
cu=torch.tensor([0,4,12],dtype=torch.int32)
mapping=torch.tensor(list(range(8,16))+list(range(8)),dtype=torch.int32)
logits=torch.zeros((12,154880))
so._apply_grammar_bitmask_kernel[(16,19)](logits,logits.stride(0),mapping,cu,masks,masks.stride(0),154880,MASK_STRIDE=8,BLOCK_SIZE=8192)
ids=torch.arange(154880)
for mi,encoded in enumerate(mapping.tolist()):
 reqidx,pos=divmod(encoded,8)
 if pos>=counts[reqidx]:continue
 forbidden=((masks[mi][ids//32]>>(ids%32))&1)==0
 assert torch.equal(torch.isneginf(logits[int(cu[reqidx])+pos]),forbidden)
# Transfer replaced only in this isolated CPU probe; original cutoff function.
so.async_tensor_h2d=lambda x,device:torch.as_tensor(x,device=device)
checks=0
for limit in range(8):
 for trim in range(8):
  lens=[trim+1,8]
  batch=SimpleNamespace(num_draft_tokens=trim+7,req_ids=['A','B'],num_reqs=2,num_draft_tokens_per_req=np.array([trim,7]),expanded_local_pos=torch.tensor(list(range(trim+1))+list(range(8))),cu_num_logits=torch.tensor([0,trim+1,trim+9]))
  got=so.grammar_invalid_drafts(batch,['B','A'],[7,limit])
  expected=batch.expanded_local_pos>torch.tensor([limit]*(trim+1)+[7]*8)
  assert (got is None and not expected.any()) or torch.equal(got,expected)
  checks+=1
print('ACTUAL_KERNEL_CPU_INTERPRETER_MAPPING_PASS','CUTOFF_CASES',checks,'CUDA',torch.cuda.is_initialized(),flush=True)
"""
(d/'mapping_probe.py').write_text(s)
