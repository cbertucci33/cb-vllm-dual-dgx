# Private captured request fixture removed from public archive.
# Private captured fixture excluded; supply a local synthetic fixture.
payload = {}
# Private captured fixture excluded; supply a local synthetic fixture.
incident = {}

import json, torch, xgrammar as xgr
from transformers import AutoTokenizer
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.parser_manager import ParserManager
t=AutoTokenizer.from_pretrained("/models/glm53-exl3",local_files_only=True,trust_remote_code=True)
r=ResponsesRequest(**payload)
f=ParserManager.get_parser(tool_parser_name="glm47",reasoning_parser_name="glm45",enable_auto_tools=True)
p=f(t,r.tools);r=p.adjust_request(r)
print("REQUEST",r.tool_choice,[(x.name,x.strict) for x in r.tools],type(p).__name__,flush=True)
tag=r.structured_outputs.structural_tag
print("TAG",tag,flush=True)
info=xgr.TokenizerInfo.from_huggingface(t,vocab_size=154880)
c=xgr.GrammarCompiler(info,max_threads=1)
ctx=c.compile_structural_tag(tag)

a=json.loads(incident["arguments"]+'"}')
xml="<tool_call>exec"+"".join("<arg_key>"+k+"</arg_key><arg_value>"+str(v)+"</arg_value>" for k,v in a.items())+"</tool_call>"

from types import SimpleNamespace
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.reasoning import ReasoningParserManager
manager=StructuredOutputManager.__new__(StructuredOutputManager)
manager.enable_in_reasoning=False
manager.reasoner_cls=ReasoningParserManager.get_reasoning_parser("glm45")
manager.tokenizer=t
import numpy as np
manager.vllm_config=SimpleNamespace(num_speculative_tokens=7,model_config=SimpleNamespace(is_diffusion=False))
manager._grammar_bitmask=xgr.allocate_token_bitmask(16,154880)
manager._full_mask=torch.tensor(-1,dtype=torch.int32)
manager.fill_bitmask_parallel_threshold=128
def newreq(name):
 sr=StructuredOutputRequest(r.structured_outputs)
 sr.grammar=XgrammarGrammar(vocab_size=154880,ctx=ctx,matcher=xgr.GrammarMatcher(ctx,override_stop_tokens=[154820,154827,154829],max_rollback_tokens=8))
 return SimpleNamespace(structured_output_request=sr,prompt_token_ids=t.encode("<think>",add_special_tokens=False),request_id=name,all_token_ids=[],use_structured_output=True)
def commit(req,ids):
 req.all_token_ids.extend(ids)
 assert manager.accept_tokens(req,ids)
def allows(row,tok):return (int(row[tok//32])>>(tok%32))&1

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
