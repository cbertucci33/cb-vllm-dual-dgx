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
cases=0
# At each possible real sequence boundary, draft up to 7 following tokens.
# Rejection at each draft position must leave that position's target mask intact.
for explicit in [False,True]:
 seq=t.encode(("</think>" if explicit else "")+xml,add_special_tokens=False)
 for start in range(len(seq)):
  for n in range(0,min(7,len(seq)-start)+1):
   req=newreq("A");commit(req,seq[:start])
   before=req.structured_output_request.grammar.num_processed_tokens
   draft=seq[start:start+n]
   valid=manager.validate_tokens(req,draft)
   assert valid==draft
   got=manager.grammar_bitmask({"A":req},["A"],{"A":draft}).copy()
   assert req.structured_output_request.grammar.num_processed_tokens==before
   # An independently advanced grammar gives the expected next-token mask.
   for j in range(n+1):
    ref=newreq("ref");commit(ref,seq[:start+j])
    expected=manager.grammar_bitmask({"ref":ref},["ref"],{}).copy()[0]
    # At transition itself explicit-end is unconstrained; after it exact match.
    assert np.array_equal(got[j],expected),(explicit,start,n,j)
   cases+=1
print("MASK_ROLLBACK_CASES",cases,flush=True)
# Open final argument: validate an invalid closer/EOS at every draft position.
openids=t.encode(xml.rsplit("</arg_value>",1)[0],add_special_tokens=False)
badcases=0
for stop in [154844,154820,154827,154829]:
 for keep in range(7):
  req=newreq("A");commit(req,openids[:-keep] if keep else openids)
  draft=(openids[-keep:] if keep else [])+[stop]
  accepted=manager.validate_tokens(req,draft)
  assert len(accepted)==keep,(stop,keep,accepted)
  padded=accepted+[-1]*(7-keep)
  masks=manager.grammar_bitmask({"A":req},["A"],{"A":padded}).copy()
  assert not allows(masks[keep],stop),(stop,keep)
  # All later rows may be full, but verifier cutoff excludes local draft >=keep.
  badcases+=1
print("INVALID_SUFFIX_CASES",badcases,flush=True)
# Request reordering/fresh request uses identity, independent of fixed mask block.
a_req=newreq("A");commit(a_req,openids)
b_req=newreq("B");commit(b_req,t.encode(xml,add_special_tokens=False))
out=manager.grammar_bitmask({"A":a_req,"B":b_req},["B","A"],{}).copy()
assert allows(out[0],154820) and not allows(out[8],154820)
print("REQUEST_REORDER_PASS",flush=True)
print("CUDA",torch.cuda.is_initialized(),flush=True)
