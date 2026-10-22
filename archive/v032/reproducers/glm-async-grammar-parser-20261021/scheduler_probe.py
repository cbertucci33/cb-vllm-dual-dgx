from __future__ import annotations
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

from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.structured_output.utils import strip_speculative_padding
def get_grammar_bitmask(self, scheduler_output: SchedulerOutput) -> GrammarOutput | None:
    if not scheduler_output.has_structured_output_requests:
        return None
    structured_output_request_ids = [req_id for req_id in scheduler_output.num_scheduled_tokens if (req := self.requests.get(req_id)) and (req.use_structured_output and (not req.is_prefill_chunk))]
    if not structured_output_request_ids:
        return None
    bitmask = self.structured_output_manager.grammar_bitmask(self.requests, structured_output_request_ids, scheduler_output.scheduled_spec_decode_tokens)
    spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    num_acceptable_drafts = [len(strip_speculative_padding(spec_tokens.get(req_id, []))) for req_id in structured_output_request_ids]
    return GrammarOutput(structured_output_request_ids, bitmask, num_acceptable_drafts)
def update_draft_token_ids_in_output(self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput) -> None:
    num_invalid_spec_tokens: dict[str, int] = {}
    sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    for req_id, spec_token_ids in zip(draft_token_ids.req_ids, draft_token_ids.draft_token_ids):
        request = self.requests.get(req_id)
        if request is None or request.is_finished():
            continue
        placeholder_spec_tokens = sched_spec_tokens.get(req_id)
        if not placeholder_spec_tokens:
            continue
        orig_num_spec_tokens = len(placeholder_spec_tokens)
        del spec_token_ids[orig_num_spec_tokens:]
        spec_token_ids = self.structured_output_manager.validate_tokens(request, spec_token_ids)
        num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
        if num_invalid_tokens:
            spec_token_ids.extend([-1] * num_invalid_tokens)
            num_invalid_spec_tokens[req_id] = num_invalid_tokens
        sched_spec_tokens[req_id] = spec_token_ids
    scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens
openids=t.encode(xml.rsplit('</arg_value>',1)[0],add_special_tokens=False)
checks=0
for bad in [154844,154820,154827,154829]:
 for keep in range(7):
  req=newreq('A');req.is_finished=lambda:False;req.is_prefill_chunk=False
  commit(req,openids[:-keep] if keep else openids)
  draft=(openids[-keep:] if keep else [])+[bad]+[0]*(6-keep)
  sched=SimpleNamespace(requests={'A':req},structured_output_manager=manager)
  output=SimpleNamespace(scheduled_spec_decode_tokens={'A':[-1]*7},has_structured_output_requests=True,num_scheduled_tokens={'A':8})
  update_draft_token_ids_in_output(sched,SimpleNamespace(req_ids=['A'],draft_token_ids=[draft]),output)
  assert output.scheduled_spec_decode_tokens['A'][keep:]==[-1]*(7-keep)
  grammar=get_grammar_bitmask(sched,output)
  assert grammar.num_acceptable_drafts==[keep]
  assert not allows(grammar.grammar_bitmask[keep],bad)
  # Rejected proposal must not advance persistent grammar; valid replacement
  # delimiter commits, then EOS cannot stop before tool closer.
  closer=t.encode('</arg_value>',add_special_tokens=False)
  commit(req,output.scheduled_spec_decode_tokens['A'][:keep]+closer)
  assert manager.validate_tokens(req,[154820])==[]
  checks+=1
print('ACTUAL_SCHEDULER_HANDOFF_CASES',checks,'CUDA',torch.cuda.is_initialized(),flush=True)
