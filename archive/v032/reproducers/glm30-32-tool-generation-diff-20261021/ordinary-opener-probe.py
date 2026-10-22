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

parts=["<tool","_call>"]
ordinary=sum((t.encode(z,add_special_tokens=False) for z in parts),[])
print("ORDINARY",ordinary,t.decode(ordinary),"SPECIAL",t.encode("<tool_call>",add_special_tokens=False),flush=True)
xml_open=xml[:xml.rfind("</arg_value>")]
ids=ordinary+t.encode(xml_open[len("<tool_call>"):],add_special_tokens=False)
sr=StructuredOutputRequest(r.structured_outputs)
sr.grammar=XgrammarGrammar(vocab_size=154880,ctx=ctx,matcher=xgr.GrammarMatcher(ctx,override_stop_tokens=[154820,154827,154829],max_rollback_tokens=8))
req=SimpleNamespace(structured_output_request=sr,prompt_token_ids=t.encode("<think>",add_special_tokens=False),request_id="offline",all_token_ids=[],use_structured_output=True)
for i in range(0,len(ids),7):
 chunk=ids[i:i+7];v=manager.validate_tokens(req,chunk);req.all_token_ids.extend(chunk);acc=manager.accept_tokens(req,chunk)
 if v!=chunk or not acc: print("REJECT",i,v,acc)
print("REASONING",sr.reasoning_ended,"BOUNDS",manager._get_constraint_bounds(req,[154820]),flush=True)
req.all_token_ids.append(154820)
print("EOS_ACCEPT",manager.accept_tokens(req,[154820]),flush=True)
p=f(t,r.tools);r.include_reasoning=True;out=""
for i in range(0,len(ids),7):
 chunk=ids[i:i+7]
 for dm in p.parse_deltas(t.decode(chunk,skip_special_tokens=False),chunk,r,prompt_token_ids=t.encode("<think>",add_special_tokens=False),finished=i+7>=len(ids)):
  for tc in dm.tool_calls or []:out+=(tc.function and tc.function.arguments) or ""
print("PARSER_OUTPUT",repr(out),"MATCH",out==incident["arguments"],flush=True)
print("CUDA",torch.cuda.is_initialized(),flush=True)
