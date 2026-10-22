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
for label,s in [("valid",xml),("missing_value_end",xml.replace("</arg_value></tool_call>","</tool_call>")),("unclosed",xml[:xml.rfind("</arg_value>")])]:
 m=xgr.GrammarMatcher(ctx,override_stop_tokens=[154820,154827,154829],max_rollback_tokens=8)
 ids=t.encode(s,add_special_tokens=False);failed=[]
 for i,x in enumerate(ids):
  if not m.accept_token(x):failed.append((i,x,t.decode([x])));break
 print(label,"tokens",len(ids),"failure",failed,"terminated",m.is_terminated(),flush=True)
 if not failed:
  for stop in [154820,154827,154829]:
   accepted=m.accept_token(stop);print("EOS",stop,accepted,flush=True)
   if accepted:m.rollback(1)
print("CUDA",torch.cuda.is_initialized(),flush=True)
