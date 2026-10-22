# Private captured fixture excluded; supply a local synthetic fixture.
payload = {}

import json,torch
from transformers import AutoTokenizer
from vllm.parser.parser_manager import ParserManager
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
t=AutoTokenizer.from_pretrained('/models/glm53-exl3',local_files_only=True,trust_remote_code=True)
factory=ParserManager.get_parser(tool_parser_name='glm47',reasoning_parser_name='glm45',enable_auto_tools=True)
r=ResponsesRequest(model='model-main',input=[],tools=payload['tools']);r.include_reasoning=True
a=json.loads(payload['arguments']+'"}')
xml='<tool_call>exec'+''.join('<arg_key>'+k+'</arg_key><arg_value>'+str(v)+'</arg_value>' for k,v in a.items())+'</tool_call>'
count=0
for explicit in [False,True]:
 ids=t.encode(('</think>' if explicit else '')+xml,add_special_tokens=False)
 # Every single boundary as well as uniform K7+bonus chunks, repeated calls.
 chunks=[[ids[:k],ids[k:]] for k in range(1,len(ids))]
 chunks += [[ids[k:k+w] for k in range(0,len(ids),w)] for w in range(1,9)]
 for parts in chunks:
  p=factory(t,tools=r.tools);out=''
  for j,chunk in enumerate(parts):
   for dm in p.parse_deltas(t.decode(chunk,skip_special_tokens=False),chunk,r,finished=j==len(parts)-1):
    for tc in dm.tool_calls or []:out+=(tc.function and tc.function.arguments) or ''
  assert json.loads(out)==a,(explicit,parts,out)
  count+=1
print('PARSER_CASES',count,'CUDA',torch.cuda.is_initialized(),flush=True)
