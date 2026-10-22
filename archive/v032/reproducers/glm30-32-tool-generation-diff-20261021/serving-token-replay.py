import json,sys,torch
from transformers import AutoTokenizer
from vllm.parser.parser_manager import ParserManager
Glm47MoeParser = ParserManager.get_parser(tool_parser_name='glm47',reasoning_parser_name='glm45',enable_auto_tools=True)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
payload=json.loads(sys.stdin.readline())
t=AutoTokenizer.from_pretrained('/models/glm53-exl3',local_files_only=True,trust_remote_code=True)
a=json.loads(payload['arguments']+'"}')
r=ResponsesRequest(model='model-main',input=[],tools=[payload['tool']]);r.include_reasoning=True
xml='<tool_call>exec'+''.join('<arg_key>'+k+'</arg_key><arg_value>'+str(v)+'</arg_value>' for k,v in a.items())+'</tool_call>'
for label,s in [('valid_implicit',xml),('valid_explicit','</think>'+xml),('missing_arg_end',xml.replace('</arg_value></tool_call>','</tool_call>'))]:
 ids=t.encode(s,add_special_tokens=False);fail=0;matches=0
 for width in range(1,len(ids)+1):
  p=Glm47MoeParser(t,tools=r.tools);out=''
  for j in range(0,len(ids),width):
   ix=ids[j:j+width]; text=t.decode(ix,skip_special_tokens=False)
   for dm in p.parse_deltas(text,ix,r,finished=j+width>=len(ids)):
    for tc in dm.tool_calls or []:out+=(tc.function and tc.function.arguments) or ''
  try:ok=json.loads(out)==a
  except:ok=False
  fail+=not ok;matches+=out==payload['arguments']
 print(label,'token_count',len(ids),'chunkings',len(ids),'failures',fail,'incident_exact_matches',matches)
print('cuda',torch.cuda.is_initialized())
