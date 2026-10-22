import json,sys,torch
from unittest.mock import MagicMock
from vllm.parser.glm47_moe import Glm47MoeParser
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.entrypoints.openai.responses.streaming_events import emit_function_call_done_events,StreamingState
payload=json.loads(sys.stdin.readline())
args=json.loads(payload['arguments']+'"}')
tool=payload['tool']
req=ResponsesRequest(model='model-main',input=[],tools=[tool],max_output_tokens=40000)
req.include_reasoning=True
vocab={'<think>':1000,'</think>':1001,'<tool_call>':1002,'</tool_call>':1003}
def parser():
 t=MagicMock();t.get_vocab.return_value=vocab;t.encode.return_value=[1,2,3];t.all_special_tokens=list(vocab);t.all_special_ids=list(vocab.values());t.decode.side_effect=lambda ids:''.join({v:k for k,v in vocab.items()}.get(i,'x') for i in ids)
 return Glm47MoeParser(t,tools=[tool])
raw='</think><tool_call>exec'+''.join('<arg_key>'+k+'</arg_key><arg_value>'+str(v)+'</arg_value>' for k,v in args.items())+'</tool_call>'
def run(text,size):
 p=parser();out=''
 for j in range(0,len(text),size):
  d=p.parse_delta(text[j:j+size],[],req,finished=j+size>=len(text))
  for tc in (d and d.tool_calls) or []:
   out+=(tc.function and tc.function.arguments) or ''
 try:parsed=json.loads(out);ok=parsed==args
 except:ok=False
 return out,ok,p
cases=[]
for label,text in [('valid',raw),('missing_tool_end',raw[:-12]),('missing_value_end',raw.replace('</arg_value></tool_call>','</tool_call>')),('truncated',raw[:raw.rfind('</arg_value>')])]:
 failures=[]
 for size in range(1,len(raw)+1):
  out,ok,p=run(text,size)
  if not ok:failures.append(size)
 print(label,'failures',len(failures),'of',len(raw),'first',failures[:10])
 out,ok,p=run(text,7)
 print('result',repr(out),'matches_incident',out==payload['arguments'])
 if p._tool_slots:print('raw_slot',repr(p._tool_slots[0].args),'streamed',repr(p._tool_slots[0].streamed_json))
print('cuda_initialized',torch.cuda.is_initialized())
from vllm.entrypoints.openai.responses.streaming_events import SimpleStreamingEventProcessor,split_delta,_StateType
for label,text in [('valid',raw),('missing_value_end',raw.replace('</arg_value></tool_call>','</tool_call>')),('truncated',raw[:raw.rfind('</arg_value>')])]:
 p=parser(); proc=SimpleStreamingEventProcessor(tools=req.tools); ev=[]
 for j in range(0,len(text),7):
  for dm0 in p.parse_deltas(text[j:j+7],[],req,finished=j+7>=len(text)):
   for dm in split_delta(dm0):
    state,tc=proc.resolve_target_state(dm)
    if state==_StateType.NONE:continue
    if proc.needs_transition(state,tc):ev+=proc.close_current();ev+=proc.open(state,tc)
    ev+=proc.emit_delta(dm,None,None)
 ev+=proc.close_current()
 items=[e.item for e in ev if e.type=='response.output_item.done' and e.item.type=='function_call']
 print('actual_responses',label,[(i.status,i.arguments==payload['arguments']) for i in items])
print('cuda_final',torch.cuda.is_initialized())
from vllm.parser.parser_manager import ParserManager
factory=ParserManager.get_parser(tool_parser_name='glm47',reasoning_parser_name='glm45',enable_auto_tools=True)
p=factory(parser().model_tokenizer,req.tools)
req=p.adjust_request(req)
print('adjusted_structured_outputs',req.structured_outputs)
print('strict',[(t.name,t.strict) for t in req.tools])
