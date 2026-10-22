from pathlib import Path
import ast,json
d=Path('/opt/project/evidence/glm-async-grammar-parser-20261021')
s=Path('/opt/project/evidence/glm30-32-tool-generation-diff-20261021/manager_probe.py').read_text()
tree=ast.parse(s)
incident=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='incident' for x in n.targets))
# Private captured fixture excluded; supply a local synthetic fixture.
payload = {}
old={}
for p in (d/'old').rglob('*.py'):
 rel=p.relative_to(d/'old').with_suffix('')
 mod='.'.join(rel.parts)
 if mod.endswith('.__init__'):mod=mod[:-9]
 old[mod]=(p.read_text(),p.name=='__init__.py')
prefix="""import sys,importlib.abc,importlib.util
class Loader(importlib.abc.MetaPathFinder,importlib.abc.Loader):
 def find_spec(self,fullname,path=None,target=None):
  if fullname in sources:return importlib.util.spec_from_loader(fullname,self,is_package=sources[fullname][1])
 def create_module(self,spec):return None
 def exec_module(self,module):
  module.__file__='exact9707345/'+module.__name__.replace('.','/')+'.py'
  exec(compile(sources[module.__name__][0],module.__file__,'exec'),module.__dict__)
sys.meta_path.insert(0,Loader())
"""
body="""
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
"""
(d/'parser_current.py').write_text('payload='+repr(payload)+'\n'+body)
(d/'parser_old.py').write_text('sources='+repr(old)+'\n'+prefix+'\npayload='+repr(payload)+'\n'+body)
print(len(old))
