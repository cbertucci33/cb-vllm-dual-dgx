from pathlib import Path
import urllib.request,hashlib,ast,json
p=Path('/opt/project/evidence/vllm-v032-migration-20261018/source/torch-2.13.0-accelerator.py');url='https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/torch/accelerator/__init__.py';data=urllib.request.urlopen(url,timeout=30).read();p.write_bytes(data)
t=ast.parse(data);functions={x.name:x for x in t.body if isinstance(x,ast.FunctionDef)}
assert 'current_device_index' in functions and 'synchronize' in functions
for n in ['current_device_index','synchronize']:
 print(ast.get_source_segment(data.decode(),functions[n]))
(p.with_suffix('.json')).write_text(json.dumps({'source':url,'sha256':hashlib.sha256(data).hexdigest(),'functions':['current_device_index','synchronize'],'status':'source-contract verified, not executed'},indent=2))
