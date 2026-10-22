from pathlib import Path
import urllib.request,json,hashlib,subprocess
p=Path('/opt/project/evidence/vllm-v032-migration-20261018/source/post32-integration');rows=[]
for n,sha in [(52477,'6540d5c8f8c73bb2dd6194e56ed002fc32a31837'),(61174,'61bb590877521e2f4516912297a40f593e222ee9')]:
 url=f'https://github.com/vllm-project/vllm/commit/{sha}.diff';data=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'private-source-integration'}),timeout=30).read();f=p/f'{n}-{sha}.diff';f.write_bytes(data);rows.append({'pr':n,'commit':sha,'url':url,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)});print(n,len(data))
(p/'inputs.json').write_text(json.dumps(rows,indent=2))
