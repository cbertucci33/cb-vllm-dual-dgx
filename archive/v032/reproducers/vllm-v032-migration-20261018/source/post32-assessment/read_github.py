from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request,json,urllib.parse,time
out=Path('/opt/project/evidence/vllm-v032-migration-20261018/source/post32-assessment')
def get(url):
 req=urllib.request.Request(url,headers={'Accept':'application/vnd.github+json','User-Agent':'private-source-assessment'})
 try:
  with urllib.request.urlopen(req,timeout=30) as r:return json.load(r)
 except Exception as e:return {'error':str(e),'url':url}
nums=[61253,62027,59511,61455,55671,61810,62180]
def pr(n):
 d=get(f'https://api.github.com/repos/vllm-project/vllm/pulls/{n}');(out/f'pr-{n}.json').write_text(json.dumps(d,indent=2));return {k:d.get(k) for k in ['number','title','state','merged','merged_at','merge_commit_sha','updated_at','html_url']}
with ThreadPoolExecutor(max_workers=7) as p:
 for r in p.map(pr,nums):print(json.dumps(r))
queries={
 'glm-gb10-dflash-responses':'repo:vllm-project/vllm is:pr is:merged merged:>=2026-10-18 (GLM OR GB10 OR SM120 OR SM121 OR DFlash OR Responses)',
 'mimo':'repo:vllm-project/vllm is:pr is:merged merged:>=2026-10-18 MiMo',
}
for name,q in queries.items():
 d=get('https://api.github.com/search/issues?'+urllib.parse.urlencode({'q':q,'per_page':100,'sort':'updated','order':'desc'}));(out/f'search-{name}.json').write_text(json.dumps(d,indent=2));print('SEARCH',name,d.get('total_count'),d.get('error'));print(json.dumps([{k:x.get(k) for k in ['number','title','state','html_url','updated_at']} for x in d.get('items',[])]))
