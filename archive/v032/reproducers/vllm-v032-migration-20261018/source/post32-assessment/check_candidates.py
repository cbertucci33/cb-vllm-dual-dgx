exec(open('/opt/project/evidence/vllm-v032-migration-20261018/source/post32-assessment/read_github.py').read().split('nums=')[0])
import subprocess
repo='/opt/project/builds/glm-mimo-main-fixes-20261017';pin='5df63a7203cdb9328d824a27b682e77c7b9928be'
nums=[61253,62027,59511,61455,55671,61810,62180,61996,61174,46399,61989,58857,62129]
def check(n):
 p=out/f'pr-{n}.json';d=json.loads(p.read_text()) if p.exists() else get(f'https://api.github.com/repos/vllm-project/vllm/pulls/{n}');p.write_text(json.dumps(d,indent=2))
 merged=bool(d.get('merged'));sha=d.get('merge_commit_sha') if merged else d.get('head',{}).get('sha');inc=False;proof='unmerged; no accepted merge commit'
 if merged and sha:
  exists=subprocess.run(['git','cat-file','-e',sha],cwd=repo,capture_output=True).returncode==0
  if exists:inc=subprocess.run(['git','merge-base','--is-ancestor',sha,pin],cwd=repo).returncode==0;proof='local exact object ancestry'
  else:
   c=get(f'https://api.github.com/repos/vllm-project/vllm/compare/{sha}...{pin}?per_page=1');(out/f'ancestry-{n}.json').write_text(json.dumps(c,indent=2));inc=c.get('merge_base_commit',{}).get('sha')==sha;proof='GitHub compare merge_base_commit '+c.get('merge_base_commit',{}).get('sha','unverified')
 f=get(f'https://api.github.com/repos/vllm-project/vllm/pulls/{n}/files?per_page=100');(out/f'files-{n}.json').write_text(json.dumps(f,indent=2))
 row={'number':n,'title':d.get('title'),'merged':merged,'state':d.get('state'),'merged_at':d.get('merged_at'),'commit':sha,'in_release':inc,'ancestry_evidence':proof,'head_sha':d.get('head',{}).get('sha'),'files':[x.get('filename') for x in f] if isinstance(f,list) else f}
 return row
with ThreadPoolExecutor(max_workers=6) as ex:rows=list(ex.map(check,nums))
(out/'candidate-status.json').write_text(json.dumps(rows,indent=2))
for row in rows:print(json.dumps(row))
print('SOURCE',subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),subprocess.check_output(['git','write-tree'],cwd=repo,text=True).strip())
