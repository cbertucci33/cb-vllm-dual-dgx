from pathlib import Path
import json,subprocess
out=Path('/opt/project/evidence/vllm-v032-migration-20261018/source/post32-assessment');root=Path('/opt/project/builds/glm-mimo-main-fixes-20261017');rows=[]
for n in [61253,62027,59511,61455,55671,61810,62180,61996,61174,46399,61989,58857,62129,52477,61938,56549,62221]:
 fs=json.loads((out/f'files-{n}.json').read_text());files=[]
 print('\nPR',n)
 for f in fs:
  path=f['filename'];text=f.get('patch');rc=None
  if text and f['status']=='modified' and (root/path).exists():
   p=f'diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n{text}\n';res=subprocess.run(['git','apply','--reverse','--check','-'],input=p,cwd=root,capture_output=True,text=True);rc=res.returncode
  files.append({'path':path,'reverse_hunks_present':rc==0 if rc is not None else None,'status':f['status']})
  print(path,'EXACT_REVERSE',rc)
  if n in [61174,46399,61989,58857,62129,52477,61938,56549,62221] and path.startswith(('vllm/','cmake/','.dockerignore')):print((text or '')[:10000])
 rows.append({'number':n,'files':files})
(out/'staged-hunk-comparison.json').write_text(json.dumps(rows,indent=2))
