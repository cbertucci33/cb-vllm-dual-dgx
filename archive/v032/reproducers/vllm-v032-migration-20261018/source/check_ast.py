from pathlib import Path
import subprocess,ast,json
r=Path('/opt/project/builds/glm-mimo-main-fixes-20261017');rows=[]
files=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=r,text=True).splitlines()
for f in files:
 p=r/f
 if p.suffix!='.py' or not p.exists():continue
 s=p.read_text()
 if '<<<<<<< ours' in s:rows.append({'path':f,'status':'unmerged'});continue
 try:
  tree=ast.parse(s,filename=f)
  dup=[]
  for n in ast.walk(tree):
   if isinstance(n,(ast.Module,ast.ClassDef)):
    names=[x.name for x in n.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
    dup+=sorted({x for x in names if names.count(x)>1})
  rows.append({'path':f,'status':'pass','duplicate_definitions':dup})
 except SyntaxError as e:rows.append({'path':f,'status':'syntax_error','line':e.lineno,'error':e.msg})
(r.parent.parent/'evidence/vllm-v032-migration-20261018/source/ast-results.json').write_text(json.dumps(rows,indent=2))
for row in rows:
 if row['status']!='pass' or row.get('duplicate_definitions'): print(row)
print('total',len(rows),'pass',sum(x['status']=='pass' for x in rows))
