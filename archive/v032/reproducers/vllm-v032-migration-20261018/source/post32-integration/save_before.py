from pathlib import Path
import re,json
r=Path('/opt/project/builds/glm-mimo-main-fixes-20261017');e=Path('/opt/project/evidence/vllm-v032-migration-20261018/source/post32-integration');files=[]
for n in [61174,52477]:
 f=next(e.glob(f'{n}-*.diff'));s=f.read_text();paths=re.findall(r'^diff --git a/(.*?) b/',s,re.M)
 for name in paths:
  p=e/'before'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((r/name).read_bytes());files.append(name)
(e/'owned-paths.json').write_text(json.dumps(files,indent=2))
