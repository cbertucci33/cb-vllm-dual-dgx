import difflib
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path('/opt/project/builds/glm-mimo-main-fixes-20261017')
OUT=Path(__file__).parent
paths=json.loads((OUT/'owned-paths.json').read_text())
def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)
manifest=[]
delta=[]
for name in paths:
    before=(OUT/'before'/name).read_bytes()
    after=(ROOT/name).read_bytes()
    manifest.append({'path':name,'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(after).hexdigest(),'changed':before!=after})
    delta.extend(difflib.unified_diff(before.decode().splitlines(True),after.decode().splitlines(True),fromfile='before/'+name,tofile='after/'+name))
(OUT/'port-only.diff').write_text(''.join(delta))
(OUT/'owned-staged.binary.diff').write_bytes(git('diff','--cached','--binary','HEAD','--',*paths))
status=git('status','--porcelain=v1').decode()
report={'head':git('rev-parse','HEAD').decode().strip(),'branch':git('branch','--show-current').decode().strip(),'staged_tree':git('write-tree').decode().strip(),'owned_paths':manifest,'owned_unstaged':git('diff','--name-only','--',*paths).decode().splitlines(),'unmerged':git('diff','--name-only','--diff-filter=U').decode().splitlines(),'global_status':status.splitlines(),'hooks':'7 affected cached hooks passed; full mandatory hook/dependency gates remain as prior BUILD-HANDOFF, not qualified by this lane.'}
(OUT/'final-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('owned_paths','global_status')},indent=2))
