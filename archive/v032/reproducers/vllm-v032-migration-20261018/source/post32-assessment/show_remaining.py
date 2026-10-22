from pathlib import Path
import json
p=Path('/opt/project/evidence/vllm-v032-migration-20261018/source/post32-assessment')
for n in [62180,61996]:
 for f in json.loads((p/f'files-{n}.json').read_text()):
  if n==62180 or f['filename'].endswith('states.py'):
   print('PR',n,f['filename']);print(f.get('patch','')[:14000])
