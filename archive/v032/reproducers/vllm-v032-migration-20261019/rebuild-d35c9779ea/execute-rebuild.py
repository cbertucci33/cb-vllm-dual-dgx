import json,os,subprocess,datetime
from pathlib import Path
p=Path(__file__).parent
c=json.loads((p/'build-command.json').read_text())
env=os.environ.copy(); env.update(c['environment']); env['PATH']='/opt/project/builds/vllm-v031-glm-compact-kv-20261010/.venv/bin:'+env['PATH']; env['BUILDKIT_PROGRESS']='plain'
r=subprocess.run(['bash',c['entrypoint'],*c['ordered_arguments']],cwd=c['cwd'],env=env)
(p/'build-exit.json').write_text(json.dumps({'exit_code':r.returncode,'finished':datetime.datetime.now(datetime.timezone.utc).isoformat()})+'\n')
raise SystemExit(r.returncode)
