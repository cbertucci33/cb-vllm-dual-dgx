import subprocess,json,socket,datetime
from pathlib import Path
def cmd(s):return subprocess.check_output(s,shell=True,text=True).strip()
host=socket.gethostname(); suffix,peer=("2802","127.0.0.1") if host=="node-2" else ("2801","127.0.0.1")
d={"host":host,"timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat(),"image":cmd('docker image inspect --format "{{.Id}}" vllm-glm-mimo:v032-5faa509fb5-candidate'),"owners":cmd('nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits'),"containers":cmd('docker ps --format "{{.Names}}"'),"gid":Path('/sys/class/infiniband/rocep1s0f0/ports/1/gids/3').read_text().strip(),"type":Path('/sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/3').read_text().strip(),"state":Path('/sys/class/infiniband/rocep1s0f0/ports/1/state').read_text().strip(),"route":cmd('ip route get '+peer),"ports":cmd('ss -ltn'),"meminfo":Path('/proc/meminfo').read_text()}
d['mem_available_kib']=int(next(x.split()[1] for x in d['meminfo'].splitlines() if x.startswith('MemAvailable:')))
assert d['image']=='sha256:9b5eba4b2dee88a6dd9c509de825554d7818a625a266181fa42d46d8ac1d0eb3'
assert not d['owners'] and not d['containers']
assert d['mem_available_kib']>=122385528,d['mem_available_kib']
assert d['gid'].endswith(suffix) and d['type']=='RoCE v2' and 'ACTIVE' in d['state'] and 'enp1s0f0np0' in d['route']
assert all(':'+str(port)+' ' not in d['ports'] for port in [8320,29521,29525])
Path(__file__).with_name(host+'-preflight.json').write_text(json.dumps(d,indent=2))
print(host,'preflight passed',d['mem_available_kib'])
