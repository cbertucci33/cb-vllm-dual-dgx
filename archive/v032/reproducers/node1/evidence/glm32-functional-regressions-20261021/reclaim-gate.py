import subprocess,json,datetime
from pathlib import Path
def mem():return {x.split(":")[0]:int(x.split()[1]) for x in Path("/proc/meminfo").read_text().splitlines() if x.split(":")[0] in ("MemAvailable","MemFree","Cached","SwapFree")}
running=subprocess.check_output(["docker","ps","-q"],text=True).strip();assert not running,running
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"],text=True).strip();assert not apps,apps
u=subprocess.run(["sudo","-n","fuser","/dev/nvidia-uvm"],capture_output=True,text=True);assert u.returncode==1 and not u.stdout.strip(),(u.returncode,u.stdout,u.stderr)
ps=subprocess.check_output(["ps","-eo","pid,comm,args"],text=True)
models=[x for x in ps.splitlines() if ("vllm::" in x or "VLLM::" in x or "vllm.entrypoints.openai.api_server" in x) and x.split()[1] not in ("bash","ssh","python3")];assert not models,models
before=mem();action="baseline already met; no cache drop"
if before["MemAvailable"]<122385528:
 subprocess.check_call(["sudo","-n","sh","-c","sync; echo 3 > /proc/sys/vm/drop_caches"]);action="approved one-shot sync/drop_caches; no UVM reset"
after=mem();assert after["MemAvailable"]>=122385528,after
print(json.dumps({"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"node":subprocess.check_output(["hostname"],text=True).strip(),"before_KiB":before,"after_KiB":after,"accepted_baseline_KiB":122385528,"action":action,"no_running_containers":True,"no_compute_apps":True,"no_UVM_owner":True,"no_model_processes":True,"memory_PSI":Path("/proc/pressure/memory").read_text(),"passed":True}))
