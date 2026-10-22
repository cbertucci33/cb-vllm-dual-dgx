import json,datetime,time,urllib.request,subprocess,re,hashlib
from pathlib import Path
P=Path(__file__).parent
UTC=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def run(c): return subprocess.check_output(c,text=True)
def get(url):return urllib.request.urlopen(url,timeout=15).read().decode()
def metrics(raw):
 d={}
 for l in raw.splitlines():
  if l and not l.startswith('#'):
   k,v=l.rsplit(' ',1);d[k]=float(v)
 return d
old=Path('/opt/project/evidence/glm-v31-candidate-build-deploy-20261015/warm-measurement-20261016/request.json')
raw=old.read_bytes();request=json.loads(raw)
assert request['max_output_tokens']==1024 and request['reasoning']['effort']=='high' and not any(k in request for k in ['seed','temperature','top_p'])
(P/'request.json').write_bytes(raw)
before=get('http://127.0.0.1:8320/metrics');(P/'baseline-before.metrics').write_text(before); b=metrics(before)
assert all(v==0 for k,v in b.items() if k.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{')))

startutc=UTC();trace='glm32-warm-20261021-baseline';t=time.monotonic();first=None;terminal=None;evtimes=[]
req=urllib.request.Request('http://127.0.0.1:8000/v1/responses',data=raw,headers={'Content-Type':'application/json','X-Flight-Trace-Id':trace})
print(json.dumps({'milestone':'launched','utc':startutc,'trace':trace}),flush=True)
with urllib.request.urlopen(req,timeout=180) as response, (P/'baseline.sse').open('w') as out:
 for bs in response:
  line=bs.decode();out.write(line)
  if line.startswith('data:'):
   s=line[5:].strip()
   if s=='[DONE]':continue
   e=json.loads(s);et=time.monotonic()-t;evtimes.append({'t':et,'type':e.get('type')})
   if first is None and e.get('type') in ('response.output_text.delta','response.reasoning_text.delta','response.reasoning_summary_text.delta') and e.get('delta'):first=et
   if e.get('type') in ('response.completed','response.incomplete','response.failed'):terminal=e['response']
elapsed=time.monotonic()-t
(P/'baseline-event-times.json').write_text(json.dumps(evtimes))
after=get('http://127.0.0.1:8320/metrics');(P/'baseline-after.metrics').write_text(after);a=metrics(after);delta={k:a[k]-v for k,v in b.items() if k in a and a[k]!=v and not k.endswith('_created')};(P/'baseline-metric-deltas.json').write_text(json.dumps(delta,indent=2))
def val(n):return sum(v for k,v in delta.items() if k.split('{')[0]=='vllm:'+n)
usage=terminal.get('usage',{}) if terminal else {};outn=usage.get('output_tokens',0);rounds=val('spec_decode_num_drafts_total');accepted=val('spec_decode_num_accepted_tokens_total');drafted=val('spec_decode_num_draft_tokens_total');decode=val('request_decode_time_seconds_sum')
r={'started':startutc,'finished':UTC(),'trace_id':trace,'http':response.status,'e2e_s':elapsed,'first_stream_text_s':first,'usage':usage,'status':terminal.get('status') if terminal else None,'request_count':val('request_generation_tokens_count'),'backend_ttft_s':val('time_to_first_token_seconds_sum'),'backend_decode_s':decode,'backend_prefill_s':val('request_prefill_time_seconds_sum'),'decode_tps_excluding_first':(outn-1)/decode if decode else None,'decode_tps_including_first':outn/decode if decode else None,'rounds':rounds,'drafted':drafted,'accepted':accepted,'accepted_per_round':accepted/rounds if rounds else None,'mean_acceptance_length_including_bonus':1+accepted/rounds if rounds else None,'token_acceptance_fraction':accepted/drafted if drafted else None,'per_position_accepted':{re.search('position="(\d+)"',k).group(1):v for k,v in delta.items() if k.startswith('vllm:spec_decode_num_accepted_tokens_per_pos_total')},'isolated':val('request_generation_tokens_count')==1,'preemptions':val('request_num_preemptions_sum')}
(P/'baseline.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)
