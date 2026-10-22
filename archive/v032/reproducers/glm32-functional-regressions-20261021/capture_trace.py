import urllib.request,json,time,pathlib
root=pathlib.Path('/opt/project/evidence/glm32-functional-regressions-20261021')
request={'model':'glm-5.3-flash','input':'What is 17 plus 25? Answer with just the number.','reasoning':{'effort':'high'},'temperature':1.0,'top_p':.95,'stream':True,'max_output_tokens':16,'seed':123}
(root/'stream-request.json').write_text(json.dumps(request,indent=2))
with urllib.request.urlopen('http://127.0.0.1:8320/metrics',timeout=10) as r:(root/'metrics-before.txt').write_bytes(r.read())
t=time.time()
r=urllib.request.Request('http://127.0.0.1:8320/v1/responses',json.dumps(request).encode(),{'Content-Type':'application/json'})
with urllib.request.urlopen(r,timeout=120) as response, (root/'trace-response.sse').open('wb') as f:
 for line in response:
  f.write(line);f.flush()
print('stream_seconds',time.time()-t)
with urllib.request.urlopen('http://127.0.0.1:8320/metrics',timeout=10) as r:(root/'metrics-after.txt').write_bytes(r.read())
