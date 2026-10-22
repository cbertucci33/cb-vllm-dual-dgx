import urllib.request,json,time,pathlib,sys
p=pathlib.Path(__file__).parent
url=sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:8320'
model=sys.argv[2] if len(sys.argv)>2 else 'glm-5.3-flash'
tag=sys.argv[3] if len(sys.argv)>3 else 'direct-final'
body={'model':model,'input':'What is 17 plus 25? Answer with just the number.','reasoning':{'effort':'high'},'temperature':1.,'top_p':.95,'stream':True,'seed':123}
(p/(tag+'-request.json')).write_text(json.dumps(body,indent=2))
start=time.monotonic(); text='';terminal=None;events=[]
request=urllib.request.Request(url+'/v1/responses',json.dumps(body).encode(),{'Content-Type':'application/json'})
with urllib.request.urlopen(request,timeout=120) as response,(p/(tag+'.sse')).open('wb') as f:
 for line in response:
  f.write(line);f.flush()
  if line.startswith(b'data: '):
   try:d=json.loads(line[6:])
   except ValueError:continue
   events.append(d)
   if d.get('type')=='response.output_text.delta':text+=d.get('delta','')
   if d.get('type') in ['response.completed','response.failed','response.incomplete']:terminal=d
receipt={'elapsed_seconds':time.monotonic()-start,'text':text,'terminal':terminal}
(p/(tag+'-receipt.json')).write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
assert terminal and terminal.get('type')=='response.completed',receipt
assert text.strip()=='42',receipt
