import urllib.request,json,sys,time
from pathlib import Path
p=Path(__file__).parent
mode=sys.argv[1];base='http://127.0.0.1:8320' if mode=='direct' else 'http://127.0.0.1:8000';model='glm-5.3-flash' if mode=='direct' else 'model-main'
r=json.load(urllib.request.urlopen(base+'/v1/models',timeout=10));assert r['data'][0]['id']==model and r['data'][0]['max_model_len']==600000
(p/(mode+'-inventory.json')).write_text(json.dumps(r,indent=2))
body={'model':model,'input':'What is 17 plus 25? Answer with just the number.','reasoning':{'effort':'high'},'temperature':1.0,'top_p':0.95,'stream':False}
start=time.time()
with urllib.request.urlopen(urllib.request.Request(base+'/v1/responses',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=120) as res:r=json.load(res)
(p/(mode+'-response.json')).write_text(json.dumps(r,indent=2))
text=''.join(c.get('text','') for i in r.get('output',[]) if i.get('type')=='message' for c in i.get('content',[]))
assert r['status']=='completed' and text.strip()=='42',(r['status'],text)
receipt={'mode':mode,'status':r['status'],'text':text,'seconds':time.time()-start,'usage':r.get('usage'),'response_id':r.get('id')}
(p/(mode+'-receipt.json')).write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
