from pathlib import Path
import urllib.request,json,hashlib,tarfile,subprocess
r=Path('/opt/project/evidence/vllm-v032-migration-20261018/source/transformers-qualification');r.mkdir(exist_ok=True)
m=json.load(urllib.request.urlopen('https://pypi.org/pypi/transformers/5.17.0/json',timeout=30))
s=next(x for x in m['urls'] if x['packagetype']=='sdist');data=urllib.request.urlopen(s['url'],timeout=60).read();sha=hashlib.sha256(data).hexdigest();assert sha==s['digests']['sha256']
p=r/s['filename'];p.write_bytes(data)
with tarfile.open(p) as t:t.extractall(r,filter='data')
root=next(x for x in r.iterdir() if x.is_dir())
patch='/opt/project/builds/glm-mimo-main-fixes-20261017/build/patches/transformers-glm5next-temporal-budget.patch'
res=subprocess.run(['git','apply','--check',patch],cwd=root,capture_output=True,text=True)
receipt={'version':'5.17.0','url':s['url'],'sha256':sha,'archive_bytes':len(data),'patch':patch,'check_exit':res.returncode,'stdout':res.stdout,'stderr':res.stderr,'no_packages_installed':True}
(r/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));assert res.returncode==0
