exec(open('/opt/project/evidence/vllm-v032-migration-20261018/source/owned_paths.py').read().split('owned=[p for p')[0])
paths=subprocess.check_output(['git','diff','--name-only'],cwd=r,text=True).splitlines()
files=[p for p in paths if owns(p) and not p.startswith('build/tests/')]
for p in files:
 s=(r/p).read_text() if (r/p).is_file() else ''
 assert '<<<<<<< ours' not in s,p
subprocess.run(['git','add','--',*files],cwd=r,check=True)
print('staged',len(files),'ownedworkingtreepaths')
