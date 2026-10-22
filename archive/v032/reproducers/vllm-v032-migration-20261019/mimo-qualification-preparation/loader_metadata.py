import os,json,pathlib,struct
import torch
def forbidden(*a,**k):raise RuntimeError('CUDA forbidden')
torch.cuda._lazy_init=forbidden
import instanttensor._impl as it
out={'env':{k:v for k,v in os.environ.items() if k.startswith('INSTANTTENSOR')},'backends':{},'page_size':os.sysconf('SC_PAGE_SIZE')}
for b in [it.Backend.URING,it.Backend.AIO]:out['backends'][b.name]=it._C.backend_status(b.value)
out['selected']=it.select_backend(it.default_backend).name
out['io']={}
for world in [1,2]:
 depth=max(512//world,3); chunk=8*1024*1024
 out['io'][world]={'initial_depth':depth,'chunk':chunk,'device_required':it.required_buffer_size_for_io(chunk,depth,world),'host_staging':chunk*depth,'padding':3*(16+out['page_size']+out['page_size']*world)}
for kind in ['target','draft']:
 p=pathlib.Path('/models/'+kind); sizes=[]
 for f in sorted(p.glob('*.safetensors')):
  with f.open('rb') as s:n=struct.unpack('<Q',s.read(8))[0];h=json.loads(s.read(n))
  rows=sorted((t['data_offsets'][0],t['data_offsets'][1]-t['data_offsets'][0]) for k,t in h.items() if k!='__metadata__')
  sizes.extend(n for off,n in rows)
 out[kind]={'recommended_sorted_file_order':it.recommended_buffer_size_for_tensors(sizes),'count':len(sizes),'max_tensor':max(sizes)}
out['cuda_initialized']=torch.cuda.is_initialized()
print(json.dumps(out,indent=2))
