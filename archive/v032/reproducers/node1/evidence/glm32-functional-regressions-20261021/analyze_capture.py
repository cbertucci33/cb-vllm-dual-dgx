import torch,json,pathlib
from transformers import AutoTokenizer
root=pathlib.Path('/glm-diagnostics')
tok=AutoTokenizer.from_pretrained('/models/glm53-exl3',trust_remote_code=True)
reports=[]
for path in sorted(root.glob('request-step-*.pt')):
 d=torch.load(path,map_location='cpu',weights_only=True)
 p=d['logits'].float(); q=d.get('draft_logits')
 vals,ids=p.topk(8,dim=-1)
 r={k:d[k].tolist() if isinstance(d[k],torch.Tensor) else d[k] for k in d if k not in ['logits','draft_logits']}
 r.update(file=path.name,p_finite=torch.isfinite(p).all().item(),p_min=p.min().item(),p_max=p.max().item(),p_top_ids=ids.tolist(),p_top_text=[[tok.decode([int(i)])for i in row]for row in ids],p_top_values=vals.tolist(),sampled_text=tok.decode(d['sampled'].flatten()[(d['sampled'].flatten()>=0)&(d['sampled'].flatten()<154880)].tolist()),prompt_text=tok.decode(d['input_ids'].tolist()))
 if q is not None:
  q=q.float();r['q_shape']=list(q.shape);r['q_nan_count']=torch.isnan(q).sum().item();r['q_finite_counts']=q.isfinite().sum(-1).tolist()
  slot=int(d['idx_mapping'][0]);qq=q[slot];qvals,qids=qq.topk(8,-1)
  r['q_top_text']=[[tok.decode([int(i)])for i in row]for row in qids];r['q_top_values']=qvals.tolist()
  overlap=[]
  for step in range(min(len(p)-1,len(qq))):
   pp=p[step].softmax(-1); vv,ii=pp.sort(descending=True);keep=(vv.cumsum(-1)-vv)<.95; mask=torch.zeros_like(pp,dtype=torch.bool).scatter_(0,ii,keep);pp=pp*mask;pp/=pp.sum()
   qp=qq[step].softmax(-1);overlap.append(torch.minimum(pp,qp).sum().item())
  r['p_q_overlap']=overlap
 reports.append(r)
(root/'distribution-summary.json').write_text(json.dumps(reports,indent=2))
for r in reports: print(json.dumps({k:r[k] for k in ['file','p_finite','p_min','p_max','sampled_text','q_nan_count','q_finite_counts','p_q_overlap']if k in r}),flush=True)
