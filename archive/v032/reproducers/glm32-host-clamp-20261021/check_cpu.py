import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
r=Path("/opt/project/worktrees/glm32-host-clamp-20261021")
# Execute the actual changed expression and actual metadata-prefix statements,
# stopping before backend work. No vLLM package import or CUDA initialization.
def method(path, name):
    tree=ast.parse((r/path).read_text())
    return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
def run(nodes, env):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),"source-prefix","exec"),env)
propose=method("vllm/v1/worker/gpu/spec_decode/dflash/speculator.py","propose")
seq=next(n for n in propose.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="max_seq_len" for t in n.targets))
base=method("vllm/v1/worker/gpu/spec_decode/speculator.py","_build_attn_metadata")
start=next(i for i,n in enumerate(base.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="query_start_loc_cpu")
end=next(i for i,n in enumerate(base.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="block_tables")
uniform=method("vllm/v1/worker/gpu/spec_decode/speculator.py","_build_uniform_attn_metadata")
ret=uniform.body[-1]; assert isinstance(ret,ast.Return)
cases=0
for n in range(1,65):
    for pad in (0,1,7):
        vals=torch.arange(n+pad,dtype=torch.int32)*997
        if pad: vals[n:]=2**30
        env=dict(input_batch=NS(seq_lens_cpu_upper_bound=vals),num_reqs=n)
        run([seq],env)
        assert env["max_seq_len"]==int(vals[:n].max())
        for stride in (1,2,8,12):
            got={}
            self=NS(arange_np=np.arange(n+1,dtype=np.int32),_build_attn_metadata=lambda **kw:got.update(kw))
            env=dict(self=self,num_reqs=n,num_query_per_req=stride,batch_desc=None,seq_lens_cpu_upper_bound=vals,step=0,causal=True,dcp_local_seq_lens=None)
            run(uniform.body[:-1]+[ast.Expr(value=ret.value)],env)
            assert got["max_query_len"]==stride
            common=dict(torch=torch,num_reqs=n,num_reqs_padded=n+pad,query_start_loc_np=got["query_start_loc_np"])
            a=dict(common,max_query_len=None); b=dict(common,max_query_len=got["max_query_len"])
            run(base.body[start:end],a);run(base.body[start:end],b)
            assert a["max_query_len"]==b["max_query_len"]
            assert torch.equal(a["query_start_loc_cpu"],b["query_start_loc_cpu"])
            # Nonuniform callers still use the original reduction.
            q=np.cumsum(np.r_[0,np.arange(n)%12+1],dtype=np.int32)
            c=dict(common,query_start_loc_np=q,max_query_len=None);run(base.body[start:end],c)
            assert c["max_query_len"]==int(np.diff(q).max())
            cases+=1
assert not torch.cuda.is_initialized()
result={"uniform_and_nonuniform_cases":cases,"sequence_cases":64*3,"cuda_initialized":False,"torch":torch.__version__,"numpy":np.__version__,"scope":"actual AST expression and metadata prefix, not backend execution"}
print(json.dumps(result));Path(__file__).with_suffix(".json").write_text(json.dumps(result,indent=2))
