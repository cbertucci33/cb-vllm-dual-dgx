"""CPU semantic oracle for exact emitted PTX; not compiler/device verification."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
r=Path("/opt/project/worktrees/glm32-clamp-native-20261021")
s=(r/"sparkinfer/moe/_shared/kernels/w4a16/kernel.py").read_text()
tree=ast.parse(s)
helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_clamp_swiglu_f32")
helper.decorator_list=[]
emitted=[]
class F32:
    def __init__(self,x):self.x=x.x if isinstance(x,F32) else np.asarray(x,dtype=np.float32)
    def ir_value(self,**kwargs):return self.x

def asm(t,args,text,constraints,**kwargs):
    assert constraints=="=f,f,f" and kwargs["has_side_effects"] is False
    a,b=args; emitted.append(text)
    if text=="min.NaN.f32 $0, $1, $2;":return np.minimum(a,b)
    assert text=="min.NaN.xorsign.abs.f32 $0, $1, $2;"
    return np.copysign(np.minimum(np.abs(a),np.abs(b)),a)
env=dict(cutlass=NS(Float32=F32),T=NS(f32=lambda:None),llvm=NS(inline_asm=asm,AsmDialect=NS(AD_ATT=None)))
exec(compile(ast.fix_missing_locations(ast.Module(body=[helper],type_ignores=[])),"native-clamp-helper","exec"),env)
# All BF16 encodings plus representative FP32 bit patterns and clamp neighbors.
bf=(np.arange(65536,dtype=np.uint32)<<16).view(np.float32)
rng=np.random.default_rng(8320)
rand=rng.integers(0,2**32,size=65536,dtype=np.uint32).view(np.float32)
count=0
with np.errstate(invalid="ignore"):
    for lim in (np.float32(10),np.float32(7),np.float32(1e-6),np.float32(1e6)):
        edge=np.array([lim,-lim,np.nextafter(lim,np.float32(np.inf)),np.nextafter(lim,np.float32(0)),np.nextafter(-lim,np.float32(-np.inf)),np.nextafter(-lim,np.float32(0)),0.,-0.,np.inf,-np.inf,np.nan],dtype=np.float32)
        x=np.concatenate((bf,rand,edge))
        expected_g=np.where(x>lim,lim,x)
        expected_u=np.where(x>lim,lim,x);expected_u=np.where(expected_u < -lim,-lim,expected_u)
        g,u=env["_clamp_swiglu_f32"](x,x,lim)
        for actual,expected in ((g.x,expected_g),(u.x,expected_u)):
            assert np.array_equal(np.isnan(actual),np.isnan(expected))
            finite=~np.isnan(expected)
            assert np.array_equal(actual[finite].view(np.uint32),expected[finite].view(np.uint32))
        count+=len(x)
assert set(emitted)=={"min.NaN.f32 $0, $1, $2;","min.NaN.xorsign.abs.f32 $0, $1, $2;"}
result=dict(gate_up_pairs=count,instructions_per_pair=2,nan_policy="propagated; payload canonicalization permitted by PTX, not bitwise NaN payload preservation",non_nan_outputs="bitwise equal including signed zero, infinities and subnormals",gpu_or_compiler_executed=False)
print(json.dumps(result));Path(__file__).with_suffix(".json").write_text(json.dumps(result,indent=2))
