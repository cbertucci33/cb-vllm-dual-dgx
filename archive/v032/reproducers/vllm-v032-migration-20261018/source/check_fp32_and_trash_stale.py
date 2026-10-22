from pathlib import Path
from types import SimpleNamespace
import ast,json,hashlib,subprocess,shutil
r=Path('/opt/project/builds/glm-mimo-main-fixes-20261017');e=Path('/opt/project/evidence/vllm-v032-migration-20261018/source');n=e.parent/'native/fp32-workspace-contract'
t=ast.parse((r/'tests/quantization/test_exl3.py').read_text());test=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='test_exl3_gemm_retains_fp32_native_accumulation_and_fp16_public_output');scope={'exl3_module':SimpleNamespace(__file__=str(r/'vllm/model_executor/layers/quantization/exl3.py'))};exec(compile(ast.Module(body=[test],type_ignores=[]),'existing-production-AST-contract','exec'),scope);scope[test.name]()
locks=json.loads((n/'source-locks.json').read_text());assert locks['native_commit']=='c5d9c657966ffeeaa9353f0cc899f18629da4a13'
checked=[]
for name,sha in locks['source_hashes'].items():
 p=n/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,name;checked.append(name)
cu=(n/'exllamav3/exllamav3_ext/quant/exl3_gemm.cu').read_text();inner=(n/'exllamav3/exllamav3_ext/quant/exl3_gemm_inner.cuh').read_text();assert 'bool c_fp32 = C.dtype() == at::kFloat;' in cu;assert 'if (!c_fp32) TORCH_CHECK_DTYPE(C, kHalf);' in cu
assert 'auto read_sum_gl' in inner and 'auto write_sum_gl' in inner and 'float* c_ptr = gl_c_ptr_32' in inner and '__floats2half2_rn' in inner
receipt={'production_ast_test':test.name,'status':'pass','native_commit':locks['native_commit'],'native_hashes_verified':checked,'native_dtype_contract':'C accepts kFloat or kHalf; dtype selects typed global split-K C reads/writes. Exact line-backed report ../native/fp32-workspace-contract/RESULT.md.','output_memory':{'M':'x.shape[0]','N':'trellis.shape[1] * 16','old_C_bytes':'2*M*N','new_C_bytes':'4*M*N','cast_result_bytes':'2*M*N','possible_C_plus_cast_bytes':'6*M*N','delta_at_cast_vs_old_C_bytes':'4*M*N'},'unverified':['real native numeric behavior','model logits/quality','performance','graph lifetime','live peak fit'],'no_fake_kernel_execution':True,'no_Torch_or_CUDA_import':True}
(e/'fp32-workspace-contract.json').write_text(json.dumps(receipt,indent=2));print('PASS production AST allocation/native positional binding/final cast +13 SHA-locked native inputs')
p=r/'build/provenance.json';raw=p.read_bytes();data=json.loads(raw);assert data['source']['vllm_version']=='0.31.0' and data['versions']['VLLM_VERSION']=='0.31.0'
ignored=subprocess.run(['git','check-ignore','--quiet',str(p)],cwd=r).returncode;assert ignored==0
owners=subprocess.run(['fuser',str(p)],capture_output=True,text=True);assert owners.returncode==1 and not owners.stderr.strip(),owners
ids=subprocess.check_output(['docker','ps','-q'],text=True).split();mounts=[]
if ids:
 for id in ids:
  mounts.extend(json.loads(subprocess.check_output(['docker','inspect','--format','{{json .Mounts}}',id],text=True)))
for mount in mounts:
 s=Path(mount['Source'])
 assert not(p==s or p.is_relative_to(s)),mount
backup=e/'stale-v031-provenance.json';backup.write_bytes(raw);sha=hashlib.sha256(raw).hexdigest()
subprocess.run(['gio','trash',str(p)],check=True);assert not p.exists()
receipt={'path':str(p),'sha256':sha,'captured':str(backup),'source_version':data['source']['vllm_version'],'git_ignored':True,'open_file_owners':False,'active_container_mount_consumer':False,'operation':'gio trash','original_absent':True,'regeneration':'build/build_image.sh runs provenance.py in fresh standalone committed source context before Docker builds.'};(e/'stale-provenance-cleanup.json').write_text(json.dumps(receipt,indent=2));print('TRASHED exact ignored stale .31 provenance after capture/no-owner/no-mount checks')
