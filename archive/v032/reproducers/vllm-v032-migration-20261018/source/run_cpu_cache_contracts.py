from pathlib import Path
from types import SimpleNamespace
import ast,json,tempfile
r=Path('/opt/project/builds/glm-mimo-main-fixes-20261017');e=Path('/opt/project/evidence/vllm-v032-migration-20261018/source')
scope={'ast':ast,'json':json,'Path':Path}
t=ast.parse((r/'vllm/model_executor/warmup/kernel_warmup.py').read_text());names={'_ast_int','_normalize_flashinfer_cache_key','_autotune_cache_fingerprint','_FLASHINFER_CACHE_RESERVED_KEYS','_FLASHINFER_UNIFIED_MOE_RUNNERS','_FlashInferMoEContext'}
body=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id in names for a in n.targets)]
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*body],type_ignores=[]);exec(compile(ast.fix_missing_locations(module),'production-cache-contract','exec'),scope)
testscope={'ast':ast,'json':json,'Path':Path,'warmup':SimpleNamespace(**scope)}
t=ast.parse((r/'tests/model_executor/test_flashinfer_autotune_warmup.py').read_text());tests=['test_rank_local_moe_keys_normalize_only_for_the_live_rank','test_rank_local_moe_cache_detects_same_count_different_bucket'];body=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name in {*tests,'_cache_key'}]
exec(compile(ast.Module(body=body,type_ignores=[]),'existing-cache-regressions','exec'),testscope)
for name in tests:
 with tempfile.TemporaryDirectory(dir=e/'tmp') as d:testscope[name](Path(d))
 print('PASS',name)
(e/'cpu-cache-contracts.json').write_text(json.dumps({'passed':tests,'method':'Execute unmodified existing regression functions and exact production definitions AST-isolated from heavy module imports; no mock cache behavior, Torch, CUDA, or installed dependency equivalence claimed.'},indent=2))
