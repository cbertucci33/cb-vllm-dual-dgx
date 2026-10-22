from pathlib import Path
import subprocess
r=Path('/opt/project/builds/glm-mimo-main-fixes-20261017')
paths=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=r,text=True).splitlines()
def owns(p):
 return not (p.startswith(('csrc/','vllm/parser/','vllm/tool_parsers/','vllm/reasoning/','vllm/tokenizers/','vllm/entrypoints/','tests/parser/','tests/reasoning/','tests/tool_parsers/','tests/tool_use/','tests/entrypoints/','vllm/v1/structured_output/','tests/v1/structured_output/','build/patches/','tests/kernels/moe/')) or p in {'CMakeLists.txt','vllm/_custom_ops.py','vllm/v1/engine/detokenizer.py','tests/kernels/test_top_k_per_row.py','tests/kernels/attention/test_flashinfer_memory_ordering.py','tests/kernels/mamba/test_causal_conv1d_acceptance_window.py'})
owned=[p for p in paths if owns(p) and (r/p).exists()]
e=Path('/opt/project/evidence/vllm-v032-migration-20261018/source');(e/'owned-paths.txt').write_text('\n'.join(owned)+'\n');(e/'owned-python.txt').write_text('\n'.join(p for p in owned if p.endswith('.py'))+'\n')
print('owned',len(owned),'python',sum(p.endswith('.py') for p in owned))
