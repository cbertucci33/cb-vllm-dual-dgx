from pathlib import Path
r=Path('/opt/project/builds/glm-mimo-main-fixes-20261017')
changes={
 'tests/model_executor/test_flashinfer_autotune_warmup.py': [('def _make_runner(modules, *, max_tokens: int = 8192):','def _make_runner(modules, *, max_tokens: int = 8192, linear_backend: str = "auto"):')],
 'tests/models/glm5next/test_sparse_indexer_topk_dispatch.py':[('import pytest\n','from types import SimpleNamespace\n\nimport pytest\n'),('from vllm.config import VllmConfig, set_current_vllm_config','from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config')],
 'tests/v1/attention/test_sparse_mla_backends.py':[('    *,\n    qk_nope_head_dim: int = 128,\n    kv_lora_rank: int = 512,\n    qk_rope_head_dim: int = 64,\n    v_head_dim: int = 128,','    *,\n    kv_lora_rank: int = 512,')],
 'vllm/utils/flashinfer.py':[('from collections.abc import Callable','from collections.abc import Callable, Iterator\nfrom contextvars import ContextVar')],
 'vllm/v1/core/kv_cache_utils.py':[('TypeAlias, overload','TypeAlias, cast, overload')],
}
for f,ss in changes.items():
 p=r/f;s=p.read_text()
 for a,b in ss:
  assert a in s,(f,a);s=s.replace(a,b)
 p.write_text(s)
