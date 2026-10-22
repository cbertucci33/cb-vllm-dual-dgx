from pathlib import Path
import difflib

root = Path('/opt/project/evidence/glm32-compact-cache-repair-20261021/original/include/flashinfer/attention/sparse_mla_sm120')
out = Path('/opt/project/evidence/glm32-functional-regressions-20261021/prefill-zero-row-source')
patch = Path('/opt/project/worktrees/glm32-three-improvements-20261020/build/patches/flashinfer-sparse-invalid-kv.patch')
prefix = 'include/flashinfer/attention/sparse_mla_sm120/'

sources = {}
for rel in ('common/kv_cache_io.cuh', 'prefill_common.cuh', 'prefill_swapab_kernel.cuh'):
    original = (root / rel).read_text()
    modified = original
    if rel == 'common/kv_cache_io.cuh':
        modified = modified.replace('#include "../model/kv_cache_traits.cuh"', '#include "../model/kv_cache_traits.cuh"\n#include "zero_row.cuh"')
        assert modified != original
        old = '  idx = (idx >= 0) ? idx : 0;\n\n  const uint8_t* src;'
        new = '  static_assert(COPY_BYTES <= SPARSE_MLA_ZERO_ROW_BYTES);\n  const bool valid = idx >= 0;\n  idx = valid ? idx : 0;\n\n  const uint8_t* src;'
        assert modified.count(old) == 1
        modified = modified.replace(old, new)
        old = '  if constexpr (USE_L2_HINT)\n    cp_async_bulk_g2s_l2hint(dst + io_tid * SMEM_STRIDE, src, COPY_BYTES, mbar, cache_policy);'
        new = '  src = valid ? src : sparse_mla_zero_row;\n  if constexpr (USE_L2_HINT)\n    cp_async_bulk_g2s_l2hint(dst + io_tid * SMEM_STRIDE, src, COPY_BYTES, mbar, cache_policy);'
        assert modified.count(old) == 1
        modified = modified.replace(old, new)
        old = '  idx = (idx >= 0) ? idx : 0;\n\n  int block_idx = idx / pbs;'
        new = '  const bool valid = idx >= 0;\n  idx = valid ? idx : 0;\n\n  int block_idx = idx / pbs;'
        assert modified.count(old) == 1
        modified = modified.replace(old, new)
        old = '                       (size_t)local_idx * SCALE_BYTES;\n  *reinterpret_cast<uint64_t*>(scale_dst + io_tid * SCALE_BYTES) ='
        new = '                       (size_t)local_idx * SCALE_BYTES;\n  src = valid ? src : sparse_mla_zero_row;\n  *reinterpret_cast<uint64_t*>(scale_dst + io_tid * SCALE_BYTES) ='
        assert modified.count(old) == 1
        modified = modified.replace(old, new)
    elif rel == 'prefill_common.cuh':
        modified = modified.replace('#include "common/kv_cache_io.cuh"', '#include "common/kv_cache_io.cuh"\n#include "common/zero_row.cuh"')
        old = '  idx = (idx >= 0) ? idx : 0;\n  // Addressing mode follows the scale layout'
        new = '  static_assert(KV::KV_ROPE_GMEM_OFFSET + KV::D_ROPE * (int)sizeof(bf16) <= SPARSE_MLA_ZERO_ROW_BYTES);\n  const bool valid = idx >= 0;\n  idx = valid ? idx : 0;\n  // Addressing mode follows the scale layout'
        assert modified.count(old) == 1
        modified = modified.replace(old, new)
        old = '  if constexpr (!KV::SCALE_IN_KV_SMEM) {\n    const int bi = idx / PAGE_BLOCK_SIZE;\n    const int li = idx % PAGE_BLOCK_SIZE;\n    return kv_global + (size_t)bi * stride_kv_block + (size_t)li * IO::IO_STRIDE;\n  } else {\n    return kv_global + (size_t)idx * IO::IO_STRIDE;\n  }'
        new = '  const uint8_t* base;\n  if constexpr (!KV::SCALE_IN_KV_SMEM) {\n    const int bi = idx / PAGE_BLOCK_SIZE;\n    const int li = idx % PAGE_BLOCK_SIZE;\n    base = kv_global + (size_t)bi * stride_kv_block + (size_t)li * IO::IO_STRIDE;\n  } else {\n    base = kv_global + (size_t)idx * IO::IO_STRIDE;\n  }\n  return valid ? base : sparse_mla_zero_row;'
        assert modified.count(old) == 1
        modified = modified.replace(old, new)
    elif rel == 'prefill_swapab_kernel.cuh':
        old = '  const uint8_t* src = kv_ptr + (size_t)(idx >= 0 ? idx : 0) * STRIDE;'
        new = '  static_assert(STRIDE <= SPARSE_MLA_ZERO_ROW_BYTES);\n  const uint8_t* src = idx >= 0 ? kv_ptr + (size_t)idx * STRIDE : sparse_mla_zero_row;'
        assert modified.count(old) == 1
        modified = modified.replace(old, new)
    assert modified != original, rel
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    (out / rel).write_text(modified)
    sources[rel] = (original, modified)

addition = ''
for rel, (old, new) in sources.items():
    path = prefix + rel
    addition += f'diff --git a/{path} b/{path}\n'
    addition += ''.join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True), fromfile=f'a/{path}', tofile=f'b/{path}'))
existing = patch.read_text()
assert 'prefill_swapab_kernel.cuh' not in existing
patch.write_text(existing + addition)
print(f'added {len(addition)} bytes of prefill zero-row patch in 3 files')
