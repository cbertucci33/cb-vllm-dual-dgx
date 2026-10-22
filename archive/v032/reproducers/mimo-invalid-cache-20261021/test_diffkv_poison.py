"""Prepared GPU regression; not executed during live GLM user testing.
Run against source 766e012cd1 in the qualified runner environment later.
Imports are deferred so syntax/fixture-plan validation does not initialize CUDA.
"""
import pytest

# name, q length, KV length, window, split segments, KV heads
CASES = [
    ("prefill_partial", 65, 97, 0, 0, 2),
    ("swa_prefill_recycled", 65, 259, 128, 0, 2),
    ("swa_verify_recycled", 8, 259, 128, 0, 2),
    ("full_decode_empty_segments", 1, 17, 0, 16, 2),
    ("full_verify_empty_segments", 8, 19, 0, 16, 2),
    ("full_verify_grouped", 8, 35, 0, 64, 2),
    ("swa_decode_3d_recycled", 1, 259, 128, 16, 2),
]

@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("mode", ["bf16", "fp8_cache", "fp8_query_cache"])
def test_unused_cache_and_scratch_poison_isolation(case, mode):
    import torch
    from vllm.v1.attention.ops import triton_unified_attention_diffkv as mod
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_diffkv as write_cache,
    )
    name, qlen, length, window, segments, kvheads = case
    device = "cuda"
    torch.manual_seed(4321)
    hqk, hv, block, heads = 192, 128, 16, kvheads * 16
    blocks = (length + block - 1) // block
    shape = (blocks + 2, block, kvheads, hqk + hv)
    fp8 = mode != "bf16"
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    ks = torch.tensor(0.3 if fp8 else 1.0, device=device)
    vs = torch.tensor(0.7 if fp8 else 1.0, device=device)
    qs = torch.tensor(0.2, device=device)
    q = torch.randn(qlen, heads, hqk, device=device, dtype=torch.bfloat16) * 0.1
    if mode == "fp8_query_cache":
        q = (q / qs).to(dtype)
    # Earliest position any query may attend. Never poison a valid row for
    # ANY query in a grouped program; that would be genuine bad model data.
    earliest = max(0, length - qlen - window + 1) if window else 0
    table = torch.arange(1, blocks + 1, device=device, dtype=torch.int32)[None, :]
    # Freed complete blocks alias reserved null block, not live owned blocks.
    table[:, :earliest // block] = 0
    slots = []
    for pos in range(earliest, length):
        slots.append((pos // block + 1) * block + pos % block)
    slot_map = torch.tensor(slots + [-1], device=device, dtype=torch.int64)
    keys = torch.randn(len(slots) + 1, kvheads, hqk, device=device, dtype=torch.bfloat16) * 0.1
    vals = torch.randn(len(slots) + 1, kvheads, hv, device=device, dtype=torch.bfloat16) * 0.1
    keys[-1].fill_(float("nan"))
    vals[-1].fill_(float("nan"))
    clean = torch.zeros(shape, device=device, dtype=dtype)
    poisoned = torch.full(shape, float("nan"), device=device, dtype=torch.float32).to(dtype)
    for cache in (clean, poisoned):
        write_cache(keys, vals, cache, slot_map, "fp8_e4m3" if fp8 else "auto", ks, vs)
    # Valid bytes remain identical after real writer execution, including FP8 scaling.
    for pos in range(earliest, length):
        b, offset = pos // block + 1, pos % block
        assert torch.equal(clean[b, offset].view(torch.uint8), poisoned[b, offset].view(torch.uint8))
    original_kernel = mod.kernel_unified_attention_diffkv
    launches = []
    class ObserveLaunch:
        def __getitem__(self, grid):
            fn = original_kernel[grid]
            def call(**kw):
                launches.append((kw["IS_3D"], kw["BLOCK_M"], kw["TILE_SIZE"]))
                return fn(**kw)
            return call
    mod.kernel_unified_attention_diffkv = ObserveLaunch()
    outputs = []
    try:
        for cache, fill in ((clean, 0.0), (poisoned, float("nan"))):
            nseg = segments or 16
            scratch = torch.full((qlen, heads, nseg, hv), fill, device=device)
            maxima = torch.full((qlen, heads, nseg), fill, device=device)
            sums = torch.full_like(maxima, fill)
            out = torch.full((qlen, heads, hv), float("nan"), device=device, dtype=torch.bfloat16)
            mod.unified_attention_diffkv(
                q=q, k=cache[..., :hqk], v=cache[..., hqk:], out=out,
                cu_seqlens_q=torch.tensor([0, qlen], device=device, dtype=torch.int32),
                seqused_k=torch.tensor([length], device=device, dtype=torch.int32),
                softmax_scale=hqk ** -0.5, causal=True,
                window_size=(window - 1, 0) if window else (-1, -1),
                block_table=table, softcap=0.0, max_seqlen_q=qlen,
                seq_threshold_3D=8 if segments else 0,
                num_par_softmax_segments=nseg,
                softmax_segm_output=scratch, softmax_segm_max=maxima,
                softmax_segm_expsum=sums,
                k_descale=ks if fp8 else None, v_descale=vs if fp8 else None,
                q_descale=qs if mode == "fp8_query_cache" else None,
            )
            assert torch.isfinite(out).all(), (name, mode)
            outputs.append(out)
    finally:
        mod.kernel_unified_attention_diffkv = original_kernel
    assert all(launch[0] == bool(segments) for launch in launches), launches
    if name == "full_verify_grouped":
        assert all(launch[1] > 16 for launch in launches), launches
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
