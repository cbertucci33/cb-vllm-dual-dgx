# Release history

## Release 6

Release 6 renames the project to `cb-vllm-dual-dgx` and moves its active
development line from vLLM 0.29 to vLLM 0.30. The old vLLM 0.29 source remains
available on the `release/v0.29` branch.

### Runtime platform

- Rebased the active runner on vLLM 0.30 and retained the two-node DGX Spark
  build and deployment contract.
- Updated the GLM runtime for the current upstream interfaces, including the
  DFlash draft boundary, resolved draft-cache layout, auxiliary hidden states,
  B12X MXFP8 loading, FP8 context KV rows, and GLM chat-template handling.
- Retained asynchronous scheduling and DFlash support while restoring the
  recurrent-state publication and replay boundary rules needed by hybrid
  models.
- Added explicit warmup coverage for request-time sampler, routing, compressed
  KV, fused projection, DFlash, and first-use kernel paths.
- Packaged the pinned FlashInfer and FlashKDA integration, including sparse MLA
  metadata support and current vLLM API compatibility.

### DGX Spark correctness and performance

- Kept the GB10-specific route-capacity and SM121 build checks.
- Restored deterministic GB10 TopK behavior and the current GLM K-pool path.
- Reduced host dispatch in DFlash and DSpark metadata preparation, draft-input
  padding, and Mamba alignment work.
- Added warmup and runtime guards so unsupported paths fail clearly instead of
  silently selecting an unqualified fallback.

### Generic rank-sliced EXL3 support

- Ported the rank-sliced EXL3 backend to vLLM 0.30.
- Added generic rank-sliced name normalization in the model weight loader.
- Hydrated dense EXL3 storage metadata alongside rank-sliced expert metadata.
- Detached scalar EXL3 markers and dense EXL3 tensors from safetensor-backed
  source storage during load.
- Preserved rank-local expert slabs while allowing compatible tensor-parallel
  MoE checkpoints to use the same loader path.

### Upstream work carried forward

Release 6 retains and extends the earlier downstream work around these vLLM
changes and related fixes:

- [#54163](https://github.com/vllm-project/vllm/pull/54163), DFlash and DSpark
  cache-tail semantics.
- [#55122](https://github.com/vllm-project/vllm/pull/55122), deterministic
  persistent TopK behavior.
- [#56196](https://github.com/vllm-project/vllm/pull/56196), convolution-state
  ownership at short prefill boundaries.
- [#56794](https://github.com/vllm-project/vllm/pull/56794), transient
  checkpoint protection.
- [#56960](https://github.com/vllm-project/vllm/pull/56960), GLM KDA
  checkpoint handling.
- [#57161](https://github.com/vllm-project/vllm/pull/57161), GLM K-pool
  numerical paths.
- [#57477](https://github.com/vllm-project/vllm/pull/57477), padded GLM K-pool
  tail indexing.
- [#57605](https://github.com/vllm-project/vllm/pull/57605), hybrid Mamba
  allocation and checkpoint boundaries.
- [#58450](https://github.com/vllm-project/vllm/pull/58450) and
  [#58762](https://github.com/vllm-project/vllm/pull/58762), current upstream
  GLM and metadata fixes.

### Upgrade notes

- Use the `release-6` tag for the Release 6 source snapshot.
- Use the same image and source revision on both tensor-parallel ranks.
- Validate each model with its own tokenizer or chat template and a focused
  generation check. A successful distributed load does not validate a
  model-specific template or checkpoint.
- The old vLLM 0.29 source remains on `release/v0.29` for maintenance only.

## Releases 1 through 5

Releases 1 through 5 remain available as tags on the vLLM 0.29 maintenance
line. They cover the original GLM EXL3 runner, DFlash2 deployment, cache and
recurrent-state correctness, GB10 TopK determinism, K-pool numerical work,
and reproducible source and model acquisition.
