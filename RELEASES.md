# Release history

## Release 7

Release 7 broadens the two-node NVIDIA DGX Spark runner from the GLM-focused
Release 6 line to a qualified MiMo V2.6 and GLM-5.3 Flash platform. The tested
models are:

- [cbert33/MiMo-V2.6-Flash-MOPD-Heretic-Uncensored-EXL3-DGX-Sliced](https://huggingface.co/cbert33/MiMo-V2.6-Flash-MOPD-Heretic-Uncensored-EXL3-DGX-Sliced)
- [cbert33/GLM-5.3-Flash-Uncensored-EXL3-DGX-Sliced](https://huggingface.co/cbert33/GLM-5.3-Flash-Uncensored-EXL3-DGX-Sliced)

This release is focused on performance for those two models. Work continues on
generation throughput, long-context prefill, prefix-cache reuse, speculative
acceptance, and concurrent agentic workloads.

### MiMo V2.6 model and protocol support

- Added target, MTP, and Omni model implementations for MiMo V2.6.
- Added attention-sink loading and semantics, shared-cache policy, Eagle3
  exposure, multimodal input handling, and a lower-overhead vision-index path.
- Added MiMo reasoning and strict tool parsing, structural-tag registration,
  stream reconciliation, auto-tool constraints, and closed-call parameter
  handling.
- Prevented structural tool-closing sequences inside parameter values from
  terminating a call early.
- Scoped processor arguments by modality and corrected encoder-cache
  allocation stalls.

### DFlash and DiffKV

- Added FP8 KV support for MiMo in the Triton DiffKV backend.
- Added quantized query support, split QK verification, fused RoPE and value
  scaling, fused verification preparation, KV reuse, and a lower SM12x launch
  threshold.
- Added bounded multi-token split-KV verification for uniform and mixed CUDA
  graph batches, including whole speculative verification per split-KV
  program.
- Added DFlash context K/V precomputation under CUDA graph capture.
- Added a verifier gather that rejects draft slots never proposed after a
  prefill boundary. This prevents stale input IDs from participating in
  probabilistic acceptance.

### EXL3 and model loading

- Corrected asymmetric QKV padding and trimming for EXL3 tensors.
- Added expert-only rank slices and calibrated KV scale loading.
- Preserved FP8 weights and scales across loader calls.
- Added rank-agnostic MoE inputs, pipeline-parallel intermediate tensors,
  pipeline-parallel MTP embeddings, and the required Mamba state dtype.
- Reworked routed-expert capture and corrected its lifecycle and shape
  handling.

### Consumer Blackwell and long-context kernels

- Added SM120/SM121 CUTLASS grouped GEMM builds and runtime advertisement.
- Corrected deterministic TopK, KDA/GDN grids, recurrent-state preservation,
  and custom-allreduce selection on consumer Blackwell.
- Added FlashInfer `gvr_2` decode TopK and tensor-parallel row-sharded
  long-context indexer prefill.
- Corrected sparse-indexer and K-pool tail behavior.
- Reworked hybrid KV group planning, weighted padding, tensor-parallel
  invariance, connector rules, and BLHNC grouping.

### GLM and shared runtime

- Corrected GLM vision rotary behavior and exact encoder-cache sizing.
- Added GLM pipeline-parallel model and MTP support, fused multi-step decode,
  shallow tool grammar, built-in tool fallback, and tool-result rendering.
- Corrected prefix-cache and Mamba resume boundaries, draft-embedding rebuilds,
  encoder allocation stalls, and resumable scheduler handoffs.

### Qualification results

The MiMo qualification used the linked rank-sliced EXL3 target and matching
DFlash checkpoint across two DGX Spark systems with FP8 target KV cache,
prefix caching, an 800K-token context limit, probabilistic DFlash K=4, and
explicit `--no-async-scheduling`. Asynchronous MiMo DFlash is not qualified in
Release 7.

The 62,287-token agentic acceptance workload passed cold, cached-prefix, and
concurrent text/reasoning/tool requests.

| Measurement | Result |
| --- | ---: |
| Accepted draft tokens | 2,178 of 7,981 |
| Overall draft-token acceptance | 27.3% |
| Draft-position acceptance, 1 through 4 | 61.1%, 27.7%, 13.2%, 7.2% |
| Cached-prefix generation throughput | 30.3 tokens/s |
| Concurrent long-request throughput | 28.4 tokens/s |
| Aggregate throughput during overlap | 29.4 tokens/s |

These are observations from one two-node deployment, not hardware limits.
MiMo audio and multimodal serving were not requalified in this release.

### Upgrade notes

- Use the `release-7` tag for this source snapshot.
- Use the same image, source revision, target checkpoint, DFlash checkpoint,
  and runtime flags on both tensor-parallel ranks.
- For the qualified MiMo DFlash route, use probabilistic K=4 and pass
  `--no-async-scheduling` explicitly.
- Treat any fallback backend, missing native symbol, rank divergence,
  cache-layout warning, or parser failure as a failed deployment.

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
