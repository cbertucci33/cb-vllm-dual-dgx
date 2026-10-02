# cb-vllm-dual-dgx

This repository contains the two-node NVIDIA DGX Spark vLLM runner for
rank-sliced EXL3 checkpoints and related hybrid-model serving work. Model
weights are published separately. The active line is Release 6 on vLLM 0.30.
The vLLM 0.29 source remains available on the `release/v0.29` maintenance line.

The runner supports GLM chat, reasoning, tools, multimodal input, hybrid KDA,
native sparse MLA on GB10, EXL3 tensor parallelism, and DFlash2. It does not
hard-code a context length, concurrency limit, KV-cache allocation, network
address, or model path.

The qualified GLM target is
**[cbert33/GLM-5.3-Flash-Uncensored-EXL3-DGX-Sliced](https://huggingface.co/cbert33/GLM-5.3-Flash-Uncensored-EXL3-DGX-Sliced)**.
Other compatible EXL3 checkpoints must satisfy the rank-slicing contract
documented below.

## Release 6

Release 6 replatforms the active runner from vLLM 0.29 to vLLM 0.30 and
renames the project from `vllm-v29-glm53flash-exl3-dgx` to
`cb-vllm-dual-dgx`. Release 6 keeps the two-node DGX Spark deployment contract
and the GLM EXL3/DFlash2 path from Releases 1 through 5.

Changes in this release:

- updated GLM for current vLLM interfaces, including the DFlash draft
  boundary, draft-cache layout, auxiliary hidden states, B12X MXFP8 loading,
  FP8 context-KV rows, and GLM chat-template handling;
- retained asynchronous scheduling and DFlash support while carrying forward
  the hybrid recurrent-state publication and replay-boundary rules;
- restored deterministic GB10 TopK behavior and the GLM K-pool path, with
  GB10 route-capacity and SM121 build checks;
- reduced host dispatch in DFlash and DSpark metadata preparation,
  draft-input padding, and Mamba alignment work;
- added explicit warmup for the request-time sampler, routing, compressed KV,
  fused projection, DFlash, and first-use kernel paths;
- ported generic rank-sliced EXL3 loading to vLLM 0.30, including rank-name
  normalization, dense metadata hydration, safetensor-storage detachment, and
  rank-local expert slabs; and
- retained the earlier cache, DFlash, KDA, TopK, and K-pool work represented by
  [vLLM #54163](https://github.com/vllm-project/vllm/pull/54163),
  [#55122](https://github.com/vllm-project/vllm/pull/55122),
  [#56196](https://github.com/vllm-project/vllm/pull/56196),
  [#56794](https://github.com/vllm-project/vllm/pull/56794),
  [#56960](https://github.com/vllm-project/vllm/pull/56960),
  [#57161](https://github.com/vllm-project/vllm/pull/57161),
  [#57477](https://github.com/vllm-project/vllm/pull/57477),
  [#57605](https://github.com/vllm-project/vllm/pull/57605),
  [#58450](https://github.com/vllm-project/vllm/pull/58450), and
  [#58762](https://github.com/vllm-project/vllm/pull/58762).

### Current capacity figures

The current two-rank GLM operating profile uses 800,000 maximum model tokens,
two sequences, FP8 DS-MLA KV cache, and 9.44 GB of KV reservation per rank.
At that setting, vLLM reported 1,174,545 KV-cache tokens, or 1.47 times the
configured model length. This is a capacity observation, not a Release 6
throughput benchmark.

The measured production throughput and DFlash2 acceptance figures below are
retained from the Release 5 two-node deployment. Release 6 has not yet produced
a comparable production benchmark, so this README makes no new throughput claim.

## Basic deployment

The qualified deployment uses two NVIDIA DGX Spark systems with tensor
parallelism across the ConnectX fabric. Both ranks must use the same image,
source revision, rank-sliced checkpoint, DFlash checkpoint, and runtime flags.

1. Clone the source release and download the qualified models:

   ```bash
   git clone --branch release-6 --single-branch \
     https://github.com/cbertucci33/cb-vllm-dual-dgx.git
   cd cb-vllm-dual-dgx
   build/download_models.sh /srv/glm53/models
   ```

   Follow [`build/README.md`](build/README.md) to fetch every pinned source,
   build the native artifacts, and create the final image.

2. The downloaded target model is already rank-sliced for TP2. To use another
   compatible GLM-5.3 Flash EXL3 checkpoint, inspect and slice it first:

   ```bash
   python3 tools/slice_exl3_checkpoint.py \
     /path/to/source-checkpoint \
     --tp 2 \
     --plan

   python3 tools/slice_exl3_checkpoint.py \
     /path/to/source-checkpoint \
     /path/to/rank-sliced-checkpoint \
     --tp 2
   ```

3. Transfer the exact built image to the peer node. Start rank 1 in headless
   mode before rank 0. The tested container contract uses host networking and
   IPC, `/dev/infiniband`, `IPC_LOCK`, unlimited memlock, and a RoCE GID index
   discovered from the current host after boot. Do not hard-code a stale GID
   index.

4. Launch with the model-facing options in [Serving configuration](#serving-configuration).
   Verify `/v1/models`, both ranks, the selected NVIDIA backends, and one cold
   plus one cached-prefix request. Treat a fallback backend, missing native
   symbol, rank error, or cache-layout warning as a failed deployment.

## Release 5 production measurements

This Release 5 snapshot covers 922 completed organic requests on one two-node
DGX Spark deployment. The runtime used an 800,000-token model limit, a
971,162-token cache, and DFlash2 with seven proposals. Prompt length, output
length, cache warmth, reasoning depth, and concurrency varied. These are
production observations, not hardware limits or a controlled benchmark.

### Generation throughput

The native vLLM interval logger reports generation throughput approximately
every 10 seconds. Intervals with no generated tokens are excluded.

| Measurement | Result |
| --- | ---: |
| Active generation intervals | 1,489 |
| Generated tokens | 349,439 |
| Mean throughput | 23.5 tokens/s |
| Median throughput | 21.2 tokens/s |
| P90 throughput | 39.7 tokens/s |
| Peak throughput | 65.8 tokens/s |

### Responsiveness and cache reuse

| Measurement | Result |
| --- | ---: |
| Completed requests | 922 of 922 |
| Errors, aborts, length stops, or repetition stops | 0 |
| Prompt tokens | 107,005,934 |
| Mean prompt length | 116,058 tokens |
| Cached prompt tokens | 102,638,592 |
| Prefix-cache reuse | 95.9% |
| Newly computed prompt tokens | 4,367,342 |
| Aggregate prompt-processing throughput | 1,423 tokens/s |
| Mean prefill time | 3.33 s |
| Mean time to first token | 3.54 s |
| Mean end-to-end request time | 15.94 s |

Prompt-processing throughput divides newly computed prompt tokens by total
prefill time. Prefix-cache reuse divides cached prompt tokens by queried prompt
tokens.

### DFlash2 acceptance

| Measurement | Result |
| --- | ---: |
| Verification steps | 106,724 |
| Draft tokens | 747,064 |
| Accepted draft tokens | 242,698 |
| Overall draft-token acceptance | 32.5% |
| Accepted draft tokens per verification step | 2.27 |
| Effective verification span | 3.27 tokens |

| Draft position | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Acceptance | 70.4% | 48.8% | 34.9% | 26.0% | 19.7% | 15.4% | 12.1% |

The effective verification span includes the target token produced by each
verification step.

## Release 5

Release 5 repairs long-prefix DFlash state ownership and makes the GB10 TopK
path deterministic. It retains the GLM, EXL3, sparse MLA, B12X, FP8 KV,
DFlash2, tool-use, reasoning, and multimodal features from earlier releases.
The runtime source ends at commit `10b33ab40e`.

Changes in this release:

- completed the GLM KDA checkpoint path from
  [vLLM #56960](https://github.com/vllm-project/vllm/pull/56960) by
  materializing the exact DFlash replay boundary before publishing recurrent
  state, so a later request cannot restore a logical checkpoint that was never
  written by a forward pass;
- completed the DFlash and DSpark capability split from
  [vLLM #54163](https://github.com/vllm-project/vllm/pull/54163) by separating
  draft-group identity from EAGLE and MTP target-cache tail dropping across
  the scheduler and hybrid cache coordinator;
- applied the DFlash replay reserve consistently to local prefix caching,
  NIXL, Mooncake, generic KV offload, and CPU offload restore paths;
- kept the private DFlash history ring out of external cache transfer while
  preserving original target cache-group identities through projected worker
  and connector layouts;
- preserved replay capability fields when cache groups are rebuilt for
  pipeline workers and external connectors;
- added copy-on-write ownership for every published Mamba, KDA, and
  convolution snapshot before a producer or cache-hit consumer mutates the
  underlying state page;
- adapted the deterministic persistent TopK work from
  [vLLM #55122](https://github.com/vllm-project/vllm/pull/55122), replacing
  atomic arrival-order emission in the GB10 kernel with index-ranked
  selection, lowest-index tie handling, and signed-zero normalization; and
- kept unsupported cooperative TopK paths fail-closed on GB10.

Focused verification covered producer and consumer copy-on-write ownership,
completion, abort, preemption, eviction, request reuse, connector projection,
exact replay boundaries, and deterministic TopK at the production 35K-row,
512-selection geometry. The final two-rank deployment passed cached-prefix and
forced-tool-call acceptance with DFlash2 configured for seven proposals.

## Release 4

Release 4 restores the optimized NVIDIA GLM K-pool numerical paths after the
Release 3 cache-correctness work and fixes DFlash prefix-checkpoint retention.
The runtime source ends at commit `3dc58d9ee0`.

Changes in this release:

- completed the optimized numerical path from
  [vLLM #57161](https://github.com/vllm-project/vllm/pull/57161), restoring
  CUDA head gating to BF16 operands with FP32 accumulation, caching the
  transposed projection weight, and using online FP32 softmax in NVIDIA
  K-pool prefill and decode;
- applied the precise DFlash and DSpark target-cache block-drop capability
  from [vLLM #54163](https://github.com/vllm-project/vllm/pull/54163) instead
  of the broader EAGLE-family capability; and
- activated the transient-checkpoint protection from
  [vLLM #56794](https://github.com/vllm-project/vllm/pull/56794) by keeping
  DFlash and DSpark on sparse checkpoint retention. Their separate draft KV
  no longer causes transient internal checkpoints to be published as reusable
  shared-prefix state.

## Release 3

Release 3 is an NVIDIA-focused cache-correctness update. It fixes the persistent
prefix corruption and speculative-boundary defects found during long-lived GLM
tool workloads. The runtime source ends at commit `73c0212232`.

Changes in this release:

- backported [vLLM #57477](https://github.com/vllm-project/vllm/pull/57477)
  so the NVIDIA GLM K-pool seed kernel addresses tail blocks with the padded
  indexer-page stride instead of overwriting unrelated cached pages;
- backported [vLLM #56196](https://github.com/vllm-project/vllm/pull/56196)
  so one-token and two-token prefill chunks write convolution state to their
  own current block rather than a shared prefix block;
- applied the DFlash and DSpark semantics from
  [vLLM #54163](https://github.com/vllm-project/vllm/pull/54163): separate
  draft KV no longer causes EAGLE-style target-tail drops;
- carried that precise target-tail capability through the scheduler, local
  cache manager, shared GDN metadata, Kimi K3 FlashKDA metadata, CPU offload,
  Mooncake storage, and generic KV offload paths;
- adapted the NVIDIA portions of
  [vLLM #55219](https://github.com/vllm-project/vllm/pull/55219) and
  [vLLM #56059](https://github.com/vllm-project/vllm/pull/56059) so the K-pool
  tail ring covers the full speculative window and fails safely on an invalid
  physical block;
- adapted [vLLM #57605](https://github.com/vllm-project/vllm/pull/57605) so
  Mamba lookahead allocation, estimation, null padding, and checkpoint identity
  use the same main-model boundary without double-counting a page; and
- retained the specialized NVIDIA GLM aliased tail layout and made invalid
  geometry fail closed. No generic fallback and no AMD-path changes were added.

Focused verification covered the repaired K-pool stride, short-chunk
convolution ownership, DFlash target-tail behavior, Kimi checkpoint placement,
speculative tail capacity, physical-block bounds, and Mamba allocation and
checkpoint boundaries. Eight focused NVIDIA GPU regressions and two specialized
layout tests passed before the final two-rank acceptance.

## Release 2

Release 2 is the first production-qualified update to the original runner. The
runtime source ends at commit `e3c9c6943`.

Changes in this release:

- backported the NVIDIA K-pool scheduling work from
  [vLLM #57161](https://github.com/vllm-project/vllm/pull/57161) and the
  internal KDA prefill checkpoints from
  [vLLM #56960](https://github.com/vllm-project/vllm/pull/56960);
- completed the scheduler, cache-manager, and worker control path for the KDA
  checkpoints introduced by
  [vLLM #56960](https://github.com/vllm-project/vllm/pull/56960);
- retained stable numerics around
  [vLLM #57161](https://github.com/vllm-project/vllm/pull/57161) by restoring
  FP32 operands for sparse-attention head gating and the two-pass FP32 K-pool
  softmax;
- preserved streamed reasoning, message, and function-call identities in the
  final OpenAI Responses object, including tool-call IDs and message logprobs;
- required an explicit prefix-match unit for FlashKDA checkpoints and promoted
  checkpoint page addressing to int64; and
- qualified DFlash2 with seven proposals on the two-node DGX Spark deployment.

The release retains the original runner's GLM, EXL3, sparse MLA, B12X,
FlashKDA, tool-use, reasoning, and multimodal paths.

## Release 1

Release 1 is the original vLLM 0.29 integration. It introduced the following
23 changes. The order matters because later fixes depend on cache layouts and
native interfaces established earlier.

1. **Started from vLLM 0.29.0.** The source base is tag `v0.29.0`, commit `98dff2a81d747d1dba01a47f939f48c3526d4206`. The runtime starts from the official `vllm/vllm-openai:v0.29.0` image pinned by digest.

2. **Added the GLM-5.3 Flash architecture.** Imported the model work from vLLM PR #53906, including hybrid KDA and NoPE sparse MLA, multimodal processing, MTP, configuration classes, registry entries, and weight-loading rules.

3. **Added the GLM serving protocol.** Packaged the official chat template from Z.ai model revision `690b705278a3a58e538fcb37c2ca8b5f9511213c`, retaining compatible reasoning, tool-call, parallel tool-result, and request-level thinking behavior.

4. **Replaced prebuilt FlashInfer with the required source build.** Built FlashInfer 0.6.18 from commit `5cc867a9bb560bc89b91dbc738a9f63f09beb89b`, which contains the native SM120/SM121 `GLM53_NOPE` sparse-MLA work from FlashInfer PRs #4802 and #4947.

5. **Matched vLLM to the native FlashInfer ABI.** Added the `FLASHINFER_MLA_SPARSE_SM120` backend and corrected packed FP8 cache records, physical sparse page tables, K/V scales, sequence lengths, indexer state, and decode metadata for GLM's NoPE layout.

6. **Corrected the target-cache format.** Selected `fp8_ds_mla` for native sparse MLA and used its real packed record size. Sliding-window layers can use their own cache format instead of inheriting the target MLA format.

7. **Added a GB10-safe exact TopK implementation.** The cooperative path exceeded the shared-memory limit on devices with less than 128 KiB per block. Added an exact non-cooperative streaming-radix path, including tie and threshold handling. Existing paths remain available on other GPUs.

8. **Added EXL3 to vLLM.** Implemented EXL3 configuration, tensor loading, prefill planning, logits handling, and routed expert integration.

9. **Added lossless tensor-parallel checkpoint slicing.** `tools/slice_exl3_checkpoint.py` splits routed expert tensors across ranks without dequantization or requantization. It reconstructs every source tensor bit for bit before publishing the output.

10. **Connected rank-sliced EXL3 to Sparkinfer.** Pinned the Sparkinfer project, now hosted as [`local-inference-lab/b12x`](https://github.com/local-inference-lab/b12x), at [commit `d4438d490691f79022fdfc8149e1c5f161d15445`](https://github.com/local-inference-lab/b12x/commit/d4438d490691f79022fdfc8149e1c5f161d15445) and used its Trellis planning, scratch, binding, and execution interfaces for supported tensor-parallel shapes. The dependency is fetched during build preparation and is not vendored in this repository.

11. **Built ExLlamaV3 for ARM64.** Pinned commit `c5d9c657966ffeeaa9353f0cc899f18629da4a13`. Removed optional x86 AVX translation units from the ARM64 build, added fail-closed stubs for the unavailable CPU all-reduce path, and packaged only the extension consumed by vLLM.

12. **Updated CUTLASS DSL compatibility.** FlashInfer requires NVIDIA CUTLASS DSL 4.7.0. Sparkinfer's exact 4.6.0 dependency and B12X's exact 4.6.2 dependency were changed to `>=4.7.0,<5`. The final build pins 4.7.0.

13. **Added B12X dense MXFP8 dispatch.** Pinned B12X 1.2.6 at commit `ab6eea89b5b5e334ac6e9f2c503c1de60c3f216c`. B12X handles supported small batches, while FlashInfer Cutlass handles other shapes. Fixed the B12X capability check so `(False, reason)` does not evaluate as supported.

14. **Added DFlash2 for GLM.** Imported auxiliary target-state capture, compact MTP prefill, grouped convolutions, candidate selection, and the speculative decoding flow. The draft checkpoint uses ModelOpt MXFP8 weights and a rowwise FP8 draft head.

15. **Separated target and draft cache formats.** The target cache uses packed `fp8_ds_mla`; the DFlash GQA history ring uses `fp8_e4m3`. Draft pages no longer inherit target-page padding.

16. **Repaired the FlashKDA call boundary.** The pinned API requires caller-allocated output, final-state, and workspace buffers. The GLM prefill path now supplies all three. A focused test binds the caller to the pinned signature.

17. **Corrected DFlash page geometry.** Used the draft tensor's actual 16-token page size instead of the larger target sparse-MLA page size.

18. **Moved DFlash history into a private fixed ring.** DFlash history now uses bounded request-local storage and does not allocate shared target-pool blocks. Allocation, copy, zeroing, retirement, prefix resume, and worker page-table generation use the same ownership rule.

19. **Fixed scheduler admission for the private ring.** The scheduler no longer charges shared-pool block IDs for fixed DFlash storage. This removed an admission ceiling that could leave a continuation waiting while physical KV usage remained near zero.

20. **Fixed reported cache capacity.** Startup and metrics now exclude the private DFlash ring from shared-pool demand, matching the managers used for real request admission.

21. **Made native packaging source-aware.** FlashInfer JIT and packaged native artifacts are keyed to source identity and ABI inputs. An older cache entry can no longer hide a source or kernel change.

22. **Added complete warmup coverage.** Warmup covers the selected FlashInfer, EXL3, Sparkinfer, B12X, FlashKDA, TopK, and speculative rejection paths before CUDA graph capture.

23. **Allowed shorter DFlash2 inference blocks.** A checkpoint may run fewer proposals than its trained maximum, but it may not exceed that maximum. The tested checkpoint was trained with block size 8 (seven proposals) and was also smoke-tested with block size 6 (five proposals).

The static interface review and third-party notices for Releases 1 through 5
remain on the `release/v0.29` maintenance line.

## Pinned build inputs

`build/versions.env` is the machine-readable source of truth.

| Component | Pinned input |
| --- | --- |
| vLLM source | `0.29.0`, commit `98dff2a81d747d1dba01a47f939f48c3526d4206` |
| vLLM runtime image | `vllm/vllm-openai:v0.29.0@sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1` |
| CUDA build image | `nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6` |
| vLLM CUTLASS source | `da5e086dab31d63815acafdac9a9c5893b1c69e2` |
| FlashInfer | `0.6.18`, commit `5cc867a9bb560bc89b91dbc738a9f63f09beb89b` |
| FlashInfer CCCL | `16bd510c9b712e82b0ab6cbb630d8e29ba1f7116` |
| FlashInfer CUTLASS | `b46b16d003484063bca4ed365e44095c4c6ed633` |
| FlashInfer spdlog | `c3aed4b68373955e1cc94307683d44dca1515d2b` |
| Sparkinfer (now `local-inference-lab/b12x`) | [`d4438d490691f79022fdfc8149e1c5f161d15445`](https://github.com/local-inference-lab/b12x/commit/d4438d490691f79022fdfc8149e1c5f161d15445) |
| ExLlamaV3 | `c5d9c657966ffeeaa9353f0cc899f18629da4a13`, format `0.0.43` |
| B12X dense-kernel package (`local-inference-lab/b12x`) | `1.2.6`, commit `ab6eea89b5b5e334ac6e9f2c503c1de60c3f216c` |
| NVIDIA CUTLASS DSL | `4.7.0` |
| GLM chat template | Z.ai revision `690b705278a3a58e538fcb37c2ca8b5f9511213c` |
| Qualified target model | [`cbert33/GLM-5.3-Flash-Uncensored-EXL3-DGX-Sliced`](https://huggingface.co/cbert33/GLM-5.3-Flash-Uncensored-EXL3-DGX-Sliced), revision `43fe4b2aba293c2df3b413f63d926aa1a0725d26` |
| DFlash2 model | [`local-inference-lab/GLM-5.3-Flash-DFlash2`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2), revision `713226ab03bc38afdf955c7450436c2f7176f6f8` |

The build scripts fetch every external source, build the custom FlashInfer
wheel, and download both model repositories. These artifacts are not vendored
here.

## Prepare a tensor-parallel checkpoint

### Compatibility

The slicer accepts a checkpoint directory, not a Docker image. The source
directory must contain `config.json`, `quantization_config.json`,
`model.safetensors.index.json`, and every Safetensors shard named by the index.

Supported inputs have all of these properties:

- GLM-5.3 Flash routed-expert tensors use the standard
  `layers.{L}.mlp.experts.{E}.{gate_proj|up_proj|down_proj}` names, optionally
  below `model.` or `language_model.`;
- EXL3 uses the MCG codebook;
- the quantization bitrate is an integral 3, 4, 5, or 6 bits;
- every routed expert has `trellis`, `suh`, `svh`, and `mcg` tensors for all
  three projections;
- MoE layer and expert IDs are contiguous; and
- each split axis is divisible by the requested tensor-parallel size.

Shard count, shard filenames, layer count, expert count, and model bitrate are
discovered from the checkpoint. They are not fixed to the qualified model.

The slicer does not support `mul1` codebooks, missing or mixed codebook markers,
differently named expert layouts, nonintegral EXL3 bitrates, or dense-only EXL3
checkpoints. It fails before writing the output when an input does not match
the contract.

Inspect the split plan:

```bash
python3 tools/slice_exl3_checkpoint.py \
  /path/to/source-checkpoint \
  --tp 2 \
  --plan
```

Create the checkpoint:

```bash
python3 tools/slice_exl3_checkpoint.py \
  /path/to/source-checkpoint \
  /path/to/rank-sliced-checkpoint \
  --tp 2
```

The output path must not exist. The tool writes to a temporary sibling, validates every tensor, then publishes the completed directory with one rename. It rejects incomplete expert records, non-MCG codebooks, unsupported bitrates, non-contiguous layers or experts, and dimensions that do not divide by the tensor-parallel size.

This conversion prepares the checkpoint for the patched rank-sliced EXL3 and
Trellis runtime. It is not a general startup-memory optimization. The converter
writes each rank's tensor under a rank-qualified name while retaining the
source shard grouping, so tensors for both ranks can remain in the same
Safetensors file. During loading, each worker discards nonlocal tensors one at
a time. The avoidable transient allocation is therefore bounded by an
individual already-sliced tensor, not half of the full checkpoint. The
conversion does not halve checkpoint I/O and should not be expected to recover
several GiB from an unrelated boot OOM.

Use the converted checkpoint only with this repository's matching rank-sliced
EXL3 loader and the tensor-parallel size recorded in its metadata. A regular
GLM EXL3 image does not understand this format.

The qualified checkpoint used a two-way tensor-parallel split, 4-bit MCG EXL3, 43 MoE layers, 288 experts per layer, and 92 output shards. These values describe the tested artifact, not runner limits.

## Build the image

The final image starts from the pinned official vLLM runtime. A separate CUDA development stage compiles the changed vLLM extension and pinned external components. The build overlays the Python source and required native artifacts onto a fresh runtime stage.

See [`build/README.md`](build/README.md) for the reproducible build flow. The build downloads and hashes its inputs before offline compilation, then records source, wheel, extension, and toolchain provenance in the image.

## Serving configuration

The tested path uses these model-facing options:

```text
--quantization exl3
--load-format instanttensor
--attention-backend FLASHINFER_MLA_SPARSE_SM120
--linear-backend b12x
--kv-cache-dtype fp8_ds_mla
--kv-cache-dtype-skip-layers sliding_window
--block-size 2304
--prefix-match-unit 512
--speculative-config {"method":"dflash","model":"/path/to/dflash2","num_speculative_tokens":7}
--tool-call-parser glm47
--reasoning-parser glm45
--enable-auto-tool-choice
```

Release 2 was qualified with seven proposals, the checkpoint's trained
maximum. Five proposals were also smoke-tested. Select the proposal count from
measured acceptance and end-to-end throughput for the intended workload.

Set model length, sequence concurrency, batch-token limits, KV memory, network addresses, ports, model paths, and chat defaults for the target deployment. Effective context and concurrency depend on checkpoint geometry, cache allocation, request mix, and available memory.

## Validation

Completed checks include:

- static caller and dependency interface review for the selected path;
- focused CPU and CUDA tests for changed cache and kernel boundaries;
- exact GB10 TopK tests, including ties and threshold cases;
- lossless rank-sliced checkpoint reconstruction checks;
- full two-rank checkpoint load;
- FlashInfer, Sparkinfer, EXL3, FlashKDA, B12X, warmup, and CUDA graph capture;
- OpenAI-compatible chat, code, and structured tool-call requests;
- multi-turn tool-use smoke testing;
- continuation beyond the original scheduler-admission failure;
- private-ring capacity reporting through the real startup path;
- five-proposal DFlash2 startup, graph capture, and exact draft accounting.

These checks do not establish every workload, context length, concurrency level, multimodal shape, or hardware revision.

## License

The vLLM-derived source remains under the Apache License 2.0 in `LICENSE`. External dependencies and model artifacts retain their own licenses and terms.
