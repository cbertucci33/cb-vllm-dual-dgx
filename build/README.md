# Unified vLLM 0.32 image inputs

> Migration status: source-only .32 integration. No .32 native build or runtime acceptance has been performed. Existing base-image digest examples below are historical operands, not a newly qualified .32 build.
`build_image.sh` builds the official `docker/Dockerfile` `vllm-openai` target
from this v0.32 checkout, then adds the pinned GLM/MiMo native dependencies in
one ARM64 image. It neither downloads nor copies model artifacts. The
`verify_runtime.py` acceptance step checks Python distributions, the GLM sparse
MLA API, EXL3/Trellis APIs, and FlashKDA operators without loading a model.

All three base images (official Docker build base, official final base, and
CUDA development base) must be supplied by immutable `@sha256:` references.
The exact vLLM commit is a build argument, not a self-reference in versions.env.
The build refuses a different HEAD or any tracked or untracked working-tree
changes. `provenance.py` records the source commit, base-image digests, pins,
patches, profile evidence, and input hashes; OCI labels retain source and
native revisions. Python package versions are pinned, but PyPI transitive
artifacts are not hash-locked, so this is not byte-for-byte package resolution.

Run from a committed clean source tree and pass that exact full commit:

```bash
build/build_image.sh '<exact-clean-source-commit>' \
  'pytorch/manylinuxaarch64-builder:cuda13.0-b8b5f17a7d9ccfc25bbc5cf17b3fcea12964a042@operator:private.invalid' \
  'nvidia/cuda:13.0.3-base-ubuntu24.04@operator:private.invalid' \
  'nvidia/cuda:13.0.3-devel-ubuntu24.04@operator:private.invalid' \
  vllm-glm-mimo:v0.32-local
```

The ARM64 builder name and tag above come from v0.32's own Buildkite ARM64
release job. The Dockerfile's default
`pytorch/manylinux2_28-builder:cuda13.0-78e737ad29420ffc4800e677c51e2a852caf8359`
is an AMD64 image and must not be used for the GB10 build.
The three immutable references above were read back from Docker Hub on
2026-10-18, including their Linux ARM64 config metadata. This verifies
identity and platform, not a new native build or runtime qualification.

The abbreviated command deliberately requires exact image digests from the
build owner; mutable tags are rejected. Build and runtime are Linux ARM64,
with `TORCH_CUDA_ARCH_LIST=12.1a` and `FLASHINFER_CUDA_ARCH_LIST=12.1a` for GB10
SM121. The official Dockerfile and `requirements/cuda.txt` remain the package
authority: Torch 2.13.0, FlashInfer 0.7.0.post1, CUTLASS DSL 4.8.0, Humming
0.1.17, NCCL 2.30.7, and Triton 3.7.1. For the current GLM compact
sparse-MLA route, the overlay removes official FlashInfer wheel/cubin/JIT-cache
packages and stale AOT/JIT content before installing pinned FlashInfer source.
The source-built `0.7.0.post1` distribution is verified only when `direct_url.json` points to the pinned checkout and the
verifier finds the public `flashinfer.decode` route, GLM sparse-MLA API,
row-stride handling, and the backend-owned FP8 KV row contract (528 bytes
for SM120 NoPE, 656 bytes for RoPE). Provenance alone does not establish this
API contract.

The .32 dependency pin is FlashInfer
`946200de1ae94fc93fdd0926f0a13afd1fa7f0f1`. Unlike the accepted earlier
`dc04f50c` dependency, this pin alone requires 656-byte GLM rows and is not
compact-cache compatible. `flashinfer-glm53-compact-cache.patch` restores
528-byte NoPE payloads across the public validator, inner Python layout resolver,
native model byte count and prefill IO traits. RoPE rows retain their 656-byte
contract. The build compiles native layout assertions; the CPU verifier passes
the actual vLLM cache-spec width through the installed FlashInfer wrapper and
inner layout guard. GPU attention numerics and performance remain runtime checks.

## Reconciliation

- vLLM 0.30.0 and its digest-pinned 0.30 image are removed from the build
  baseline. The sole source is vLLM 0.32.0 at the explicit clean source commit
  argument and the official Dockerfile/current requirements.
- CUDA 13.0.2 is replaced with CUDA 13.0.3. All three base-image digests are
  explicit Linux ARM64 inputs and were resolved and validated against their
  registry manifests on 2026-10-08.
- The official package pins are retained: Torch 2.13.0, Python 3.12, FlashInfer
  0.7.0.post1, CUTLASS DSL 4.8.0, Humming 0.1.17, NCCL 2.30.7, and Triton
  3.7.1. FlashInfer is rebuilt from its pinned source plus the explicit
  compact-cache compatibility patch.
- The accepted native commits are retained for FlashInfer and its submodules,
  Sparkinfer, ExLlamaV3, B12X, FlashKDA and its CUTLASS submodule, and vLLM's
  CUTLASS checkout. Model repository/revision fields and model weights are not
  part of image provenance.
- The accepted Sparkinfer route-pack/CUTLASS-DSL patch, B12X vLLM-loader/
  CUTLASS-DSL patch, and GB10 48-SM MXFP8 profile plus evidence are tracked as
  source inputs; native patch application is checked before application.
- v0.32's current CMake is retained with a narrow opt-in
  `VLLM_DGX_SM121_ONLY` gate; it ports the accepted per-kernel Marlin,
  scaled-mm, Hadacore, and Marlin-MoE architecture branches while retaining
  v0.32's CMake. The official Docker build forwards the option through
  `CMAKE_ARGS`. The verifier uses `cuobjdump --list-elf` to require only the
  exact `sm_121` SASS ELF target in the core, FlashKDA, ExLlamaV3, and
  FlashInfer TopK artifacts. It does not inspect embedded PTX, so it makes no
  claim that those binaries contain no PTX.
- No old Dockerfile, private EXL3 metadata hotfix, or v0.30 API overwrite is
  imported.

## Verification boundary

This source-only handoff has not built an image. It does not establish that the
published CUDA development image, package build backends, FlashInfer source
checkout, or SM121 native compilation succeeds. A successful image build is
still required to close those boundaries. The
post-build verifier establishes installed package versions and exported
symbols/APIs, the public FlashInfer decode route through the registered vLLM
SM120 backend, and the configured 528-byte NoPE and 656-byte RoPE FP8 MLA
cache-spec/dispatcher contracts. It does not establish the native cache
tensor-layout ABI or GPU execution behavior; those remain dynamic, and this is not model-load, GLM,
MiMo, performance, or live-service acceptance.

## Linked worktree staging

The build script stages a standalone clone beneath `.build-contexts/` when
the release checkout is a linked worktree. Docker receives an internal `.git`
directory, the verified official version tag, and no object alternates. The
source commit is unchanged; version metadata is not synthesized. Context
creation and native builds remain deferred until the build resource envelope
and any required maintenance window are established.

### Invalid sparse KV slots

The pinned FlashInfer V32-family kernel gathered cache row zero for negative indices. QK masking did not prevent NaNs in an unselected V row from poisoning PV. The native patch restores the accepted immutable zero-row gather for invalid indices across decode and all shared FP8 prefill gathers (MG, SG, and swapAB), rather than reading mutable cache row zero. The GLM first-prefill call selects PREFILL_MG at its actual 26-token shape, so a decode-only correction is insufficient. No additional per-tile zeroing or synchronization is introduced. It preserves valid rows, FP8, GLM compact528/RoPE656 layouts, and sampling. GPU regression checks require unused-row poisoning to leave results bit-identical, including holes and empty selections.
