# Reproducible image build

This directory contains the pinned inputs and scripts used to build the
GLM-5.3 Flash runner image. The repository does not store model weights,
third-party source trees, or compiled wheels.

## Requirements

- Linux on ARM64
- an NVIDIA DGX Spark or another CUDA 13 system that can compile SM121 code
- Docker with NVIDIA Container Toolkit support
- Git
- the Hugging Face `hf` CLI for model downloads
- enough storage for the target model, draft model, source trees, build
  workspace, and final image

Install the Hugging Face CLI if needed:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash -s
```

Public model downloads do not require a token. Set `HF_TOKEN` if Hugging Face
applies an anonymous-download limit.

## Download the models

`download_models.sh` downloads and verifies both qualified model revisions:

```bash
build/download_models.sh /srv/glm53/models
```

The resulting paths are:

- `/srv/glm53/models/target`
- `/srv/glm53/models/dflash2`

The repository and immutable revision for each model are stored in
`versions.env`.

## Fetch the build sources

Fetch every external source at its pinned commit, including the FlashInfer
submodules:

```bash
mkdir -p /srv/glm53/build
build/fetch_sources.sh /srv/glm53/build/checkouts

build/prepare_sources.sh \
  /srv/glm53/build/checkouts/flashinfer \
  /srv/glm53/build/checkouts/sparkinfer \
  /srv/glm53/build/checkouts/exllamav3 \
  /srv/glm53/build/checkouts/b12x \
  /srv/glm53/build/checkouts/cutlass \
  /srv/glm53/build/source-archives
```

`prepare_sources.sh` rejects dirty checkouts and wrong commits before it
creates the source archive manifest.

## Build the artifacts

Build the isolated compiler image from the two pinned base-image digests:

```bash
set -a
source build/versions.env
set +a

docker build \
  -f build/Dockerfile.builder \
  --build-arg BASE_IMAGE="$VLLM_BASE_IMAGE" \
  --build-arg CUDA_DEVEL_IMAGE="$CUDA_DEVEL_IMAGE" \
  -t glm53-runner-builder:internal-v3 .
```

Use the builder for all Python and CUDA artifacts:

```bash
repo=$(pwd)
work=/srv/glm53/build
run_builder() {
  docker run --rm --gpus all --entrypoint /bin/bash \
    -v "$repo:/repo:ro" \
    -v "$work:/work" \
    -w /repo \
    glm53-runner-builder:internal-v3 -lc "$1"
}

run_builder 'build/download_build_deps.sh /work/build-deps'

run_builder 'build/build_flashinfer_topk.sh \
  /work/checkouts/flashinfer /work/artifacts'

run_builder 'build/build_flashinfer_wheel.sh \
  /work/checkouts/flashinfer /work/build-deps /work/flashinfer-wheel'

flashinfer_wheel=$(find "$work/flashinfer-wheel" -maxdepth 1 \
  -name 'flashinfer_python-*.whl' -print -quit)

run_builder "build/build_wheels.sh \
  /work/source-archives \
  /work/flashinfer-wheel/$(basename "$flashinfer_wheel") \
  /work/build-deps /work/artifacts"

run_builder 'build/build_core_extension.sh \
  /work/source-archives /work/build-deps /work/artifacts'

build/finalize_artifacts.sh "$work/artifacts"
```

Each script verifies its source revision or input manifest. The FlashInfer
wheel is built from the pinned checkout instead of relying on an unpublished
binary.

## Build the runner image

Create the minimal Docker context and build the final image:

```bash
python3 build/prepare_image_context.py \
  /srv/glm53/build/source-archives \
  /srv/glm53/build/artifacts \
  /srv/glm53/build/image-context

set -a
source build/versions.env
set +a
vllm_commit=$(cat /srv/glm53/build/source-archives/vllm-commit.txt)

docker build \
  -f build/Dockerfile \
  --build-arg BASE_IMAGE="$VLLM_BASE_IMAGE" \
  --build-arg VLLM_COMMIT="$vllm_commit" \
  --build-arg FLASHINFER_COMMIT="$FLASHINFER_COMMIT" \
  --build-arg SPARKINFER_COMMIT="$SPARKINFER_COMMIT" \
  --build-arg EXLLAMAV3_COMMIT="$EXLLAMAV3_COMMIT" \
  --build-arg B12X_COMMIT="$B12X_COMMIT" \
  -t glm53-flash-exl3:internal-v3.0 \
  /srv/glm53/build/image-context
```

The final image writes its provenance to
`/opt/glm53-runner-provenance.json` and its verified runtime versions to
`/opt/glm53-runner-versions.json`.

## Build design

`Dockerfile.builder` combines the official runtime's Python and Torch stack
with CUDA development headers. Those headers are used only during compilation.
`Dockerfile` starts the final stage from the clean official runtime image.

The build recompiles the changed `_C_stable_libtorch` extension and overlays
the Python package plus the required native artifacts. Unchanged vLLM native
extensions, including the Rust frontend, remain from the official base image.

The pinned FlashInfer 0.7.0 source is the exact revision used to validate
compact 528-byte GLM-5.3 NoPE sparse-MLA rows in vLLM #58305. It includes
FlashInfer #5075, which added runtime KV row strides and the compact-row model
type. vLLM 0.30 uses CUTLASS DSL 4.7.1. The Sparkinfer and B12X patches relax
their older 4.6.x package constraints to the compatible 4.7.x line. B12X 1.3.0
already contains the corrected MXFP8 capability predicate.

The ExLlamaV3 patch excludes optional x86 CPU all-reduce translation units on
ARM64 and supplies fail-closed stubs. It does not change the CUDA EXL3 kernels.

## Reproducibility rules

- Build from clean source checkouts at the pinned commits.
- Hash source archives, wheels, extensions, and the toolchain.
- Compile external dependencies after the download step with network access
  disabled.
- Use the architecture flags in `versions.env` for GB10.
- Install local wheels with `--no-deps`.
- Adjust build parallelism to available RAM. The values in `versions.env` are
  conservative build defaults, not runtime settings.
- Verify imports, native symbols, packaged revisions, and artifact hashes in
  the final image.
