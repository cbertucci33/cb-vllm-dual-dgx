#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(git -C "$here/.." rev-parse --show-toplevel)
source "$here/versions.env"

if [[ $# -ne 5 ]]; then
  echo "usage: $0 EXACT_SOURCE_COMMIT BUILD_BASE_IMAGE@operator:private.invalid FINAL_BASE_IMAGE@operator:private.invalid CUDA_DEVEL_IMAGE@operator:private.invalid FINAL_IMAGE_TAG" >&2
  exit 2
fi
source_commit=$1
build_base_image=$2
final_base_image=$3
cuda_devel_image=$4
final_tag=$5
for image in "$build_base_image" "$final_base_image" "$cuda_devel_image"; do
  if [[ ! $image =~ @sha256:[0-9a-f]{64}$ ]]; then
    echo "all build/final/CUDA development base images must be digest-pinned" >&2
    exit 2
  fi
done
if [[ ! $source_commit =~ ^[0-9a-f]{40}$ ]]; then
  echo "source commit must be a full 40-character commit ID" >&2
  exit 2
fi
actual=$(git -C "$repo" rev-parse HEAD)
if [[ $actual != "$source_commit" ]]; then
  echo "checkout is $actual; requested source commit is $source_commit" >&2
  exit 1
fi
if [[ -n $(git -C "$repo" status --porcelain --untracked-files=all) ]]; then
  echo "source checkout must have a clean tracked and untracked tree" >&2
  exit 1
fi

# Docker bind-mounts .git while deriving SCM versions. A linked worktree's
# .git file points outside its context, so give Docker a standalone clone.
if [[ -f "$repo/.git" ]]; then
  context_parent="$repo/.build-contexts"
  mkdir -p "$context_parent"
  context_repo=$(mktemp -d "$context_parent/${source_commit:0:12}.XXXXXX")
  branch=$(git -C "$repo" symbolic-ref --quiet --short HEAD) || {
    echo "linked source worktree must have a branch for context staging" >&2
    exit 1
  }
  git clone --no-local --no-hardlinks --single-branch --no-tags \
    --branch "$branch" --no-checkout "$repo" "$context_repo"
  git -C "$context_repo" fetch --no-tags "$repo" \
    "refs/tags/v${VLLM_VERSION}:refs/tags/v${VLLM_VERSION}"
  git -C "$context_repo" checkout --detach "$source_commit"
  test "$(git -C "$context_repo" rev-parse HEAD)" = "$source_commit"
  test -d "$context_repo/.git"
  test ! -f "$context_repo/.git/objects/info/alternates"
  repo=$context_repo
  here="$repo/build"
fi

base_tag="vllm-v${VLLM_VERSION}-source:${source_commit:0:12}"
python3 "$here/provenance.py" "$here/provenance.json" "$source_commit" \
  "$build_base_image" "$final_base_image" "$cuda_devel_image"
docker buildx build --platform linux/arm64 --load \
  --file "$repo/docker/Dockerfile" \
  --build-arg BUILD_BASE_IMAGE="$build_base_image" \
  --build-arg FINAL_BASE_IMAGE="$final_base_image" \
  --build-arg CUDA_VERSION="$CUDA_VERSION" \
  --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
  --build-arg NCCL_VERSION="$NCCL_VERSION" \
  --build-arg torch_cuda_arch_list="$TORCH_CUDA_ARCH_LIST" \
  --build-arg max_jobs="$VLLM_CORE_MAX_JOBS" \
  --build-arg nvcc_threads="$NVCC_THREADS" \
  --build-arg VLLM_DGX_SM121_ONLY="$VLLM_DGX_SM121_ONLY" \
  --build-arg VLLM_BUILD_COMMIT="$source_commit" \
  --target vllm-openai --tag "$base_tag" "$repo"
base_image_id=$(docker image inspect --format '{{.Id}}' "$base_tag")
if [[ ! $base_image_id =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "built vLLM base image has invalid identity: $base_image_id" >&2
  exit 1
fi

docker buildx build --platform linux/arm64 --load \
  --file "$here/Dockerfile.runner" \
  --build-arg VLLM_BASE_IMAGE="$base_tag" \
  --build-arg VLLM_BASE_IMAGE_ID="$base_image_id" \
  --build-arg BUILD_BASE_IMAGE="$build_base_image" \
  --build-arg FINAL_BASE_IMAGE="$final_base_image" \
  --build-arg CUDA_DEVEL_IMAGE="$cuda_devel_image" \
  --build-arg VLLM_SOURCE_COMMIT="$source_commit" \
  --build-arg CUDA_VERSION="$CUDA_VERSION" \
  --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
  --build-arg NCCL_VERSION="$NCCL_VERSION" \
  --build-arg FLASHINFER_COMMIT="$FLASHINFER_COMMIT" \
  --build-arg CUSPARSELT_VERSION="$CUSPARSELT_VERSION" \
  --build-arg CUSPARSELT_WHEEL_SHA256="$CUSPARSELT_WHEEL_SHA256" \
  --build-arg FLASHINFER_CCCL_COMMIT="$FLASHINFER_CCCL_COMMIT" \
  --build-arg FLASHINFER_CUTLASS_COMMIT="$FLASHINFER_CUTLASS_COMMIT" \
  --build-arg FLASHINFER_NIXL_COMMIT="$FLASHINFER_NIXL_COMMIT" \
  --build-arg FLASHINFER_SPDLOG_COMMIT="$FLASHINFER_SPDLOG_COMMIT" \
  --build-arg SPARKINFER_COMMIT="$SPARKINFER_COMMIT" \
  --build-arg EXLLAMAV3_COMMIT="$EXLLAMAV3_COMMIT" \
  --build-arg B12X_COMMIT="$B12X_COMMIT" \
  --build-arg FLASHKDA_COMMIT="$FLASHKDA_COMMIT" \
  --build-arg FLASHKDA_CUTLASS_COMMIT="$FLASHKDA_CUTLASS_COMMIT" \
  --build-arg VLLM_CUTLASS_COMMIT="$VLLM_CUTLASS_COMMIT" \
  --build-arg NVIDIA_CUTLASS_DSL_VERSION="$NVIDIA_CUTLASS_DSL_VERSION" \
  --build-arg CMAKE_VERSION="$CMAKE_VERSION" \
  --build-arg NINJA_VERSION="$NINJA_VERSION" \
  --build-arg MAX_JOBS="$MAX_JOBS" \
  --build-arg NVCC_THREADS="$NVCC_THREADS" \
  --tag "$final_tag" "$repo"

if [[ ${RUNNER_VERIFY_BUILD_ONLY:-0} == 1 ]]; then
  # Build beside a live model without acquiring its GPU or initializing CUDA.
  docker run --rm --runtime runc --network none --platform linux/arm64 \
    -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void \
    --entrypoint python3 "$final_tag" \
    /opt/vllm-runner/verify_runtime.py --build-only
else
  docker run --rm --gpus all --platform linux/arm64 --entrypoint python3 "$final_tag" \
    /opt/vllm-runner/verify_runtime.py
fi
