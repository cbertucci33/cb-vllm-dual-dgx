#!/usr/bin/env bash
set -euo pipefail
export RUNNER_VERIFY_BUILD_ONLY=1
exec bash /opt/project/worktrees/glm32-three-improvements-20261020/build/build_image.sh \
  95c9e5fc0e44b67dcbd9cebbb2ca5491da8b96e0 \
  pytorch/manylinuxaarch64-builder:cuda13.0-b8b5f17a7d9ccfc25bbc5cf17b3fcea12964a042@operator:private.invalid \
  nvidia/cuda:13.0.3-base-ubuntu24.04@operator:private.invalid \
  nvidia/cuda:13.0.3-devel-ubuntu24.04@operator:private.invalid \
  vllm-glm-mimo:v032-95c9e5fc0e-candidate
