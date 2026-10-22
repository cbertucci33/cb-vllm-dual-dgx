#!/usr/bin/env bash
set -euo pipefail
export RUNNER_VERIFY_BUILD_ONLY=1
exec bash /opt/project/worktrees/glm32-three-improvements-20261020/build/build_image.sh \
  766e012cd1b87f1a2d85153d94d138831092a7cd \
  pytorch/manylinuxaarch64-builder:cuda13.0-b8b5f17a7d9ccfc25bbc5cf17b3fcea12964a042@operator:private.invalid \
  nvidia/cuda:13.0.3-base-ubuntu24.04@operator:private.invalid \
  nvidia/cuda:13.0.3-devel-ubuntu24.04@operator:private.invalid \
  vllm-glm-mimo:v032-766e012cd1-candidate
