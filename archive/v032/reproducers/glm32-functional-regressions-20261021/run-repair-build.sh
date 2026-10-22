#!/usr/bin/env bash
set -euo pipefail
export RUNNER_VERIFY_BUILD_ONLY=1
exec bash /opt/project/worktrees/glm32-three-improvements-20261020/build/build_image.sh \
  8db682eda5c32f91a4d464543c041b773b550933 \
  pytorch/manylinuxaarch64-builder:cuda13.0-b8b5f17a7d9ccfc25bbc5cf17b3fcea12964a042@operator:private.invalid \
  nvidia/cuda:13.0.3-base-ubuntu24.04@operator:private.invalid \
  nvidia/cuda:13.0.3-devel-ubuntu24.04@operator:private.invalid \
  vllm-glm-mimo:v032-8db682eda5-candidate
