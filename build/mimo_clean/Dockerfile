# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG VLLM_COMMIT
ARG RUNNER_VERSION
ARG EXLLAMAV3_COMMIT
ARG FLASHINFER_COMMIT
ARG SPARKINFER_COMMIT

ENV TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer/${FLASHINFER_COMMIT} \
    VLLM_EXL3_EXT_PATH=/usr/local/lib/python3.12/dist-packages

COPY build/mimo_clean/provenance.json /opt/mimo-runner/provenance.json

RUN --mount=type=bind,source=build/mimo_clean/artifacts,target=/mnt/artifacts,readonly \
    set -eux; \
    cd /mnt/artifacts; \
    sha256sum -c artifact-sha256.txt; \
    cd /usr/local/lib/python3.12/dist-packages; \
    find vllm -type f \( -name '*.so' -o -name 'vllm-rs' \) -print0 \
      | sort -z \
      | tar --null -T - -cf /tmp/vllm-base-native.tar; \
    uv pip uninstall --system vllm; \
    uv pip install --system --no-deps \
      /mnt/artifacts/vllm-0.30.0+51b74d6d8-cp312-cp312-linux_aarch64.whl; \
    tar -xf /tmp/vllm-base-native.tar -C /usr/local/lib/python3.12/dist-packages; \
    install -m 0755 /mnt/artifacts/_C_stable_libtorch.abi3.so \
      /usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so; \
    uv pip uninstall --system \
      flashinfer-python flashinfer-jit-cache flashinfer-cubin \
      nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base \
      nvidia-cutlass-dsl-libs-core nvidia-cutlass-dsl-libs-cu12 \
      nvidia-cutlass-dsl-libs-cu13 || true; \
    uv pip install --system --no-deps \
      /mnt/artifacts/nvidia_cutlass_dsl_libs_base-4.7.1-cp312-cp312-manylinux_2_28_aarch64.whl \
      /mnt/artifacts/nvidia_cutlass_dsl_libs_core-4.7.1-py3-none-any.whl \
      /mnt/artifacts/nvidia_cutlass_dsl_libs_cu12-4.7.1-cp312-cp312-manylinux_2_28_aarch64.whl \
      /mnt/artifacts/nvidia_cutlass_dsl_libs_cu13-4.7.1-cp312-cp312-manylinux_2_28_aarch64.whl \
      /mnt/artifacts/nvidia_cutlass_dsl-4.7.1-py3-none-any.whl \
      /mnt/artifacts/flashinfer_python-0.7.0-py3-none-any.whl \
      /mnt/artifacts/exllamav3_ext_vllm-0.0.43-cp312-cp312-linux_aarch64.whl \
      /mnt/artifacts/sparkinfer-1.0.1-py3-none-any.whl; \
    install -D -m 0755 /mnt/artifacts/flashinfer-topk-sm121.so \
      /usr/local/lib/python3.12/dist-packages/flashinfer/data/aot/topk/topk.so; \
    find /usr/local/lib/python3.12/dist-packages/vllm -type d \
      \( -name tests -o -name __pycache__ \) -prune -exec rm -rf '{}' +; \
    find /usr/local/lib/python3.12/dist-packages \
      -type d -name __pycache__ -prune -exec rm -rf '{}' +; \
    rm -f /tmp/vllm-base-native.tar; \
    mkdir -p /opt/mimo-runner; \
    find /usr/local/lib/python3.12/dist-packages/vllm -type f -print0 \
      | sort -z \
      | xargs -0 sha256sum > /opt/mimo-runner/vllm-files.sha256; \
    find /usr/local/lib/python3.12/dist-packages/vllm -type f \
      \( -name '*.so' -o -name 'vllm-rs' \) -print0 \
      | sort -z \
      | xargs -0 sha256sum > /opt/mimo-runner/native-files.sha256; \
    uv cache clean

LABEL org.opencontainers.image.revision="${VLLM_COMMIT}" \
      org.opencontainers.image.version="${RUNNER_VERSION}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      io.mimo.runner.exllamav3.revision="${EXLLAMAV3_COMMIT}" \
      io.mimo.runner.flashinfer.revision="${FLASHINFER_COMMIT}" \
      io.mimo.runner.sparkinfer.revision="${SPARKINFER_COMMIT}" \
      io.mimo.runner.clean-build="true"
