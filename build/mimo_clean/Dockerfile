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
    uv pip uninstall --system vllm; \
    uv pip install --system --no-deps \
      /mnt/artifacts/vllm-0.30.0+5ab5a277e-cp312-cp312-linux_aarch64.whl; \
    uv pip install --system --no-deps \
      /mnt/artifacts/xgrammar-0.2.8-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl \
      /mnt/artifacts/nccl_extensions-0.1.0-cp312-cp312-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl; \
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
    mkdir -p /opt/mimo-runner; \
    find /usr/local/lib/python3.12/dist-packages/vllm -type f -print0 \
      | sort -z \
      | xargs -0 sha256sum > /opt/mimo-runner/vllm-files.sha256; \
    find /usr/local/lib/python3.12/dist-packages/vllm -type f \
      \( -name '*.so' -o -name 'vllm-rs' \) -print0 \
      | sort -z \
      | xargs -0 sha256sum > /opt/mimo-runner/native-files.sha256; \
    python3 -c 'import importlib.metadata as m; assert m.version("xgrammar") == "0.2.8"; assert m.version("nccl-extensions") == "0.1.0"; from xgrammar import get_model_structural_tag; assert get_model_structural_tag(model="mimo", tools=[{"type":"function","function":{"name":"probe","parameters":{"type":"object","properties":{}}}}], tool_choice="auto", reasoning=True) is not None'; \
    uv pip freeze --system | sort > /opt/mimo-runner/python-packages.txt; \
    uv cache clean

LABEL org.opencontainers.image.revision="${VLLM_COMMIT}" \
      org.opencontainers.image.version="${RUNNER_VERSION}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      io.mimo.runner.exllamav3.revision="${EXLLAMAV3_COMMIT}" \
      io.mimo.runner.flashinfer.revision="${FLASHINFER_COMMIT}" \
      io.mimo.runner.sparkinfer.revision="${SPARKINFER_COMMIT}" \
      io.mimo.runner.clean-build="true"
