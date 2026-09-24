# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG VLLM_COMMIT
ARG FLASHINFER_COMMIT
ARG SPARKINFER_COMMIT
ARG EXLLAMAV3_COMMIT
ARG B12X_COMMIT

ENV FLASHINFER_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer/${FLASHINFER_COMMIT} \
    VLLM_EXL3_EXT_PATH=/usr/local/lib/python3.12/dist-packages

COPY artifacts/ /opt/glm53-runner-artifacts/
COPY vllm-overlay/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY verify_runtime.py /opt/verify_runtime.py
COPY provenance.json /opt/glm53-runner-provenance.json
COPY chat_template.jinja /opt/glm53/chat_template.jinja

RUN cd /opt/glm53-runner-artifacts \
    && sha256sum -c artifact-sha256.txt \
    && install -m 0755 ./_C_stable_libtorch*.so \
       /usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so \
    && /usr/bin/python3 -m pip install --no-deps --force-reinstall ./*.whl \
    && install -m 0755 ./flashinfer-topk-sm121.so \
       /usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/topk/topk.so \
    && rm -rf \
       /usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120 \
    && /usr/bin/python3 /opt/verify_runtime.py --build-only \
    && rm -rf /root/.cache/pip

LABEL org.opencontainers.image.source="https://github.com/cbertucci33/vllm-v29-glm53flash-exl3-dgx" \
      org.opencontainers.image.revision="${VLLM_COMMIT}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      io.glm53runner.flashinfer.revision="${FLASHINFER_COMMIT}" \
      io.glm53runner.sparkinfer.revision="${SPARKINFER_COMMIT}" \
      io.glm53runner.exllamav3.revision="${EXLLAMAV3_COMMIT}" \
      io.glm53runner.b12x.revision="${B12X_COMMIT}"
