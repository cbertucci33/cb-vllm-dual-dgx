# syntax=docker/dockerfile:1.7
ARG CUDA_DEVEL_IMAGE
ARG BASE_IMAGE

FROM ${CUDA_DEVEL_IMAGE} AS cuda_devel
FROM ${BASE_IMAGE}

# The official vLLM runtime contains nvcc for JIT compilation but omits CUDA
# development headers such as cusparse.h. Copy the matching CUDA 13.0.2
# toolchain into the isolated builder only. Runtime images still start from the
# untouched official vLLM base.
COPY --from=cuda_devel /usr/local/cuda-13.0/ /usr/local/cuda-13.0/
