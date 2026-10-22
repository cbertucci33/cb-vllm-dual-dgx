#!/usr/bin/env bash
# Review candidate only: NOT installed, NOT launched, memory admission pending.
# KV bytes derive 850012 source-reported tokens for LBNHC/block16/batch4096.
# Verify materialized group layout and full-process memory before approving use.
set -euo pipefail
if [[ $# != 7 ]]; then
  echo "usage: $0 IMAGE_DIGEST NODE_RANK HOST_IP FABRIC_IFACE IB_HCA IB_GID SOURCE_COMMIT" >&2
  exit 2
fi
image=$1
node_rank=$2
host_ip=$3
fabric_iface=$4
ib_hca=$5
ib_gid=$6
source_commit=$7
[[ $image =~ ^sha256:[0-9a-f]{64}$ && $source_commit =~ ^[0-9a-f]{40}$ ]]
[[ $node_rank == 0 || $node_rank == 1 ]]
headless_args=()
if [[ $node_rank == 1 ]]; then headless_args+=(--headless); fi
cache_root="/opt/project/mimo-v032-cache/$source_commit"
docker run -d --name vllm_mimo_v032_guard_removed \
  --gpus all --network host --ipc host --ulimit memlock=-1:-1 \
  -v "/opt/project/evidence/vllm-v032-migration-20261019/mimo-qualification-preparation/guard-removal/mem_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/utils/mem_utils.py:ro" \
  --device /dev/infiniband:/dev/infiniband --restart no \
  -v "$cache_root/b12x:/root/.cache/b12x" \
  -v "$cache_root/flashinfer:/root/.cache/flashinfer" \
  -v "$cache_root/vllm:/root/.cache/vllm" \
  -v "$cache_root/triton:/root/.triton" \
  -v "$cache_root/huggingface:/root/.cache/huggingface" \
  -v /opt/project/models/mimo-heretic-v031-mm-b12f5573-targetkv-cdb36376ae25-20261013:/models/mimo-exl3:ro \
  -v /opt/project/models/MiMo-DFlash-BF16-FP8KV-d6a1db9110-cd30a6f5-20261013:/models/mimo-dflash:ro \
  -e NCCL_DEBUG=WARN -e NCCL_IB_DISABLE=0 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_MIN_NCHANNELS=8 -e NCCL_MAX_NCHANNELS=8 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_NET=IB -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_SOCKET_IFNAME="$fabric_iface" -e GLOO_SOCKET_IFNAME="$fabric_iface" \
  -e TP_SOCKET_IFNAME="$fabric_iface" -e MN_IF_NAME="$fabric_iface" \
  -e NCCL_IB_HCA="$ib_hca" -e NCCL_IB_GID_INDEX="$ib_gid" \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE=10.100.40.0/24 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e FLASHINFER_WORKSPACE_BASE=/root/.cache/flashinfer \
  -e HF_HOME=/root/.cache/huggingface -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_HOST_IP="$host_ip" -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_DFLASH_KV_RING=1 \
  "$image" /models/mimo-exl3 \
  --served-model-name mimo-v2.6-flash-mopd-heretic-exl3 \
  --host 0.0.0.0 --port 8320 --trust-remote-code \
  --quantization exl3 --dtype bfloat16 --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 --kv-cache-memory-bytes 9000000000 \
  --max-model-len 600000 --max-num-seqs 2 --block-size 16 \
  --mm-processor-cache-gb 0.5 --load-format instanttensor \
  --max-num-batched-tokens 4096 \
  --speculative-config '{"model":"/models/mimo-dflash","method":"dflash","num_speculative_tokens":7,"draft_tensor_parallel_size":2,"kv_cache_dtype":"fp8_e4m3","draft_sample_method":"probabilistic"}' \
  --kv-cache-dtype fp8_e4m3 --attention-backend TRITON_ATTN_DIFFKV --linear-backend b12x \
  --enable-prefix-caching --enable-chunked-prefill \
  --tool-call-parser mimo --enable-auto-tool-choice --reasoning-parser mimo \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --override-generation-config '{"temperature":1.0,"top_p":0.95}' \
  --no-async-scheduling \
  --distributed-executor-backend mp --nnodes 2 --node-rank "$node_rank" \
  --master-addr 10.100.40.1 --master-port 29525 "${headless_args[@]}"
