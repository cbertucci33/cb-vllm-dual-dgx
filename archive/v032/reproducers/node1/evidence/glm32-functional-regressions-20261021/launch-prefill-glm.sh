#!/usr/bin/env bash
set -euo pipefail

rank="${1:?usage: $0 <rank:0|1> <expected-image-id>}"
expected_image="${2:?expected candidate image ID required}"

case "$rank" in
  0)
    expected_host="node-1"
    host_ip="127.0.0.1"
    extra_args=()
    ;;
  1)
    expected_host="node-2"
    host_ip="127.0.0.1"
    extra_args=(--headless)
    ;;
  *)
    echo "rank must be 0 or 1" >&2
    exit 2
    ;;
esac

if [[ $(hostname) != "$expected_host" ]]; then
  echo "host/rank mismatch: rank $rank belongs on $expected_host" >&2
  exit 1
fi
if [[ ! $expected_image =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "expected image ID must be a full sha256 ID" >&2
  exit 2
fi

image="$expected_image"
container="vllm_glm53_v032_prefillfix_20261021"
reference_container="vllm_glm53_v030_fast_9707345"

actual_image=$(docker image inspect "$image" --format '{{.Id}}')
if [[ $actual_image != "$expected_image" ]]; then
  echo "candidate image mismatch: $actual_image, expected $expected_image" >&2
  exit 1
fi
source_revision=$(docker image inspect "$image" \
  --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')
if [[ ! $source_revision =~ ^[0-9a-f]{40}$ ]]; then
  echo "candidate image has no full source-revision label: $source_revision" >&2
  exit 1
fi
cache_root="/opt/project/glm53-vllm-cache-v032-${source_revision:0:10}"
reference_running=$(docker inspect "$reference_container" \
  --format '{{.State.Running}}')
if [[ $reference_running != "false" ]]; then
  echo "v0.30 definition reference is running; refusing launch" >&2
  exit 1
fi
if docker container inspect "$container" >/dev/null 2>&1; then
  echo "candidate container already exists; refusing to reuse or replace it" >&2
  exit 1
fi
other_models=$(docker ps --format '{{.Names}}' | grep '^vllm_' || true)
if [[ -n $other_models ]]; then
  echo "a model container is already running:" >&2
  echo "$other_models" >&2
  exit 1
fi
cuda_owners=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  | sed '/^[[:space:]]*$/d')
if [[ -n $cuda_owners ]]; then
  echo "CUDA compute owners remain: $cuda_owners" >&2
  exit 1
fi
mem_available_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
if (( mem_available_kb < 122385528 )); then
  echo "MemAvailable=${mem_available_kb}kB; stopped/reclaim gate has not passed" >&2
  exit 1
fi

install -d \
  "$cache_root/huggingface" \
  "$cache_root/jit/b12x" \
  "$cache_root/jit/flashinfer" \
  "$cache_root/jit/vllm" \
  "$cache_root/jit/triton"

candidate_id=$(docker create \
  --name "$container" \
  --user root \
  --gpus all \
  --device /dev/infiniband:/dev/infiniband \
  --network host \
  --ipc host \
  --security-opt label=disable \
  --ulimit memlock=-1:-1 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e HF_HOME=/cache/huggingface \
  -e MN_IF_NAME=enp1s0f0np0 \
  -e NCCL_CROSS_NIC=0 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_DEBUG=WARN \
  -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE=127.0.0.1/24 \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_HCA=rocep1s0f0 \
  -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_MAX_NCHANNELS=8 \
  -e NCCL_MIN_NCHANNELS=8 \
  -e NCCL_NET=IB \
  -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e TP_SOCKET_IFNAME=enp1s0f0np0 \
  -e VLLM_DFLASH_FP8_DRAFT_HEAD=1 \
  -e VLLM_DFLASH_KV_RING=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_HOST_IP="$host_ip" \
  -v /opt/project/models/glm53-exl3-tp2-rank-sliced:/models/glm53-exl3:ro \
  -v /opt/project/models/glm53-dflash2-mxfp8:/models/glm53-dflash2:ro \
  -v "$cache_root:/cache" \
  -v "$cache_root/jit/b12x:/root/.cache/b12x" \
  -v "$cache_root/jit/flashinfer:/root/.cache/flashinfer" \
  -v "$cache_root/jit/vllm:/root/.cache/vllm" \
  -v "$cache_root/jit/triton:/root/.triton" \
  "$image" \
  /models/glm53-exl3 \
  --served-model-name glm-5.3-flash \
  --host 0.0.0.0 \
  --port 8320 \
  --trust-remote-code \
  --quantization exl3 \
  --chat-template /models/glm53-exl3/chat_template.jinja \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 600000 \
  --max-num-seqs 3 \
  --block-size 2304 \
  --mm-processor-cache-gb 0.5 \
  --load-format instanttensor \
  --max-num-batched-tokens 16384 \
  --speculative-config '{"method":"dflash","model":"/models/glm53-dflash2","num_speculative_tokens":7,"kv_cache_dtype":"fp8_e4m3"}' \
  --kv-cache-dtype fp8_ds_mla \
  --attention-backend FLASHINFER_MLA_SPARSE_SM120 \
  --kda-prefill-backend flashkda \
  --linear-backend b12x \
  --kv-cache-memory-bytes 7300000000 \
  --prefix-match-unit 512 \
  --skip-mm-profiling \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"high"}' \
  --jit-monitor-verbose \
  --async-scheduling \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank "$rank" \
  --master-addr 127.0.0.1 \
  --master-port 29521 \
  "${extra_args[@]}")

if [[ $candidate_id != "$(docker inspect "$container" --format '{{.Id}}')" ]]; then
  echo "created candidate container ID mismatch" >&2
  exit 1
fi
echo "candidate created; definition verification and explicit start remain separate"
