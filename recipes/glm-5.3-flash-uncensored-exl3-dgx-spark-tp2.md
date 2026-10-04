# GLM-5.3 Flash Uncensored EXL3 on two DGX Spark systems

This recipe serves
[cbert33/GLM-5.3-Flash-Uncensored-EXL3-DGX-Sliced](https://huggingface.co/cbert33/GLM-5.3-Flash-Uncensored-EXL3-DGX-Sliced)
with tensor parallelism across two NVIDIA DGX Spark systems.

## Qualified configuration

| Setting | Value |
| --- | --- |
| Target weights | Rank-sliced EXL3, TP2 |
| Draft method | DFlash2 |
| Draft depth | K=7 |
| Target KV cache | FP8 DS MLA |
| Draft KV cache | FP8 E4M3 |
| Attention backend | `FLASHINFER_MLA_SPARSE_SM120` |
| KDA prefill backend | `flashkda` |
| Context limit | 800,000 tokens |
| Maximum sequences | 3 |
| Qualified request types | Chat, reasoning, structured tools, and multimodal input |

K=7 is the qualified GLM configuration and matches the checkpoint's trained
maximum. Select a lower draft depth only after measuring end-to-end throughput
for the intended workload.

## Prepare both nodes

Follow [the reproducible build guide](../build/README.md). The build directory
contains a pinned downloader for the target and DFlash artifacts:

```bash
build/download_models.sh /srv/glm53/models
```

Transfer the exact runner image and downloaded model directories to both
nodes. Resolve the fabric interface, RoCE device, GID index, fabric CIDR, and
each node's fabric address from the current system.

Set these variables on each node:

```bash
export RUNNER_IMAGE='<release-7-image>'
export TARGET_MODEL='<absolute-path-to-glm-exl3-target>'
export DRAFT_MODEL='<absolute-path-to-glm-dflash2-checkpoint>'
export CACHE_ROOT='<absolute-path-to-persistent-cache>'
export NODE_RANK='<0-or-1>'
export HOST_IP='<this-node-fabric-address>'
export MASTER_ADDR='<rank-0-fabric-address>'
export MASTER_PORT='29521'
export FABRIC_IFACE='<connectx-ethernet-interface>'
export ROCE_HCA='<roce-device>'
export ROCE_GID_INDEX='<gid-index>'
export FABRIC_CIDR='<fabric-cidr>'
```

## Launch

Run the following command on both nodes. Rank 1 adds `--headless`
automatically.

```bash
set -euo pipefail

headless_args=()
if [[ "$NODE_RANK" == "1" ]]; then
  headless_args+=(--headless)
fi

docker run -d --name vllm-glm53 \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 32g \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1:-1 \
  --device /dev/infiniband:/dev/infiniband \
  -v "$TARGET_MODEL:/models/target:ro" \
  -v "$DRAFT_MODEL:/models/dflash2:ro" \
  -v "$CACHE_ROOT:/cache" \
  -v "$CACHE_ROOT/jit/b12x:/root/.cache/b12x" \
  -v "$CACHE_ROOT/jit/flashinfer:/root/.cache/flashinfer" \
  -v "$CACHE_ROOT/jit/vllm:/root/.cache/vllm" \
  -v "$CACHE_ROOT/jit/triton:/root/.triton" \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_CROSS_NIC=0 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_DEBUG=WARN \
  -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE="$FABRIC_CIDR" \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_GID_INDEX="$ROCE_GID_INDEX" \
  -e NCCL_IB_HCA="$ROCE_HCA" \
  -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_MAX_NCHANNELS=8 \
  -e NCCL_MIN_NCHANNELS=8 \
  -e NCCL_NET=IB \
  -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_SOCKET_IFNAME="$FABRIC_IFACE" \
  -e GLOO_SOCKET_IFNAME="$FABRIC_IFACE" \
  -e TP_SOCKET_IFNAME="$FABRIC_IFACE" \
  -e MN_IF_NAME="$FABRIC_IFACE" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  -e VLLM_DFLASH_FP8_DRAFT_HEAD=1 \
  -e VLLM_DFLASH_KV_RING=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_HOST_IP="$HOST_IP" \
  "$RUNNER_IMAGE" \
  /models/target \
  --served-model-name glm-5.3-flash \
  --host 0.0.0.0 \
  --port 8320 \
  --trust-remote-code \
  --quantization exl3 \
  --chat-template /opt/glm53/chat_template.jinja \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 800000 \
  --max-num-seqs 3 \
  --block-size 2304 \
  --mm-processor-cache-gb 0.5 \
  --load-format instanttensor \
  --max-num-batched-tokens 16384 \
  --speculative-config '{"method":"dflash","model":"/models/dflash2","num_speculative_tokens":7,"kv_cache_dtype":"fp8_e4m3"}' \
  --kv-cache-dtype fp8_ds_mla \
  --kv-cache-dtype-skip-layers sliding_window \
  --attention-backend FLASHINFER_MLA_SPARSE_SM120 \
  --kda-prefill-backend flashkda \
  --linear-backend b12x \
  --kv-cache-memory-bytes 8900000000 \
  --prefix-match-unit 512 \
  --skip-mm-profiling \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"high"}' \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  "${headless_args[@]}"
```

## Start order and checks

1. Start rank 1 and wait for its distributed worker to initialize.
2. Start rank 0.
3. Confirm both logs report the same model, image revision, and runtime flags.
4. Confirm the FlashInfer sparse MLA, FlashKDA, B12X, and EXL3 backends load.
5. Check `/v1/models`, then send chat, reasoning, and required-tool requests.
6. Check prefix-cache reuse and DFlash acceptance under the intended workload.

The retained Release 5 production sample observed 32.5 percent overall
draft-token acceptance with K=7. See the main README for the full workload and
throughput measurements.
