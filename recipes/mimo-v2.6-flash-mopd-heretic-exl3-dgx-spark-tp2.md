# MiMo V2.6 Flash MOPD Heretic EXL3 on two DGX Spark systems

This recipe serves
[cbert33/MiMo-V2.6-Flash-MOPD-Heretic-Uncensored-EXL3-DGX-Sliced](https://huggingface.co/cbert33/MiMo-V2.6-Flash-MOPD-Heretic-Uncensored-EXL3-DGX-Sliced)
with tensor parallelism across two NVIDIA DGX Spark systems.

## Qualified configuration

| Setting | Value |
| --- | --- |
| Target weights | Rank-sliced EXL3, TP2 |
| Draft method | DFlash, probabilistic sampling |
| Draft depth | K=4 |
| Target KV cache | FP8 E4M3 |
| Draft KV cache | BF16 |
| Attention backend | `TRITON_ATTN_DIFFKV` |
| Context limit | 800,000 tokens |
| Maximum sequences | 2 |
| Asynchronous scheduling | Disabled |
| Qualified request types | Text, reasoning, and structured tools |

Release 8 does not qualify asynchronous MiMo DFlash. Keep
`--no-async-scheduling` in the launch command. Audio and multimodal serving
were not requalified with this configuration.

## Prepare both nodes

Build the Release 8 image with the clean image recipe in
[`build/mimo_clean`](../build/mimo_clean), then transfer the exact image to
both nodes. Download the linked model release and identify these two local
directories:

- the rank-sliced EXL3 target checkpoint;
- the matching DFlash checkpoint included with the release.

The MiMo repository may require Hugging Face authentication when its access is
restricted.

Create a persistent cache directory on each node. Resolve the fabric interface,
RoCE device, GID index, fabric CIDR, and each node's fabric address from the
current system.

Set these variables on each node:

```bash
export RUNNER_IMAGE='<release-8-image>'
export TARGET_MODEL='<absolute-path-to-mimo-exl3-target>'
export DRAFT_MODEL='<absolute-path-to-mimo-dflash-checkpoint>'
export CACHE_ROOT='<absolute-path-to-persistent-cache>'
export NODE_RANK='<0-or-1>'
export HOST_IP='<this-node-fabric-address>'
export MASTER_ADDR='<rank-0-fabric-address>'
export MASTER_PORT='29525'
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

docker run -d --name vllm-mimo-v26 \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1:-1 \
  --device /dev/infiniband:/dev/infiniband \
  -v "$TARGET_MODEL:/models/target:ro" \
  -v "$DRAFT_MODEL:/models/dflash:ro" \
  -v "$CACHE_ROOT/huggingface:/root/.cache/huggingface" \
  -v "$CACHE_ROOT/flashinfer:/root/.cache/flashinfer" \
  -v "$CACHE_ROOT/vllm:/root/.cache/vllm" \
  -v "$CACHE_ROOT/triton:/root/.triton" \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e FLASHINFER_WORKSPACE_BASE=/root/.cache/flashinfer \
  -e HF_HOME=/root/.cache/huggingface \
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
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_EXL3_EXT_PATH=/usr/local/lib/python3.12/dist-packages \
  -e VLLM_HOST_IP="$HOST_IP" \
  "$RUNNER_IMAGE" \
  /models/target \
  --served-model-name mimo-v2.6-flash-mopd-heretic-exl3 \
  --host 0.0.0.0 \
  --port 8320 \
  --trust-remote-code \
  --quantization exl3 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 800000 \
  --max-num-seqs 2 \
  --block-size 16 \
  --mm-processor-cache-gb 0.5 \
  --load-format instanttensor \
  --max-num-batched-tokens 4096 \
  --speculative-config '{"model":"/models/dflash","method":"dflash","num_speculative_tokens":4,"draft_tensor_parallel_size":2,"kv_cache_dtype":"bfloat16","draft_sample_method":"probabilistic"}' \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend TRITON_ATTN_DIFFKV \
  --kv-cache-memory-bytes 7100000000 \
  --skip-mm-profiling \
  --tool-call-parser mimo \
  --enable-auto-tool-choice \
  --reasoning-parser mimo \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --no-async-scheduling \
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
4. Confirm `async_scheduling` resolves to `False`.
5. Check `/v1/models`, then send one reasoning request and one required-tool
   request through the intended client route.
6. Check DFlash acceptance by draft position under the intended workload.

The Release 8 qualification covers the complete FP8 runner matrix, the original
62,287-token agentic request, concurrent long requests, and a tool-result
continuation. Broader production soak testing remains in progress.
