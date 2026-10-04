# Model recipes

These recipes reproduce the qualified two-node NVIDIA DGX Spark configurations
for the two model releases supported by this repository:

- [MiMo V2.6 Flash MOPD Heretic EXL3, TP2](mimo-v2.6-flash-mopd-heretic-exl3-dgx-spark-tp2.md)
- [GLM-5.3 Flash Uncensored EXL3, TP2](glm-5.3-flash-uncensored-exl3-dgx-spark-tp2.md)

Each recipe uses one GB10 GPU per node and tensor parallelism across the two
nodes. The commands leave host-specific values as environment variables. Do
not copy interface names, RoCE devices, GID indices, addresses, or model paths
from another cluster.

Both ranks require the same runner image digest, source revision, target
checkpoint, draft checkpoint, and launch flags. Start rank 1 with `--headless`
before starting rank 0. Verify both rank logs before sending a request.

The example servers bind to `0.0.0.0`. Restrict the service port to a trusted
network or place an authenticated proxy in front of it before allowing remote
client access.

The recipes describe the configurations that were qualified for Release 7.
They are baselines, not universal capacity recommendations. Reduce context,
batch size, or concurrency when the available memory or workload differs.
