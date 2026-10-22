# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextvars
import functools
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import regex as re
import torch
from contract import validate_rows

_batch = contextvars.ContextVar("draft_capture_batch", default=None)
_context_rows = contextvars.ContextVar("draft_capture_context_rows", default=None)
_query_rows = contextvars.ContextVar("draft_capture_query_rows", default=None)

_phase = contextvars.ContextVar("draft_calibration_phase", default=None)
_counter = 0


def observe(layer, key, value, slots, marker, rank, root, phase):
    global _counter
    match = re.search(r"\.layers\.(\d+)\.self_attn\.attn$", layer.layer_name)
    if match is None or not 48 <= int(match[1]) <= 52:
        raise ValueError("Unexpected draft attention owner: " + layer.layer_name)
    if phase is None or phase[0] not in ("context", "query"):
        raise ValueError("Draft write lacks an explicit context/query owner")
    # Native cache insertion owns slots.numel() rows; trailing model padding
    # is not written and must not enter calibration.
    rows = slots.numel()
    if key.shape[0] < rows or value.shape[0] < rows:
        raise ValueError("Calibration cannot ignore missing cache-write rows")
    k = key[:rows].reshape(rows, 4, 128)
    v = value[:rows].reshape(rows, 4, 128)
    valid = slots.reshape(-1) >= 0
    k = k[valid].detach().to("cpu").contiguous()
    v = v[valid].detach().to("cpu").contiguous()
    if not k.numel():
        return
    if not torch.isfinite(k).all() or not torch.isfinite(v).all():
        raise ValueError("Fresh draft activation contains NaN/Inf")
    positions = phase[1]
    if positions.ndim != 1 or positions.numel() < rows:
        raise ValueError("Fresh draft positions disagree with cache-write rows")
    positions = positions[:rows][valid].detach().to("cpu")
    owners = _context_rows.get() if phase[0] == "context" else _query_rows.get()
    if owners is None:
        raise ValueError("Cache write lacks scheduled token/sequence provenance")
    token_ids, sequence_ids, boundaries = owners
    selected = valid.detach().to("cpu").nonzero().reshape(-1).tolist()
    token_ids = token_ids[:rows].detach().to("cpu")[selected].tolist()
    sequence_ids = [sequence_ids[i] for i in selected]
    actual_slots = slots.reshape(-1)[:rows][valid].detach().to("cpu").tolist()
    validate_rows(token_ids, positions.tolist(), sequence_ids, actual_slots)
    local_layer = int(match[1]) - 48
    _counter += 1
    name = (
        f"{marker['sample']}-rank{rank}-layer{local_layer}-{phase[0]}-"
        f"{os.getpid()}-{_counter}.pt"
    )
    path = root / name
    if path.exists():
        raise ValueError("Refusing to overwrite calibration capture")
    torch.save(
        {
            "key": k,
            "value": v,
            "positions": positions,
            "marker": marker,
            "rank": rank,
            "layer": local_layer,
            "phase": phase[0],
            "token_ids": token_ids,
            "sequence_ids": sequence_ids,
            "slots": actual_slots,
            "sequence_boundaries": boundaries,
            "provenance_version": 2,
        },
        path,
    )
    event = {
        "tensor": name,
        "rank": rank,
        "layer": local_layer,
        "phase": phase[0],
        "stage": marker["stage"],
        "sample": marker["sample"],
        "rows": k.shape[0],
        "k_absmax": k.float().abs().max().item(),
        "v_absmax": v.float().abs().max().item(),
        "positions_min": positions.min().item(),
        "positions_max": positions.max().item(),
        "request_sha256": marker["request_sha256"],
        "capture_id": marker["capture_id"],
        "runtime_contract_sha256": marker["runtime_contract_sha256"],
        "provenance_version": 2,
    }
    with (root / "events.jsonl").open("a") as out:
        out.write(json.dumps(event) + "\n")


def install():
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.model_executor.models.qwen3_dflash import (
        DFlashQwen3Attention,
        DFlashQwen3Model,
    )
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    root = Path(os.environ["MIMO_DFLASH_CALIBRATION_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    marker_file = root / "active.json"
    expected_rank = int(os.environ["MIMO_CALIBRATION_RANK"])
    capture_id = os.environ["MIMO_CALIBRATION_CAPTURE_ID"]
    runtime_hash = os.environ["MIMO_CALIBRATION_RUNTIME_SHA256"]
    if len(runtime_hash) != 64:
        raise ValueError("Exact runtime-definition SHA256 required")
    if expected_rank not in (0, 1) or not capture_id:
        raise ValueError("Explicit TP2 capture identity required")
    original_write = TritonAttentionImpl.do_kv_cache_update
    original_context = DFlashQwen3Model.precompute_and_store_context_kv
    original_query = DFlashQwen3Attention.forward

    def context(self, context_states, context_positions, context_slot_mapping=None):
        token = _phase.set(("context", context_positions))
        try:
            return original_context(
                self, context_states, context_positions, context_slot_mapping
            )
        finally:
            _phase.reset(token)

    def query(self, positions, hidden_states):
        token = _phase.set(("query", positions))
        try:
            return original_query(self, positions, hidden_states)
        finally:
            _phase.reset(token)

    def write(self, layer, key, value, kv_cache, slot_mapping):
        if marker_file.exists():
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Calibration requires diagnostic eager execution")
            rank = get_tensor_model_parallel_rank()
            if rank != expected_rank:
                raise ValueError("Calibration rank/host contract mismatch")
            marker = json.loads(marker_file.read_text())
            if (
                marker["capture_id"] != capture_id
                or marker["runtime_contract_sha256"] != runtime_hash
            ):
                raise ValueError("Stale calibration marker")
            observe(layer, key, value, slot_mapping, marker, rank, root, _phase.get())
        return original_write(self, layer, key, value, kv_cache, slot_mapping)

    DFlashQwen3Model.precompute_and_store_context_kv = context
    DFlashQwen3Attention.forward = query
    TritonAttentionImpl.do_kv_cache_update = write
    files = {}
    for owner in (
        DFlashQwen3Model,
        DFlashQwen3Attention,
        TritonAttentionImpl,
        DFlashSpeculator,
        GPUModelRunner,
    ):
        file = Path(inspect.getfile(owner))
        files[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    (root / f"instrumentation-{os.getpid()}.json").write_text(
        json.dumps(
            {
                "rank": expected_rank,
                "source": os.environ["MIMO_CALIBRATION_SOURCE"],
                "instrumented_source_files": files,
                "q_changes": False,
                "capture_boundary": (
                    "actual post-normalization/RoPE/V-multiplier cache write"
                ),
            },
            indent=2,
        )
    )

    original_propose = DFlashSpeculator.propose
    original_precompute = DFlashSpeculator._precompute_context_kv
    original_run = DFlashSpeculator._run_model
    original_load = GPUModelRunner.load_model
    propose_signature = inspect.signature(original_propose)

    @functools.wraps(original_propose)
    def propose(self, *args, **kwargs):
        if not marker_file.exists():
            return original_propose(self, *args, **kwargs)
        bound = propose_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        if bound.arguments["dummy_run"] or bound.arguments["is_profile"]:
            raise ValueError("Warmup/profile cannot enter active calibration")
        batch = bound.arguments["input_batch"]
        if batch.num_reqs != 1:
            raise ValueError("Calibration requires one isolated sequence")
        marker = json.loads(marker_file.read_text())
        if (
            marker["capture_id"] != capture_id
            or marker["runtime_contract_sha256"] != runtime_hash
        ):
            raise ValueError("Stale capture lease")
        inflight = root / f"inflight-{os.getpid()}.json"
        if inflight.exists():
            raise ValueError("Unfinished producer lease")
        inflight.write_text(json.dumps(marker))
        token = _batch.set(batch)
        try:
            result = original_propose(self, *args, **kwargs)
            receipt = dict(
                marker,
                rank=get_tensor_model_parallel_rank(),
                req_ids=batch.req_ids,
                status="producer_completed",
            )
            with (root / "producer-completions.jsonl").open("a") as out:
                out.write(json.dumps(receipt) + "\n")
            return result
        except BaseException:
            (root / f"producer-failed-{os.getpid()}.json").write_text(
                json.dumps(marker)
            )
            raise
        finally:
            _batch.reset(token)
            inflight.unlink()

    @functools.wraps(original_precompute)
    def precompute(self, start, end, dummy_run=False):
        batch = _batch.get()
        if batch is None:
            return original_precompute(self, start, end, dummy_run)
        boundaries = batch.query_start_loc_np.tolist()
        sequences = [
            req_id
            for i, req_id in enumerate(batch.req_ids)
            for _ in range(boundaries[i], boundaries[i + 1])
        ]
        if len(sequences) != batch.num_tokens:
            raise ValueError("Scheduled target rows disagree with boundaries")
        token = _context_rows.set(
            (
                batch.input_ids[start:end],
                sequences[start:end],
                dict(query_start_loc=boundaries, slice_start=start, slice_end=end),
            )
        )
        try:
            return original_precompute(self, start, end, dummy_run)
        finally:
            _context_rows.reset(token)

    @functools.wraps(original_run)
    def run(self, num_tokens, *args, **kwargs):
        batch = _batch.get()
        if batch is None:
            return original_run(self, num_tokens, *args, **kwargs)
        sequences = [
            req_id for req_id in batch.req_ids for _ in range(self.num_query_per_req)
        ]
        # Slot filtering excludes padding, which has no sequence owner.
        sequences += ["padding"] * (num_tokens - len(sequences))
        token = _query_rows.set(
            (
                self.input_buffers.input_ids[:num_tokens],
                sequences,
                dict(
                    query_start_loc=[
                        i * self.num_query_per_req for i in range(batch.num_reqs + 1)
                    ],
                    slice_start=0,
                    slice_end=num_tokens,
                ),
            )
        )
        try:
            return original_run(self, num_tokens, *args, **kwargs)
        finally:
            _query_rows.reset(token)

    @functools.wraps(original_load)
    def load(self, *args, **kwargs):
        config = self.vllm_config
        if (
            config.scheduler_config.async_scheduling is not False
            or not config.model_config.enforce_eager
            or config.scheduler_config.max_num_seqs != 1
        ):
            raise ValueError(
                "Capture requires the explicit isolated eager/no-async contract"
            )
        result = original_load(self, *args, **kwargs)
        rank = get_tensor_model_parallel_rank()
        if rank != expected_rank:
            raise ValueError("Loaded rank differs from capture contract")
        receipt = dict(
            rank=rank,
            capture_id=capture_id,
            runtime_contract_sha256=runtime_hash,
            pid=os.getpid(),
            capture_root=str(root),
            executable=sys.executable,
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            source=os.environ["MIMO_CALIBRATION_SOURCE"],
            instrumented_source_files=files,
            hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        path = root / "capture-ready.json"
        if path.exists():
            raise ValueError("Refusing stale readiness receipt")
        path.write_text(json.dumps(receipt, indent=2))
        return result

    DFlashSpeculator.propose = propose
    DFlashSpeculator._precompute_context_kv = precompute
    DFlashSpeculator._run_model = run
    GPUModelRunner.load_model = load
