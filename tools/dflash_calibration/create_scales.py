# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from contract import validate_events, validate_rows
from safetensors.torch import save_file

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--source", required=True)
parser.add_argument("--runtime-contract", type=Path, required=True)
parser.add_argument("--capture-id", required=True)
args = parser.parse_args()
root = args.root
runtime_hash = hashlib.sha256(args.runtime_contract.read_bytes()).hexdigest()
rows = []
for rank in (0, 1):
    path = root / "capture" / f"node-{rank + 1}" / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    if not events or not all(e["rank"] == rank for e in events):
        raise ValueError("Empty/wrong-rank event stream")
    for event in events:
        event["_path"] = path.parent / event["tensor"]
        rows.append(event)
samples = json.loads((root / "requests-text-only.json").read_text())
validate_events(rows, samples)
for rank in (0, 1):
    capture = root / "capture" / f"node-{rank + 1}"
    if (capture / "active.json").exists() or list(capture.glob("inflight-*.json")):
        raise ValueError("Capture still active; producer completion is required")
    if list(capture.glob("producer-failed-*.json")):
        raise ValueError("Producer failure invalidates this capture")
    ready = json.loads((capture / "capture-ready.json").read_text())
    if (
        ready["rank"] != rank
        or ready["source"] != args.source
        or ready["capture_id"] != args.capture_id
        or ready["runtime_contract_sha256"] != runtime_hash
    ):
        raise ValueError("Stale capture readiness identity")
    for sample in samples:
        finished = json.loads(
            (capture / ("finished-" + sample["sample"] + ".json")).read_text()
        )
        if finished["capture_id"] != args.capture_id:
            raise ValueError("Stale sample completion")
for row in rows:
    if (
        row["capture_id"] != args.capture_id
        or row["runtime_contract_sha256"] != runtime_hash
    ):
        raise ValueError("Stale tensor event")
    tensor = torch.load(row["_path"], map_location="cpu", weights_only=True)
    marker = tensor["marker"]
    if (
        marker["capture_id"] != args.capture_id
        or marker["sample"] != row["sample"]
        or marker["request_sha256"] != row["request_sha256"]
    ):
        raise ValueError("Tensor/event identity mismatch")
    validate_rows(
        tensor["token_ids"],
        tensor["positions"].tolist(),
        tensor["sequence_ids"],
        tensor["slots"],
    )
    if (
        len(tensor["token_ids"]) != row["rows"]
        or tensor["rank"] != row["rank"]
        or tensor["layer"] != row["layer"]
        or tensor["phase"] != row["phase"]
    ):
        raise ValueError("Tensor/event row ownership mismatch")
    boundaries = tensor.get("sequence_boundaries")
    if not boundaries or not boundaries.get("query_start_loc"):
        raise ValueError("Scheduled sequence boundaries are missing")
    if any(tensor[key].shape != (row["rows"], 4, 128) for key in ("key", "value")):
        raise ValueError("Draft TP2 K/V geometry differs from capture contract")
    for kind, key in (("k", "key"), ("v", "value")):
        x = tensor[key].float()
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite captured tensor")
        if x.abs().max().item() != row[kind + "_absmax"]:
            raise ValueError("Recorded maximum differs from actual tensor")
coverage = {}
scales = {}
for layer in range(5):
    group = [r for r in rows if r["layer"] == layer]
    observed = {(r["rank"], r["phase"]) for r in group}
    if observed != {(0, "context"), (0, "query"), (1, "context"), (1, "query")}:
        raise ValueError(f"Missing rank/phase for layer {layer}")
    training = [r for r in group if r["stage"] == "calibration"]
    held = [r for r in group if r["stage"] == "heldout"]
    if not training or not held:
        raise ValueError("Both calibration and heldout stages are required")
    coverage[str(layer)] = {
        "phase_rank_coverage": sorted(observed),
        "observations": len(group),
        "rows": sum(r["rows"] for r in group),
    }
    for kind in ("k", "v"):
        maximum = max(r[kind + "_absmax"] for r in training)
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("Scale requires a positive finite observed maximum")
        value = np.nextafter(np.float32(maximum / 448.0), np.float32(np.inf)).item()
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid FP32 scale")
        scales[f"layers.{layer}.self_attn.attn.{kind}_scale"] = torch.tensor(
            value, dtype=torch.float32
        )
        coverage[str(layer)][kind] = {
            "calibration_absmax": maximum,
            "scale": value,
            "heldout_absmax": max(r[kind + "_absmax"] for r in held),
        }
metrics = {}
for layer in range(5):
    for kind, key in (("k", "key"), ("v", "value")):
        scale = scales[f"layers.{layer}.self_attn.attn.{kind}_scale"].item()
        overflow = 0
        count = 0
        error2 = 0.0
        signal2 = 0.0
        maxerror = 0.0
        for row in rows:
            if row["layer"] != layer or row["stage"] != "heldout":
                continue
            x = torch.load(row["_path"], map_location="cpu", weights_only=True)[
                key
            ].float()
            normalized = x / scale
            overflow += int((normalized.abs() > 448).sum())
            count += x.numel()
            restored = (
                normalized.clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
            )
            if not torch.isfinite(restored).all():
                raise ValueError("Nonfinite heldout quantization")
            difference = restored.double() - x.double()
            error2 += difference.square().sum().item()
            signal2 += x.double().square().sum().item()
            maxerror = max(maxerror, difference.abs().max().item())
        metrics[f"{layer}_{kind}"] = {
            "values": count,
            "out_of_range": overflow,
            "relative_l2": (math.sqrt(error2 / signal2) if signal2 else 0.0),
            "maximum_absolute_error": maxerror,
        }
if any(m["values"] <= 0 or m["out_of_range"] for m in metrics.values()):
    raise ValueError("Heldout range validation failed; no scale artifact exported")
if any(
    not math.isfinite(value) for metric in metrics.values() for value in metric.values()
):
    raise ValueError("Nonfinite heldout metric; no artifact exported")
out = root / "scale-candidate"
out.mkdir(exist_ok=False)
save_file(scales, str(out / "kv_cache_scales.safetensors"))
provenance = {
    "source_commit": args.source,
    "capture_id": args.capture_id,
    "runtime_contract_sha256": runtime_hash,
    "target": "mimo-heretic-v031-mm-b12f5573",
    "draft": "MiMo-V2.6-Flash-MOPD-DFlash-BF16-TP2-20261002",
    "calibration": (
        "fresh actual in-house cache writes; recorded runtime contract "
        "determines dtype; context and query on TP2"
    ),
    "formula": "nextafter_float32(calibration_absmax/448,+infinity)",
    "margin": (
        "one upward float32 rounding step to avoid arithmetic undercoverage; "
        "no inherited 1.1 factor"
    ),
    "Q_changes": False,
    "old_captures_used": False,
    "scale_count": len(scales),
    "coverage": coverage,
    "heldout_quantization_metrics": metrics,
    "heldout_range_pass": all(m["out_of_range"] == 0 for m in metrics.values()),
    "native_FP8_reload": "not executed",
    "requests_sha256": hashlib.sha256(
        (root / "requests-text-only.json").read_bytes()
    ).hexdigest(),
}
(out / "KV_CACHE_CALIBRATION.json").write_text(
    json.dumps(provenance, indent=2, allow_nan=False)
)
print(
    json.dumps(
        {
            "scales": len(scales),
            "heldout_range_pass": provenance["heldout_range_pass"],
            "metrics": metrics,
        },
        indent=2,
    )
)
