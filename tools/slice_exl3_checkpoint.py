#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create a tensor-parallel rank-sliced MCG EXL3 checkpoint.

The converter preserves the source checkpoint and writes a new sibling tree.
Only routed-expert EXL3 tensors are transformed. Quantized Trellis payloads are
sliced without dequantization or requantization; Hadamard vectors and codebook
markers are sliced or replicated according to the projection's TP axis.

The source must be a GLM-5.3 Flash checkpoint directory with the standard
routed-expert tensor names, an integral 3/4/5/6 bitrate, and the MCG codebook.
The rank-sliced runtime does not support the ``mul1`` codebook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPERT_TENSOR_RE = re.compile(
    r"^(?P<prefix>(?:model\.)?(?:language_model\.)?layers\.(?P<layer>\d+)"
    r"\.mlp\.experts\.(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj))"
    r"\.(?P<field>trellis|suh|svh|mcg|mul1)$"
)
RANK_SLICED_SCHEMA = (
    "model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}"
)
REQUIRED_FIELDS = frozenset({"trellis", "suh", "svh", "mcg"})
MCG_SENTINEL = 0xCBAC1FED


@dataclass(frozen=True)
class TensorRule:
    split_dim: int | None


def tensor_rule(projection: str, field: str) -> TensorRule:
    """Return the lossless TP transform for one routed-expert component."""
    if field == "mul1":
        raise ValueError("rank-sliced EXL3 requires the MCG codebook")
    if field == "mcg":
        return TensorRule(None)
    if projection in {"gate_proj", "up_proj"}:
        return TensorRule(1 if field == "trellis" else (0 if field == "svh" else None))
    if projection == "down_proj":
        return TensorRule(0 if field in {"trellis", "suh"} else None)
    raise ValueError(f"unsupported projection: {projection}")


def ranked_name(prefix: str, rank: int, field: str) -> str:
    return f"{prefix}.rank{rank}.{field}"


def split_tensor(tensor: torch.Tensor, *, dim: int, tp: int) -> list[torch.Tensor]:
    if dim >= tensor.ndim:
        raise ValueError(f"cannot split shape {tuple(tensor.shape)} on dimension {dim}")
    extent = int(tensor.shape[dim])
    if extent % tp:
        raise ValueError(
            f"shape {tuple(tensor.shape)} is not divisible by TP={tp} on dim={dim}"
        )
    width = extent // tp
    return [tensor.narrow(dim, rank * width, width).contiguous() for rank in range(tp)]


def transform_tensor(
    name: str, tensor: torch.Tensor, *, tp: int
) -> dict[str, torch.Tensor]:
    match = EXPERT_TENSOR_RE.fullmatch(name)
    if match is None:
        return {name: tensor}
    rule = tensor_rule(match.group("projection"), match.group("field"))
    pieces = (
        [tensor.clone() for _ in range(tp)]
        if rule.split_dim is None
        else split_tensor(tensor, dim=rule.split_dim, tp=tp)
    )
    return {
        ranked_name(match.group("prefix"), rank, match.group("field")): piece
        for rank, piece in enumerate(pieces)
    }


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_expert_tensor_abi(
    name: str, tensor: torch.Tensor, *, field: str, bits: int
) -> None:
    if field == "trellis":
        if tensor.dtype != torch.int16 or tensor.ndim != 3:
            raise ValueError(f"{name} must be rank-3 int16")
        if tensor.shape[2] != 16 * bits:
            raise ValueError(
                f"{name} has trellis width {tensor.shape[2]}, expected {16 * bits}"
            )
        return
    if field in {"suh", "svh"}:
        if tensor.dtype != torch.float16 or tensor.ndim != 1:
            raise ValueError(f"{name} must be rank-1 float16")
        return
    if field == "mcg":
        if tensor.dtype != torch.int32 or tensor.numel() != 1:
            raise ValueError(f"{name} must be a scalar int32 sentinel")
        value = int(tensor.reshape(()).item()) & 0xFFFFFFFF
        if value != MCG_SENTINEL:
            raise ValueError(
                f"{name} has sentinel 0x{value:08x}, expected 0x{MCG_SENTINEL:08x}"
            )


def inspect_source(source: Path, *, tp: int) -> dict[str, Any]:
    """Validate the source ABI and calculate exact output tensor bytes."""
    if tp < 2:
        raise ValueError("TP must be at least 2")
    index_path = source / "model.safetensors.index.json"
    config_path = source / "config.json"
    quant_path = source / "quantization_config.json"
    for path in (index_path, config_path, quant_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    index = _json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model.safetensors.index.json has no weight_map")
    quant = _json(quant_path)
    quant_cfg = _json(config_path).get("quantization_config", {})
    bits = quant.get("bits", quant_cfg.get("bits"))
    codebook = quant.get("codebook", quant_cfg.get("codebook"))
    if float(bits or 0) not in {3.0, 4.0, 5.0, 6.0}:
        raise ValueError(
            f"rank slicing requires an integral 3/4/5/6 bitrate, got {bits!r}"
        )
    if codebook != "mcg":
        raise ValueError(f"rank slicing requires the MCG codebook, got {codebook!r}")

    fields: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    shards: dict[str, list[str]] = defaultdict(list)
    for name, filename in weight_map.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"unsafe shard path for {name}: {filename!r}")
        shards[filename].append(name)
        match = EXPERT_TENSOR_RE.fullmatch(name)
        if match is not None:
            key = (
                int(match.group("layer")),
                int(match.group("expert")),
                match.group("projection"),
            )
            fields[key].add(match.group("field"))
    if not fields:
        raise ValueError("checkpoint has no routed-expert EXL3 tensors")
    bad = {key: value for key, value in fields.items() if value != REQUIRED_FIELDS}
    if bad:
        key, value = next(iter(bad.items()))
        raise ValueError(
            f"incomplete or unsupported expert record {key}: {sorted(value)}"
        )

    layers = sorted({key[0] for key in fields})
    experts_by_layer = {
        layer: {key[1] for key in fields if key[0] == layer} for layer in layers
    }
    if layers != list(range(layers[0], layers[-1] + 1)):
        raise ValueError(f"MoE layer IDs are not contiguous: {layers}")
    counts = {len(value) for value in experts_by_layer.values()}
    if len(counts) != 1:
        raise ValueError(f"expert count varies by layer: {sorted(counts)}")
    experts = counts.pop()
    expected_experts = set(range(experts))
    for layer, found in experts_by_layer.items():
        if found != expected_experts:
            raise ValueError(f"layer {layer} has non-contiguous expert IDs")
        for expert in found:
            projections = {
                key[2] for key in fields if key[0] == layer and key[1] == expert
            }
            if projections != {"gate_proj", "up_proj", "down_proj"}:
                raise ValueError(
                    f"layer {layer} expert {expert} has projections "
                    f"{sorted(projections)}"
                )

    output_bytes = 0
    output_tensors = 0
    expert_shapes: dict[str, dict[str, tuple[int, ...]]] = defaultdict(dict)
    for filename, names in sorted(shards.items()):
        shard = source / filename
        if not shard.is_file():
            raise FileNotFoundError(shard)
        with safe_open(shard, framework="pt", device="cpu") as handle:
            actual = set(handle.keys())
            if actual != set(names):
                raise ValueError(f"index contents do not match {filename}")
            for name in names:
                tensor = handle.get_tensor(name)
                match = EXPERT_TENSOR_RE.fullmatch(name)
                if match is not None:
                    field = match.group("field")
                    _validate_expert_tensor_abi(
                        name, tensor, field=field, bits=int(float(bits))
                    )
                    expert_shapes[match.group("prefix")][field] = tuple(tensor.shape)
                transformed = transform_tensor(name, tensor, tp=tp)
                output_tensors += len(transformed)
                output_bytes += sum(
                    _tensor_nbytes(value) for value in transformed.values()
                )

    for prefix, shapes in expert_shapes.items():
        trellis = shapes["trellis"]
        expected_suh = (trellis[0] * 16,)
        expected_svh = (trellis[1] * 16,)
        if shapes["suh"] != expected_suh or shapes["svh"] != expected_svh:
            raise ValueError(
                f"{prefix} Hadamard geometry does not match trellis: "
                f"suh={shapes['suh']} expected={expected_suh}, "
                f"svh={shapes['svh']} expected={expected_svh}"
            )

    return {
        "bits": int(float(bits)),
        "codebook": codebook,
        "exllamav3_version": quant.get("version", quant_cfg.get("version")),
        "experts_per_layer": experts,
        "moe_layers": [layers[0], layers[-1]],
        "source_shards": len(shards),
        "source_tensors": len(weight_map),
        "output_tensors": output_tensors,
        "output_tensor_bytes": output_bytes,
    }


def _copy_support_files(source: Path, target: Path) -> None:
    for path in source.iterdir():
        if path.is_symlink():
            raise ValueError(f"source contains a symlink: {path.name}")
        if path.is_dir():
            shutil.copytree(path, target / path.name)
        elif path.suffix != ".safetensors" and path.name not in {
            "config.json",
            "quantization_config.json",
            "model.safetensors.index.json",
        }:
            shutil.copy2(path, target / path.name)


def _rank_metadata(plan: dict[str, Any], tp: int) -> dict[str, Any]:
    metadata = {
        "format": "exl3-trellis",
        "bits": plan["bits"],
        "codebook": plan["codebook"],
        "experts_per_layer": plan["experts_per_layer"],
        "moe_layers": plan["moe_layers"],
        "tensor_schema": RANK_SLICED_SCHEMA,
        "tp": tp,
    }
    if plan.get("exllamav3_version") is not None:
        metadata["exllamav3_version"] = plan["exllamav3_version"]
    return metadata


def convert_checkpoint(
    source: Path,
    output: Path,
    *,
    tp: int,
    validate: bool = True,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if output.parent == source or source in output.parents:
        raise ValueError("output must not be inside the source checkpoint")
    plan = inspect_source(source, tp=tp)
    partial = output.with_name(f".{output.name}.partial-{uuid.uuid4().hex[:12]}")
    partial.mkdir(parents=False)
    try:
        _copy_support_files(source, partial)
        source_index = _json(source / "model.safetensors.index.json")
        source_map = source_index["weight_map"]
        by_shard: dict[str, list[str]] = defaultdict(list)
        for name, filename in source_map.items():
            by_shard[filename].append(name)

        output_map: dict[str, str] = {}
        total_size = 0
        for filename, names in sorted(by_shard.items()):
            tensors: dict[str, torch.Tensor] = {}
            source_shard = source / filename
            with safe_open(source_shard, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
                for name in names:
                    transformed = transform_tensor(name, handle.get_tensor(name), tp=tp)
                    for new_name, tensor in transformed.items():
                        if new_name in output_map:
                            raise ValueError(f"duplicate output tensor: {new_name}")
                        tensors[new_name] = tensor
                        output_map[new_name] = filename
                        total_size += _tensor_nbytes(tensor)
            save_file(tensors, partial / filename, metadata=metadata)

        config = _json(source / "config.json")
        config["hybrid_tr3_tail"] = _rank_metadata(plan, tp)
        embedded_quant = config.setdefault("quantization_config", {})
        embedded_quant["serving_reader_qualified"] = False
        _write_json(partial / "config.json", config)

        quant = _json(source / "quantization_config.json")
        # Routed-expert tensor names change during rank slicing, so their old
        # storage records are invalid. Dense EXL3 records remain byte-identical
        # and are required for vLLM to instantiate their quantized modules.
        storage = quant.get("tensor_storage", {})
        quant["tensor_storage"] = {
            prefix: entry
            for prefix, entry in storage.items()
            if not any(
                EXPERT_TENSOR_RE.fullmatch(name)
                for name in entry.get("stored_tensors", {})
            )
        }
        quant["serving_reader_qualified"] = False
        quant["rank_sliced"] = _rank_metadata(plan, tp)
        _write_json(partial / "quantization_config.json", quant)
        _write_json(
            partial / "model.safetensors.index.json",
            {"metadata": {"total_size": total_size}, "weight_map": output_map},
        )

        result = {
            **plan,
            "format": "exl3-trellis",
            "tp": tp,
            "source": str(source),
            "source_config_sha256": _sha256(source / "config.json"),
            "source_index_sha256": _sha256(source / "model.safetensors.index.json"),
        }
        _write_json(partial / "rank-slice-manifest.json", result)
        if validate:
            validate_checkpoint(source, partial, tp=tp)
        os.rename(partial, output)
        return result
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def validate_checkpoint(source: Path, output: Path, *, tp: int) -> None:
    """Reassemble every transformed tensor and compare it with the source."""
    source_index = _json(source / "model.safetensors.index.json")["weight_map"]
    output_index = _json(output / "model.safetensors.index.json")["weight_map"]
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, filename in source_index.items():
        by_shard[filename].append(name)
    for source_filename, names in sorted(by_shard.items()):
        with (
            safe_open(
                source / source_filename, framework="pt", device="cpu"
            ) as source_handle,
            safe_open(
                output / source_filename, framework="pt", device="cpu"
            ) as output_handle,
        ):
            for name in names:
                original = source_handle.get_tensor(name)
                match = EXPERT_TENSOR_RE.fullmatch(name)
                expected_names = [name]
                rule = None
                if match is not None:
                    rule = tensor_rule(match.group("projection"), match.group("field"))
                    expected_names = [
                        ranked_name(match.group("prefix"), rank, match.group("field"))
                        for rank in range(tp)
                    ]
                pieces = []
                for expected in expected_names:
                    filename = output_index.get(expected)
                    if filename is None:
                        raise ValueError(f"output index is missing {expected}")
                    if filename != source_filename:
                        raise ValueError(
                            f"tensor {expected} moved to unexpected shard {filename}"
                        )
                    pieces.append(output_handle.get_tensor(expected))
                if rule is None:
                    if not torch.equal(pieces[0], original):
                        raise ValueError(f"ordinary tensor changed: {name}")
                elif rule.split_dim is None:
                    if any(not torch.equal(piece, original) for piece in pieces):
                        raise ValueError(f"replicated tensor changed: {name}")
                else:
                    rebuilt = torch.cat(pieces, dim=rule.split_dim)
                    if not torch.equal(rebuilt, original):
                        raise ValueError(f"rank slices do not reassemble: {name}")


def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument(
        "--plan", action="store_true", help="inspect and report without writing"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip the bitwise reconstruction check",
    )
    args = parser.parse_args(argv)
    if not args.plan and args.output is None:
        parser.error("output is required unless --plan is used")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.plan:
            result = inspect_source(args.source.resolve(), tp=args.tp)
        else:
            result = convert_checkpoint(
                args.source,
                args.output,
                tp=args.tp,
                validate=not args.no_validate,
            )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.plan:
        planned_size = _human_bytes(result["output_tensor_bytes"])
        print(f"planned output tensor bytes: {planned_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
