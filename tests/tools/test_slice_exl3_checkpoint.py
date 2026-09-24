# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools.slice_exl3_checkpoint import (
    convert_checkpoint,
    inspect_source,
    validate_checkpoint,
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_checkpoint(path: Path, *, incomplete: bool = False) -> None:
    path.mkdir()
    tensors = {"model.embed_tokens.weight": torch.arange(12).reshape(3, 4)}
    storage = {}
    for expert in range(2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"model.language_model.layers.3.mlp.experts.{expert}.{projection}"
            components = {
                "trellis": torch.arange(4 * 4 * 64, dtype=torch.int16).reshape(4, 4, 64)
                + expert,
                "suh": torch.arange(64, dtype=torch.float16) + expert,
                "svh": torch.arange(64, dtype=torch.float16) + expert,
                "mcg": torch.tensor([0xCBAC1FED - (1 << 32)], dtype=torch.int32),
            }
            if incomplete and expert == 1 and projection == "down_proj":
                components.pop("svh")
            stored = {}
            for field, tensor in components.items():
                name = f"{prefix}.{field}"
                tensors[name] = tensor
                stored[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            storage[prefix] = {
                "quant_format": "exl3",
                "bits_per_weight": 4,
                "stored_tensors": stored,
            }
    save_file(tensors, path / "model-00001.safetensors", metadata={"format": "pt"})
    _write_json(
        path / "model.safetensors.index.json",
        {
            "metadata": {
                "total_size": sum(
                    t.numel() * t.element_size() for t in tensors.values()
                )
            },
            "weight_map": {name: "model-00001.safetensors" for name in tensors},
        },
    )
    _write_json(
        path / "config.json",
        {"quantization_config": {"bits": 4, "codebook": "mcg", "quant_method": "exl3"}},
    )
    _write_json(
        path / "quantization_config.json",
        {
            "bits": 4,
            "codebook": "mcg",
            "quant_method": "exl3",
            "tensor_storage": storage,
        },
    )
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_conversion_is_lossless_and_preserves_source(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _make_checkpoint(source)
    source_hash = _sha256(source / "model-00001.safetensors")

    result = convert_checkpoint(source, output, tp=2)

    assert _sha256(source / "model-00001.safetensors") == source_hash
    assert result["experts_per_layer"] == 2
    assert result["moe_layers"] == [3, 3]
    config = json.loads((output / "config.json").read_text())
    assert config["hybrid_tr3_tail"]["tp"] == 2
    assert config["quantization_config"]["serving_reader_qualified"] is False
    assert (output / "tokenizer.json").is_file()
    validate_checkpoint(source, output, tp=2)

    with safe_open(output / "model-00001.safetensors", framework="pt") as handle:
        gate = "model.language_model.layers.3.mlp.experts.0.gate_proj"
        down = "model.language_model.layers.3.mlp.experts.0.down_proj"
        assert handle.get_tensor(f"{gate}.rank0.trellis").shape == (4, 2, 64)
        assert handle.get_tensor(f"{gate}.rank0.svh").shape == (32,)
        assert handle.get_tensor(f"{gate}.rank0.suh").shape == (64,)
        assert handle.get_tensor(f"{down}.rank0.trellis").shape == (2, 4, 64)
        assert handle.get_tensor(f"{down}.rank0.suh").shape == (32,)
        assert handle.get_tensor(f"{down}.rank0.svh").shape == (64,)


def test_plan_rejects_incomplete_expert_records(tmp_path: Path):
    source = tmp_path / "source"
    _make_checkpoint(source, incomplete=True)

    with pytest.raises(ValueError, match="incomplete or unsupported expert record"):
        inspect_source(source, tp=2)


def test_converter_refuses_existing_output(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _make_checkpoint(source)
    output.mkdir()

    with pytest.raises(FileExistsError):
        convert_checkpoint(source, output, tp=2)


def test_plan_rejects_invalid_mcg_marker_dtype(tmp_path: Path):
    source = tmp_path / "source"
    _make_checkpoint(source)
    shard = source / "model-00001.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    marker = next(name for name in tensors if name.endswith(".mcg"))
    tensors[marker] = tensors[marker].to(torch.int64)
    save_file(tensors, shard, metadata={"format": "pt"})

    with pytest.raises(ValueError, match="scalar int32 sentinel"):
        inspect_source(source, tp=2)
