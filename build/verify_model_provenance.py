#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_versions(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        versions[key] = value
    return versions


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"hash mismatch for {path}: {actual} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument("dflash", type=Path)
    parser.add_argument("target_source", type=Path)
    parser.add_argument("dflash_source", type=Path)
    parser.add_argument("versions", type=Path)
    args = parser.parse_args()

    versions = parse_versions(args.versions)

    target_release_path = args.target / "RELEASE_MANIFEST.json"
    target_release = json.loads(target_release_path.read_text(encoding="utf-8"))
    target_conversion = target_release["conversion"]
    if target_release["base_model"] != versions["TARGET_SOURCE_REPO"]:
        raise ValueError("target base-model repository mismatch")
    if (
        target_conversion["source_config_sha256"]
        != versions["TARGET_SOURCE_CONFIG_SHA256"]
        or target_conversion["source_index_sha256"]
        != versions["TARGET_SOURCE_INDEX_SHA256"]
    ):
        raise ValueError("target source hashes do not match the release manifest")
    require_hash(
        target_release_path, versions["TARGET_MODEL_RELEASE_MANIFEST_SHA256"]
    )
    require_hash(args.target / "config.json", versions["TARGET_MODEL_CONFIG_SHA256"])
    require_hash(
        args.target / "model.safetensors.index.json",
        versions["TARGET_MODEL_INDEX_SHA256"],
    )
    require_hash(
        args.target / versions["TARGET_MODEL_LICENSE_FILE"],
        versions["TARGET_MODEL_LICENSE_SHA256"],
    )
    require_hash(
        args.target_source / "config.json",
        versions["TARGET_SOURCE_CONFIG_SHA256"],
    )
    require_hash(
        args.target_source / "model.safetensors.index.json",
        versions["TARGET_SOURCE_INDEX_SHA256"],
    )

    dflash_manifest_path = args.dflash / "conversion_manifest.json"
    dflash_manifest = json.loads(
        dflash_manifest_path.read_text(encoding="utf-8")
    )
    dflash_source = dflash_manifest["source"]
    if (
        dflash_source["model"] != versions["DFLASH_SOURCE_REPO"]
        or dflash_source["revision"] != versions["DFLASH_SOURCE_REVISION"]
        or dflash_source["config_sha256"]
        != versions["DFLASH_SOURCE_CONFIG_SHA256"]
        or dflash_source["weights_sha256"]
        != versions["DFLASH_SOURCE_WEIGHTS_SHA256"]
    ):
        raise ValueError("DFlash source provenance mismatch")
    if (
        dflash_manifest["output_weights_sha256"]
        != versions["DFLASH_MODEL_WEIGHTS_SHA256"]
    ):
        raise ValueError("DFlash output-weight hash mismatch")
    require_hash(
        dflash_manifest_path,
        versions["DFLASH_MODEL_CONVERSION_MANIFEST_SHA256"],
    )
    require_hash(
        args.dflash / "config.json", versions["DFLASH_MODEL_CONFIG_SHA256"]
    )
    require_hash(
        args.dflash / "model.safetensors",
        versions["DFLASH_MODEL_WEIGHTS_SHA256"],
    )
    require_hash(
        args.dflash_source / "config.json",
        versions["DFLASH_SOURCE_CONFIG_SHA256"],
    )

    print("verified target and DFlash artifact/source provenance")


if __name__ == "__main__":
    main()
