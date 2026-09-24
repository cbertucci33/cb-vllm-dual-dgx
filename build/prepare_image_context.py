#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path, manifest_name: str) -> dict[str, str]:
    manifest = root / manifest_name
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    hashes: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.removeprefix("*").removeprefix("./")
        path = root / name
        if path.parent != root or not path.is_file():
            raise ValueError(f"unsafe or missing manifest entry: {name}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"hash mismatch for {name}")
        hashes[name] = actual
    return dict(sorted(hashes.items()))


def parse_versions(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        versions[key] = value
    return versions


def extract_vllm_package(archive: Path, destination: Path) -> None:
    package_prefix = "vllm/vllm/"
    with tarfile.open(archive, "r:gz") as stream:
        members = [m for m in stream.getmembers() if m.name.startswith(package_prefix)]
        if not members:
            raise ValueError(f"vLLM package is missing from {archive}")
        for member in members:
            relative = Path(member.name.removeprefix(package_prefix))
            if not relative.parts or ".." in relative.parts:
                continue
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"unsupported vLLM archive entry: {member.name}")
            source = stream.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read vLLM archive entry: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())
            target.chmod(member.mode & 0o777)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_archive_dir", type=Path)
    parser.add_argument("wheelhouse_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    versions = parse_versions(script_dir / "versions.env")
    source_hashes = verify_manifest(args.source_archive_dir, "source-sha256.txt")
    artifact_hashes = verify_manifest(args.wheelhouse_dir, "artifact-sha256.txt")
    vllm_commit = (
        (args.source_archive_dir / "vllm-commit.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    if f"vllm-{vllm_commit}.tar.gz" not in source_hashes:
        raise ValueError("vLLM source archive does not match vllm-commit.txt")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        args.wheelhouse_dir,
        args.output_dir / "artifacts",
        dirs_exist_ok=False,
    )
    extract_vllm_package(
        args.source_archive_dir / f"vllm-{vllm_commit}.tar.gz",
        args.output_dir / "vllm-overlay",
    )
    shutil.copy2(script_dir / "verify_runtime.py", args.output_dir)
    template = (
        script_dir.parent
        / "templates/chat_template_glm53_official-690b705.jinja"
    )
    template_hash = sha256(template)
    shutil.copy2(template, args.output_dir / "chat_template.jinja")

    provenance = {
        "schema_version": 1,
        "vllm": {
            "base_version": versions["VLLM_BASE_VERSION"],
            "commit": vllm_commit,
        },
        "images": {
            "runtime": versions["VLLM_BASE_IMAGE"],
            "cuda_devel_builder": versions["CUDA_DEVEL_IMAGE"],
        },
        "external_revisions": {
            "flashinfer": versions["FLASHINFER_COMMIT"],
            "flashinfer_cccl": versions["FLASHINFER_CCCL_COMMIT"],
            "flashinfer_cutlass": versions["FLASHINFER_CUTLASS_COMMIT"],
            "flashinfer_spdlog": versions["FLASHINFER_SPDLOG_COMMIT"],
            "vllm_cutlass": versions["VLLM_CUTLASS_COMMIT"],
            "sparkinfer": versions["SPARKINFER_COMMIT"],
            "exllamav3": versions["EXLLAMAV3_COMMIT"],
            "b12x": versions["B12X_COMMIT"],
        },
        "chat_template": {
            "source_revision": "690b705278a3a58e538fcb37c2ca8b5f9511213c",
            "sha256": template_hash,
            "image_path": "/opt/glm53/chat_template.jinja",
            "tool_call_parser": "glm47",
            "reasoning_parser": "glm45",
        },
        "runtime_contract": {
            "attention_backend": "FLASHINFER_MLA_SPARSE_SM120",
            "target_cache_dtype": "fp8_ds_mla",
            "draft_cache_dtype": "fp8_e4m3",
            "linear_backend": "b12x",
            "b12x_max_m": 16,
        },
        "build": {
            "torch_cuda_arch_list": versions["TORCH_CUDA_ARCH_LIST"],
            "flashinfer_cuda_arch_list": versions["FLASHINFER_CUDA_ARCH_LIST"],
            "max_jobs": versions["MAX_JOBS"],
            "vllm_core_max_jobs": versions["VLLM_CORE_MAX_JOBS"],
            "nvcc_threads": versions["NVCC_THREADS"],
            "cmake": versions["CMAKE_VERSION"],
            "wheel": versions["WHEEL_VERSION"],
            "nvidia_cutlass_dsl": versions["NVIDIA_CUTLASS_DSL_VERSION"],
        },
        "source_sha256": source_hashes,
        "artifact_sha256": artifact_hashes,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"prepared image context for vLLM {vllm_commit}")


if __name__ == "__main__":
    main()
