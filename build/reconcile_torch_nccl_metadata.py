#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reconcile audited dependency metadata with the installed ARM64 runtime."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import re
from importlib import metadata
from pathlib import Path

from packaging.tags import parse_tag, sys_tags


def _record_hash(payload: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    return "sha256=" + digest.decode("ascii")


def _update_record(dist: metadata.Distribution, path: Path, payload: bytes) -> None:
    record_path = Path(dist._path) / "RECORD"
    site_root = Path(dist.locate_file("")).resolve()
    record_key = path.resolve().relative_to(site_root).as_posix()
    rows = list(csv.reader(record_path.read_text(encoding="utf-8").splitlines()))
    matches = 0
    for row in rows:
        if row and row[0] == record_key:
            row[1] = _record_hash(payload)
            row[2] = str(len(payload))
            matches += 1
    if matches != 1:
        raise RuntimeError(
            f"expected one RECORD entry for {record_key}, found {matches}"
        )
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    record_path.write_text(output.getvalue(), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nccl-version", required=True)
    parser.add_argument("--cusparselt-version", required=True)
    args = parser.parse_args()

    installed_nccl = metadata.version("nvidia-nccl-cu13")
    if installed_nccl != args.nccl_version:
        raise RuntimeError(
            f"installed NCCL distribution is {installed_nccl}; "
            f"expected {args.nccl_version}"
        )
    installed_cusparselt = metadata.version("nvidia-cusparselt-cu13")
    if installed_cusparselt != args.cusparselt_version:
        raise RuntimeError(
            f"installed cuSPARSELt distribution is {installed_cusparselt}; "
            f"expected {args.cusparselt_version}"
        )

    torch_dist = metadata.distribution("torch")
    dist_info = Path(torch_dist._path)
    metadata_path = dist_info / "METADATA"
    original = metadata_path.read_text(encoding="utf-8")
    replacement = rf"\g<1>{args.nccl_version}\g<2>"
    updated, count = re.subn(
        r"(?m)^(Requires-Dist: nvidia-nccl-cu13==)[^;]+(;.*)$",
        replacement,
        original,
    )
    if count != 1:
        raise RuntimeError(f"expected one Torch NCCL requirement, found {count}")
    metadata_bytes = updated.encode("utf-8")
    metadata_path.write_bytes(metadata_bytes)

    _update_record(torch_dist, metadata_path, metadata_bytes)

    reconciled = next(
        requirement
        for requirement in metadata.distribution("torch").requires or ()
        if requirement.startswith("nvidia-nccl-cu13==")
    )
    expected = f'nvidia-nccl-cu13=={args.nccl_version}; platform_system == "Linux"'
    if reconciled != expected:
        raise RuntimeError(
            f"Torch NCCL metadata is {reconciled!r}; expected {expected!r}"
        )
    cusparselt_dist = metadata.distribution("nvidia-cusparselt-cu13")
    wheel_path = Path(cusparselt_dist._path) / "WHEEL"
    wheel_text = wheel_path.read_text(encoding="utf-8")
    bad_tag = "Tag: py3-none-manylinux2014_sbsa"
    good_tag = "Tag: py3-none-manylinux2014_aarch64"
    if wheel_text.count(bad_tag) != 1:
        raise RuntimeError(
            "cuSPARSELt WHEEL does not contain exactly one incorrect SBSA tag"
        )
    wheel_bytes = wheel_text.replace(bad_tag, good_tag).encode("utf-8")
    wheel_path.write_bytes(wheel_bytes)
    _update_record(cusparselt_dist, wheel_path, wheel_bytes)

    supported_tags = set(sys_tags())
    corrected_tags = parse_tag("py3-none-manylinux2014_aarch64")
    if corrected_tags.isdisjoint(supported_tags):
        raise RuntimeError(
            "corrected cuSPARSELt wheel tag is not supported by this runtime"
        )
    if bad_tag in wheel_path.read_text(encoding="utf-8"):
        raise RuntimeError("cuSPARSELt WHEEL still advertises the SBSA tag")


if __name__ == "__main__":
    main()
