#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build the pinned InstantTensor release with its cleanup repair."""

import argparse
import hashlib
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

SOURCES = {
    "0.2.2": (
        "https://files.pythonhosted.org/packages/ca/a1/95d0d4cee908ab0bd8e56b44cdb19e678081c54de52d280b0b3c9f7e4c43/instanttensor-0.2.2.tar.gz",
        "aca07c1afe4fde23dcbfb53044e2bef50dad4c0af054317acc669a1c8dc2b325",
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=SOURCES, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    url, expected = SOURCES[args.version]
    data = (
        args.source_archive.read_bytes()
        if args.source_archive
        else urllib.request.urlopen(url, timeout=60).read()
    )
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("InstantTensor source hash mismatch")
    archive = args.work_dir / "source.tar.gz"
    archive.write_bytes(data)
    with tarfile.open(archive) as stream:
        stream.extractall(args.work_dir, filter="data")
    source = args.work_dir / f"instanttensor-{args.version}"
    patch = (
        Path(__file__).resolve().parent
        / "patches"
        / f"instanttensor-{args.version}-cleanup.patch"
    )
    subprocess.run(
        ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(patch)],
        cwd=source,
        check=True,
    )
    if args.version == "0.2.2":
        safety_patch = patch.with_name(
            f"instanttensor-{args.version}-startup-safety.patch"
        )
        subprocess.run(
            [
                "patch",
                "--batch",
                "--forward",
                "--fuzz=0",
                "-p1",
                "-i",
                str(safety_patch),
            ],
            cwd=source,
            check=True,
        )
    if args.prepare_only:
        return
    args.wheel_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "setup.py",
            "bdist_wheel",
            "--build-number",
            "1",
            "--dist-dir",
            str(args.wheel_dir.resolve()),
        ],
        cwd=source,
        check=True,
    )


if __name__ == "__main__":
    main()
