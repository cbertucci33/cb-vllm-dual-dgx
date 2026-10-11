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
    "0.2.0": (
        "https://files.pythonhosted.org/packages/27/8e/b209471a5a73a7ef1dad7365e05adf4c95b20fadc16fc9aea10ceaba1fc8/instanttensor-0.2.0.tar.gz",
        "acafa964f9ba6d2e342d8ffcb7e68bacc6d59d60e2fda222dc1333045cd50c9c",
    ),
    "0.2.1": (
        "https://files.pythonhosted.org/packages/0f/bb/15f742b5c5a671d0455520d107ea10c84ad277d346107d6699e6988cde8a/instanttensor-0.2.1.tar.gz",
        "d9a986b98d6d4c95102bc8471733cbe9be1e4bb425b7c8be2b16948f116bcb65",
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
