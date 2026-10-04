#!/usr/bin/env python3
"""Replace explicit wheel members and regenerate the complete RECORD."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import zipfile
from pathlib import Path


def _record_value(payload: bytes) -> tuple[str, str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    return f"sha256={digest.decode()}", str(len(payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="WHEEL_PATH=LOCAL_PATH",
    )
    args = parser.parse_args()

    replacements: dict[str, Path] = {}
    for item in args.replace:
        wheel_path, separator, local_path = item.partition("=")
        if not separator or not wheel_path or not local_path:
            parser.error(f"invalid replacement: {item!r}")
        replacements[wheel_path] = Path(local_path)

    with zipfile.ZipFile(args.input) as source:
        infos = {info.filename: info for info in source.infolist()}
        record_names = [name for name in infos if name.endswith(".dist-info/RECORD")]
        if len(record_names) != 1:
            raise ValueError(f"expected one RECORD, found {record_names}")
        record_name = record_names[0]
        missing = sorted(set(replacements) - set(infos))
        if missing:
            raise ValueError(f"replacement members absent from wheel: {missing}")

        payloads: dict[str, bytes] = {}
        for name in infos:
            if name == record_name:
                continue
            payloads[name] = (
                replacements[name].read_bytes()
                if name in replacements
                else source.read(name)
            )

        record_buffer = io.StringIO(newline="")
        writer = csv.writer(record_buffer, lineterminator="\n")
        for name in sorted(payloads):
            digest, size = _record_value(payloads[name])
            writer.writerow((name, digest, size))
        writer.writerow((record_name, "", ""))
        payloads[record_name] = record_buffer.getvalue().encode()

        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as target:
            for info in source.infolist():
                clone = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                clone.compress_type = info.compress_type
                clone.comment = info.comment
                clone.extra = info.extra
                clone.internal_attr = info.internal_attr
                clone.external_attr = info.external_attr
                clone.create_system = info.create_system
                clone.flag_bits = info.flag_bits
                target.writestr(clone, payloads[info.filename])
        os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
