# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline validation for fresh TP2 DFlash cache-write captures."""

import hashlib
import json
import math

PAIRS = {(rank, phase) for rank in (0, 1) for phase in ("context", "query")}
STAGES = {"calibration", "heldout"}


def validate_events(events, samples):
    expected = {row["sample"]: row["stage"] for row in samples}
    if len(expected) != len(samples) or set(expected.values()) != STAGES:
        raise ValueError("Fixtures require unique samples in both stages")
    seen = {}
    for event in events:
        sample = event["sample"]
        if sample not in expected or event["stage"] != expected[sample]:
            raise ValueError("Unexpected sample or stage")
        fixture = next(row for row in samples if row["sample"] == sample)
        if "request" in fixture:
            expected_hash = hashlib.sha256(
                json.dumps(fixture["request"], separators=(",", ":")).encode()
            ).hexdigest()
            if event["request_sha256"] != expected_hash:
                raise ValueError("Capture request differs from pinned fixture")
        layer = event["layer"]
        if layer not in range(5) or (event["rank"], event["phase"]) not in PAIRS:
            raise ValueError("Unexpected layer/rank/phase")
        if event["rows"] <= 0 or event.get("provenance_version") != 2:
            raise ValueError("Missing actual-row provenance")
        if not event.get("request_sha256") or not event.get("capture_id"):
            raise ValueError("Missing request/capture identity")
        for kind in ("k", "v"):
            maximum = event[kind + "_absmax"]
            if not math.isfinite(maximum) or maximum < 0:
                raise ValueError("Nonfinite activation maximum")
        seen.setdefault((sample, layer), set()).add((event["rank"], event["phase"]))
    for sample in expected:
        for layer in range(5):
            if seen.get((sample, layer)) != PAIRS:
                raise ValueError(
                    f"Incomplete sample {sample}, layer {layer} rank/phase coverage"
                )
    return seen


def validate_rows(token_ids, positions, sequence_ids, slots):
    count = len(slots)
    if not count or any(len(x) != count for x in (token_ids, positions, sequence_ids)):
        raise ValueError("Actual write rows lack aligned token/sequence provenance")
    if any(slot < 0 for slot in slots):
        raise ValueError("Padding/rejected rows entered calibration")
    if any(not isinstance(token, int) or token < 0 for token in token_ids):
        raise ValueError("Invalid rendered token IDs")
    if any(not isinstance(pos, int) or pos < 0 for pos in positions):
        raise ValueError("Invalid original positions")
    if any(
        not isinstance(seq, str) or not seq or seq == "padding" for seq in sequence_ids
    ):
        raise ValueError("Missing scheduled sequence owner")
