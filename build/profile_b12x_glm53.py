#!/usr/bin/env python3
"""Measure and merge the GLM-5.3 DFlash MXFP8 geometries on GB10."""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Mapping
from pathlib import Path

from b12x.policy import detect_device
from b12x.policy.generation import (
    DecisionRecord,
    GenerationContext,
    GenerationSettings,
    build_axis_tree,
    decision_node_to_dict,
)
from b12x.policy.generation.progress import RichProgressReporter
from b12x.policy.generation.providers.blockscaled import (
    QUERY_FIELDS,
    ROW_COUNTS,
    BlockscaledPrecisionGenerator,
    precision_cases,
)
from b12x.policy.generation.runner import (
    estimate_generators,
    generate_profile_artifact,
    runtime_profile_payload,
    write_artifact_atomic,
)
from b12x.policy.serialization import profile_from_dict
from b12x.policy.types import FrozenMapping

EXPECTED_B12X_COMMIT = "00b69ac22e21413622c4ecd98f607a2c3e015161"

# B12X geometries are (out_features, in_features). These are the seven
# GLM-5.3 DFlash MXFP8 K->N shapes that fell through the embedded GB10 profile.
GLM53_GEOMETRIES = (
    (4096, 20480),
    (1024, 4096),
    (3072, 4096),
    (4096, 2048),
    (12288, 4096),
    (4096, 6144),
    (256, 4096),
)


def read_json(path: Path) -> Mapping[str, object]:
    payload = path.read_bytes()
    if path.suffix == ".gz":
        payload = gzip.decompress(payload)
    value = json.loads(payload)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def git_head(source_root: Path) -> str:
    git_dir = source_root / ".git"
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    loose_ref = git_dir / ref
    if loose_ref.is_file():
        return loose_ref.read_text(encoding="utf-8").strip()
    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or line.startswith("^"):
                continue
            revision, name = line.split(" ", 1)
            if name == ref:
                return revision
    raise RuntimeError(f"cannot resolve {ref} in {git_dir}")


def planner_records(
    node: Mapping[str, object],
    query: Mapping[str, object] | None = None,
) -> tuple[DecisionRecord, ...]:
    values = dict(query or {})
    kind = node.get("kind")
    if kind == "leaf":
        config = node.get("config")
        if not isinstance(config, Mapping):
            raise TypeError("profile leaf config must be an object")
        return (
            DecisionRecord(
                query=FrozenMapping(values),
                config=FrozenMapping(dict(config)),
            ),
        )
    if kind != "exact":
        raise ValueError(f"unsupported planner node kind {kind!r}")
    field = node.get("field")
    branches = node.get("branches")
    if not isinstance(field, str) or not isinstance(branches, list):
        raise TypeError("exact planner node is malformed")
    records: list[DecisionRecord] = []
    for branch in branches:
        if not isinstance(branch, Mapping) or not isinstance(
            branch.get("node"), Mapping
        ):
            raise TypeError("exact planner branch is malformed")
        records.extend(
            planner_records(
                branch["node"],
                {**values, field: branch.get("value")},
            )
        )
    return tuple(records)


def replace_blockscaled_component(
    base_profile: Mapping[str, object],
    measured_component: Mapping[str, object],
) -> dict[str, object]:
    component_id = "gemm.blockscaled_precision"
    components = base_profile.get("components")
    if not isinstance(components, list):
        raise TypeError("base profile components must be a list")
    base_component = next(
        (
            component
            for component in components
            if isinstance(component, Mapping)
            and component.get("component_id") == component_id
        ),
        None,
    )
    if base_component is None:
        raise ValueError(f"base profile is missing {component_id}")
    base_planner = base_component.get("planner")
    measured_planner = measured_component.get("planner")
    if not isinstance(base_planner, Mapping) or not isinstance(
        measured_planner, Mapping
    ):
        raise TypeError("blockscaled profile planners must be objects")

    by_query: dict[tuple[object, ...], DecisionRecord] = {}
    for record in (*planner_records(base_planner), *planner_records(measured_planner)):
        key = tuple(record.query[field] for field in QUERY_FIELDS)
        by_query[key] = record
    merged_component = dict(base_component)
    merged_component["planner"] = decision_node_to_dict(
        build_axis_tree(
            tuple(by_query.values()),
            field_order=QUERY_FIELDS,
            range_fields=frozenset(),
        )
    )
    base_coverage = base_component.get("coverage", {})
    measured_coverage = measured_component.get("coverage", {})
    if not isinstance(base_coverage, Mapping) or not isinstance(
        measured_coverage, Mapping
    ):
        raise TypeError("blockscaled profile coverage must be an object")
    merged_component["coverage"] = {
        **base_coverage,
        "glm53_mxfp8_extension": dict(measured_coverage),
    }

    merged_components = [
        merged_component
        if isinstance(component, Mapping)
        and component.get("component_id") == component_id
        else component
        for component in components
    ]
    merged_profile = dict(base_profile)
    merged_profile["components"] = merged_components
    profile_from_dict(merged_profile)
    return merged_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b12x-source", type=Path, required=True)
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    revision = git_head(args.b12x_source)
    if revision != EXPECTED_B12X_COMMIT:
        raise RuntimeError(
            f"B12X source mismatch: {revision}, expected {EXPECTED_B12X_COMMIT}"
        )
    detected = detect_device(args.device)
    if detected.identity is None or detected.ordinal is None:
        raise RuntimeError(f"CUDA device did not resolve: {args.device}")

    generator = BlockscaledPrecisionGenerator(
        cases=precision_cases(
            geometries=GLM53_GEOMETRIES,
            counts=ROW_COUNTS,
            recipes=("mxfp8",),
        )
    )
    generators = (generator,)
    context = GenerationContext(
        device=detected.identity,
        device_ordinal=detected.ordinal,
        work_dir=args.work_dir.resolve(),
        source_revision=revision,
        settings=GenerationSettings(
            warmup=2,
            repetitions=5,
            groups=5,
            seed=20260828,
            minimum_cosine=0.998,
            cold_l2=True,
            max_candidate_seconds=2.0,
        ),
    )
    estimates = estimate_generators(generators, context)
    with RichProgressReporter(estimates) as progress:
        measured = generate_profile_artifact(
            profile_id="nvidia.gb10.48sm",
            generators=generators,
            context=context,
            progress=progress,
        )

    measured_profile = measured.get("profile")
    if not isinstance(measured_profile, Mapping):
        raise TypeError("measured artifact profile must be an object")
    measured_components = measured_profile.get("components")
    if not isinstance(measured_components, list) or len(measured_components) != 1:
        raise ValueError("expected exactly one measured profile component")
    base = read_json(args.base_profile)
    base_profile = base.get("profile", base)
    if not isinstance(base_profile, Mapping):
        raise TypeError("base profile must be an object")
    merged_profile = replace_blockscaled_component(
        base_profile,
        measured_components[0],
    )
    evidence_artifact = dict(measured)
    evidence_artifact["profile"] = merged_profile
    write_artifact_atomic(
        args.evidence_output,
        evidence_artifact,
        overwrite=True,
    )
    write_artifact_atomic(
        args.runtime_output,
        runtime_profile_payload(merged_profile),
        overwrite=True,
        compact=True,
    )


if __name__ == "__main__":
    main()
