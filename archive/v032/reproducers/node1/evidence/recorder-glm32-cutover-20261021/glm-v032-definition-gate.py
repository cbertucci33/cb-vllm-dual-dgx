#!/usr/bin/env python3
"""Compare a created GLM candidate with the preserved accepted container."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


IMAGE_ENV_DIFFERENCES = {
    "CRIU_CUDA_PLUGIN_DIR",
    "CUDA_VERSION",
    "MAX_JOBS",
    "NVCC_THREADS",
    "VLLM_BUILD_COMMIT",
    "VLLM_BUILD_PIPELINE",
    "VLLM_BUILD_URL",
    "VLLM_IMAGE_TAG",
    "VLLM_SOURCE_COMMIT",
    "UV_COMPILE_BYTECODE",
}
REQUIRED_CANDIDATE_LABELS = {
    "org.opencontainers.image.revision",
    "org.opencontainers.image.base.name",
    "io.vllm-runner.vllm-base.id",
    "io.vllm-runner.cuda",
    "io.vllm-runner.flashinfer.revision",
    "io.vllm-runner.sparkinfer.revision",
    "io.vllm-runner.exllamav3.revision",
    "io.vllm-runner.b12x.revision",
    "io.vllm-runner.flashkda.revision",
}
CACHE_DESTINATIONS = {
    "/cache": ".",
    "/root/.cache/b12x": "jit/b12x",
    "/root/.cache/flashinfer": "jit/flashinfer",
    "/root/.cache/vllm": "jit/vllm",
    "/root/.triton": "jit/triton",
}

APPROVED_ARG_TRANSITIONS = {
    "--chat-template": (
        "/opt/glm53/chat_template.jinja",
        "/models/glm53-exl3/chat_template.jinja",
    ),
}


def normalize_approved_arg_transitions(
    accepted_values: list[str] | None, candidate_values: list[str] | None
) -> tuple[list[str] | None, list[str] | None]:
    """Normalize only the verified identical mounted template path."""
    if accepted_values is None or candidate_values is None:
        return accepted_values, candidate_values
    accepted = list(accepted_values)
    candidate = list(candidate_values)
    for flag, (accepted_value, candidate_value) in APPROVED_ARG_TRANSITIONS.items():
        if accepted.count(flag) != 1 or candidate.count(flag) != 1:
            continue
        accepted_index = accepted.index(flag) + 1
        candidate_index = candidate.index(flag) + 1
        if accepted_index >= len(accepted) or candidate_index >= len(candidate):
            continue
        if accepted[accepted_index] == accepted_value and candidate[candidate_index] == candidate_value:
            candidate[candidate_index] = accepted_value
    return accepted, candidate

def inspect(name: str) -> dict[str, Any]:
    output = subprocess.check_output(["docker", "inspect", name], text=True)
    return json.loads(output)[0]


def env_map(values: list[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values or []:
        key, separator, value = item.partition("=")
        if not separator:
            raise RuntimeError(f"malformed environment entry: {item!r}")
        result[key] = value
    return result


def mount_map(container: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mounts = container.get("Mounts") or []
    result = {mount["Destination"]: mount for mount in mounts}
    if len(result) != len(mounts):
        raise RuntimeError("duplicate mount destination")
    return result


def normalized_config(container: dict[str, Any]) -> dict[str, Any]:
    config = json.loads(json.dumps(container["Config"]))
    config.pop("Labels", None)
    config.pop("Image", None)
    config["Env"] = {
        key: value
        for key, value in env_map(config.get("Env")).items()
        if key not in IMAGE_ENV_DIFFERENCES
    }
    workspace = config["Env"].get("FLASHINFER_WORKSPACE_BASE")
    labels = container["Config"].get("Labels") or {}
    revision = labels.get("org.opencontainers.image.revision", "")
    flashinfer = labels.get("io.vllm-runner.flashinfer.revision", "")
    # Accepted and candidate images use distinct pinned FlashInfer revisions.
    # Candidate workspace is validated against both image labels below.
    if workspace and workspace.startswith("/cache/jit/flashinfer/"):
        config["Env"]["FLASHINFER_WORKSPACE_BASE"] = "/cache/jit/flashinfer/<pinned-revision>"
    return config


def normalized_host_config(
    container: dict[str, Any], cache_root: Path
) -> dict[str, Any]:
    host_config = json.loads(json.dumps(container["HostConfig"]))
    normalized_binds: list[str] = []
    for bind in host_config.get("Binds") or []:
        parts = bind.rsplit(":", 2)
        if len(parts) < 2:
            raise RuntimeError(f"malformed bind: {bind!r}")
        source, destination = parts[:2]
        mode = parts[2] if len(parts) == 3 else ""
        if destination in CACHE_DESTINATIONS:
            source = str((cache_root / CACHE_DESTINATIONS[destination]).resolve())
        normalized_binds.append(":".join((source, destination, mode)))
    host_config["Binds"] = sorted(normalized_binds)
    return host_config


def changed_keys(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    def equal(left_value: Any, right_value: Any) -> bool:
        if left_value == right_value:
            return True
        return (left_value is None and right_value == []) or (
            right_value is None and left_value == []
        )

    return sorted(
        key
        for key in set(left) | set(right)
        if not equal(left.get(key), right.get(key))
    )


def compare(accepted: dict[str, Any], candidate: dict[str, Any], cache_root: Path) -> None:
    differences: list[str] = []
    accepted_config = normalized_config(accepted)
    candidate_config = normalized_config(candidate)
    accepted_config["Cmd"], candidate_config["Cmd"] = (
        normalize_approved_arg_transitions(
            accepted_config.get("Cmd"), candidate_config.get("Cmd")
        )
    )
    for field in ("AttachStdout", "AttachStderr"):
        if accepted_config.get(field) is False and candidate_config.get(field) is True:
            candidate_config[field] = False
    if accepted_config != candidate_config:
        differences.extend(
            f"Config.{key}" for key in changed_keys(accepted_config, candidate_config)
        )
    accepted_host = normalized_host_config(accepted, cache_root)
    candidate_host = normalized_host_config(candidate, cache_root)
    if (
        accepted_host.get("OomKillDisable") is None
        and candidate_host.get("OomKillDisable") is False
    ):
        candidate_host["OomKillDisable"] = None
    if accepted_host != candidate_host:
        differences.extend(
            f"HostConfig.{key}"
            for key in changed_keys(accepted_host, candidate_host)
        )
    accepted_args, candidate_args = normalize_approved_arg_transitions(
        accepted.get("Args"), candidate.get("Args")
    )
    for field in ("Path", "Driver", "Platform"):
        if accepted.get(field) != candidate.get(field):
            differences.append(field)
    if accepted_args != candidate_args:
        differences.append("Args")

    accepted_mounts = mount_map(accepted)
    candidate_mounts = mount_map(candidate)
    if set(accepted_mounts) != set(candidate_mounts):
        differences.append("Mount destination set")
    else:
        for destination, accepted_mount in accepted_mounts.items():
            candidate_mount = candidate_mounts[destination]
            for field in ("Type", "Destination", "Mode", "RW", "Propagation"):
                if accepted_mount.get(field) != candidate_mount.get(field):
                    differences.append(f"Mounts[{destination!r}].{field}")
            if destination in CACHE_DESTINATIONS:
                expected_source = (cache_root / CACHE_DESTINATIONS[destination]).resolve()
                if Path(candidate_mount["Source"]).resolve() != expected_source:
                    differences.append(f"Mounts[{destination!r}].Source")
            elif accepted_mount.get("Source") != candidate_mount.get("Source"):
                differences.append(f"Mounts[{destination!r}].Source")

    if candidate["State"].get("Running"):
        differences.append("candidate started before definition verification")
    labels = candidate["Config"].get("Labels") or {}
    missing_labels = sorted(
        key for key in REQUIRED_CANDIDATE_LABELS if not labels.get(key)
    )
    if missing_labels:
        differences.append(f"candidate image labels missing {missing_labels}")
    revision = labels.get("org.opencontainers.image.revision", "")
    candidate_env = env_map(candidate["Config"].get("Env"))
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        differences.append("candidate source-revision label")
    if candidate_env.get("VLLM_SOURCE_COMMIT") != revision:
        differences.append("candidate source revision environment")
    flashinfer = labels.get("io.vllm-runner.flashinfer.revision", "")
    expected_workspace = f"/cache/jit/flashinfer/{flashinfer}/{revision}"
    if candidate_env.get("FLASHINFER_WORKSPACE_BASE") != expected_workspace:
        differences.append("candidate FlashInfer revision-scoped workspace")
    if differences:
        raise RuntimeError("candidate definition differs at: " + ", ".join(differences))

    print(
        "candidate definition verified: "
        f"{len(accepted_config)} normalized Config fields, "
        f"{len(accepted_host)} normalized HostConfig fields, "
        f"{len(accepted_mounts)} mounts, and {len(REQUIRED_CANDIDATE_LABELS)} "
        "candidate provenance labels"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("accepted")
    parser.add_argument("candidate")
    parser.add_argument("cache_root", type=Path)
    args = parser.parse_args()
    compare(inspect(args.accepted), inspect(args.candidate), args.cache_root)


if __name__ == "__main__":
    main()
