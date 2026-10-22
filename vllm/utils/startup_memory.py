# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocation admission policy, independent of Torch and model imports."""

from dataclasses import dataclass


@dataclass
class StartupMemoryBudget:
    initial_available: int
    requested_bytes: int
    integrated: bool
    initial_host_available: int
    observed_peak: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.requested_bytes <= self.initial_available:
            raise ValueError("Startup memory request exceeds initial availability")

    @property
    def reserve_bytes(self) -> int:
        return self.initial_available - self.requested_bytes

    def observe(self, available: int) -> None:
        self.observed_peak = max(self.observed_peak, self.initial_available - available)

    def remaining(self, available: int) -> int:
        self.observe(available)
        return max(0, available - self.reserve_bytes)

    def check(
        self,
        *,
        device_bytes: int,
        host_bytes: int,
        available: int,
        host_available: int,
        label: str,
        respect_reserve: bool = True,
    ) -> None:
        if device_bytes < 0 or host_bytes < 0:
            raise ValueError("Allocation byte counts must be nonnegative")
        self.observe(available)
        demand = device_bytes + host_bytes if self.integrated else device_bytes
        remaining = self.remaining(available) if respect_reserve else available
        if demand > remaining or (respect_reserve and available < self.reserve_bytes):
            raise MemoryError(
                f"{label}: startup allocation requires {demand} bytes, "
                f"only {remaining} bytes remain after the "
                f"{self.reserve_bytes if respect_reserve else 0}-byte startup reserve; "
                f"device={device_bytes}, host={host_bytes}, "
                f"shared_memory={self.integrated}"
            )
        if not self.integrated and host_bytes > host_available:
            raise MemoryError(
                f"{label}: host allocation requires {host_bytes} bytes, "
                f"only {host_available} bytes are available"
            )
        # Prospective native peaks are recorded before allocation. A temporary
        # may disappear before the end snapshot or bypass Torch statistics.
        self.observed_peak = max(
            self.observed_peak,
            self.initial_available - available + demand,
        )

    def transient_headroom(self, available_after: int) -> int:
        self.observe(available_after)
        consumed = self.initial_available - available_after
        return max(0, self.observed_peak - consumed)


def memory_config_identity(value):
    """Encode complete memory configuration without process object identities."""
    import dataclasses
    import enum
    import math
    from pathlib import Path

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite memory configuration")
        return value
    if isinstance(value, enum.Enum):
        return memory_config_identity(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        import json

        entries = [
            [
                type(key).__module__ + "." + type(key).__qualname__,
                memory_config_identity(key),
                memory_config_identity(item),
            ]
            for key, item in value.items()
        ]
        entries.sort(key=lambda item: json.dumps(item[:2], sort_keys=True))
        return {"mapping": entries}
    if isinstance(value, (list, tuple)):
        return [memory_config_identity(item) for item in value]
    if isinstance(value, (set, frozenset)):
        import json

        items = [memory_config_identity(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if dataclasses.is_dataclass(value):
        attributes = {
            field.name: getattr(value, field.name)
            for field in dataclasses.fields(value)
        }
        attributes.update(getattr(value, "__dict__", {}))
        # These are outputs of memory planning, not its requested inputs.
        return {
            name: memory_config_identity(item)
            for name, item in sorted(attributes.items())
            if name not in {"num_gpu_blocks", "num_cpu_blocks"}
        }
    if hasattr(value, "to_dict"):
        return memory_config_identity(value.to_dict())
    if hasattr(value, "model_dump"):
        return memory_config_identity(value.model_dump(mode="json"))
    if type(value).__module__ == "torch" and type(value).__name__ in {
        "dtype",
        "device",
    }:
        return str(value)
    if isinstance(value, type):
        return value.__module__ + "." + value.__qualname__
    if callable(value) and getattr(value, "__name__", "") != "<lambda>":
        closure = getattr(value, "__closure__", None)
        if not closure:
            return value.__module__ + "." + value.__qualname__
    raise TypeError(
        f"Cannot safely fingerprint memory config type {type(value).__name__}"
    )
