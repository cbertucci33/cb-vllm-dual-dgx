# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Release quantization-owned resources before discarding model modules."""


def release_model_runtime_resources(model) -> None:
    """Release shared runtime owners once, without importing optional backends."""
    if model is None:
        return
    seen = set()
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        owner = getattr(method, "quant_config", None)
        release = getattr(method, "release_runtime_resources", None)
        if release is None:
            continue
        identity = id(owner) if owner is not None else id(method)
        if identity not in seen:
            release()
            seen.add(identity)
