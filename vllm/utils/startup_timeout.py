# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-local startup deadline shared with loading/native adapters."""

import time

_deadline: float | None = None


def set_startup_deadline(deadline: float | None) -> None:
    global _deadline
    _deadline = deadline


def remaining_startup_time() -> float | None:
    if _deadline is None:
        return None
    return max(0.0, _deadline - time.monotonic())
