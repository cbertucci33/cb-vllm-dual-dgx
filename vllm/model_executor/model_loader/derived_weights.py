# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refresh derived weight copies without reprocessing kernel-format parameters."""

import torch
from torch import nn


@torch.no_grad()
def refresh_derived_weights(model: nn.Module) -> None:
    """Run opted-in module hooks after all source weights have been updated."""
    for module in model.modules():
        refresh = getattr(module, "refresh_derived_weights", None)
        if callable(refresh):
            refresh()
