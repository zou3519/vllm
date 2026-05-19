# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flat DeepSeek V4 model entry point.

This module keeps the original DeepSeek V4 module hierarchy for weight loading
and swaps only the backbone forward for the flat implementation.
"""

from collections.abc import Iterable

import torch

from vllm.model_executor.models.interfaces import SupportsPP
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4ForCausalLM,
    _make_deepseek_v4_weights_mapper,
)

from .flat_deepseek_v4_forward import FlatDeepseekV4Model


class FlatDeepseekV4ForCausalLM(DeepseekV4ForCausalLM, SupportsPP):
    """DeepSeek V4 with a flat backbone forward path."""

    model_cls = FlatDeepseekV4Model
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params = super().load_weights(weights)
        # Weight loading finalizes MegaMoE tensors; discard any extraction cache
        # from profiling/dummy forwards so the next pass sees the final layout.
        if hasattr(self.model, "_flat_deepseek_v4_params"):
            delattr(self.model, "_flat_deepseek_v4_params")
        return loaded_params
