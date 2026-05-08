# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flat DeepSeek V3.2 model definition.

This file keeps the original DeepSeek module hierarchy intact so vLLM's
existing weight loader can populate all tensors.  The backbone forward pass is
redirected into flat_deepseek_v32_forward.py, where the layer body is expressed
as tensor operations over the loaded parameters.
"""

from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2ForCausalLM,
    DeepseekV2Model,
)
from vllm.sequence import IntermediateTensors

from .flat_deepseek_v32_forward import flat_forward


class FlatDeepseekV32Model(DeepseekV2Model):
    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds=None,
    ):
        return flat_forward(
            self,
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )


class FlatDeepseekV32ForCausalLM(DeepseekV2ForCausalLM):
    model_cls = FlatDeepseekV32Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
