# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn.functional as F

from vllm.distributed import (
    get_pp_group,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors


def transformer_layer(
    layer_params: tuple[Any, ...],
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    (
        input_layernorm,
        qkv_weight,
        qkv_bias,
        q_size,
        kv_size,
        rotary_emb,
        attn,
        o_proj_weight,
        o_proj_bias,
        post_attention_layernorm,
        router_weight,
        router_bias,
        experts,
        hidden_size,
        is_sequence_parallel,
    ) = layer_params

    if residual is None:
        residual = hidden_states
        hidden_states = input_layernorm.forward_native(hidden_states)
    else:
        hidden_states, residual = input_layernorm.forward_native(hidden_states, residual)

    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q, k = rotary_emb.forward_native(positions, q, k)
    assert k is not None
    attn_output = attn.forward(q, k, v)
    hidden_states = F.linear(attn_output, o_proj_weight, o_proj_bias)

    hidden_states, residual = post_attention_layernorm.forward_native(
        hidden_states, residual
    )

    num_tokens = hidden_states.shape[0]
    mlp_input = hidden_states
    if is_sequence_parallel:
        mlp_input = sequence_parallel_chunk(mlp_input)

    router_logits = F.linear(mlp_input, router_weight, router_bias)
    output = experts.forward_cuda(
        hidden_states=mlp_input,
        router_logits=router_logits,
    )[:, :hidden_size]

    if is_sequence_parallel:
        output = tensor_model_parallel_all_gather(output.contiguous(), 0)
        output = output[:num_tokens]
    return output, residual


def flat_forward(
    model: Any,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    cached_params = getattr(model, "_flat_gpt_oss_params", None)
    if cached_params is None:
        layer_params = []
        for layer in model.layers:
            if layer is None:
                layer_params.append(None)
                continue
            qkv_proj = layer.attn.qkv_proj
            o_proj = layer.attn.o_proj
            router = layer.mlp.router
            layer_params.append(
                (
                    layer.input_layernorm,
                    qkv_proj.weight,
                    None if qkv_proj.skip_bias_add else qkv_proj.bias,
                    layer.attn.q_size,
                    layer.attn.kv_size,
                    layer.attn.rotary_emb,
                    layer.attn.attn,
                    o_proj.weight,
                    None
                    if (o_proj.tp_rank > 0 or o_proj.skip_bias_add)
                    else o_proj.bias,
                    layer.post_attention_layernorm,
                    router.weight,
                    None if router.skip_bias_add else router.bias,
                    layer.mlp.experts,
                    layer.mlp.hidden_size,
                    layer.mlp.is_sequence_parallel,
                )
            )
        cached_params = (model.embedding, layer_params, model.norm)
        model._flat_gpt_oss_params = cached_params
    embedding, layer_params, norm = cached_params

    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            assert input_ids is not None
            x = embedding.forward_native(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        x = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    aux_hidden_states = model._maybe_add_hidden_state(
        [], model.start_layer, x, residual
    )
    for i in range(model.start_layer, model.end_layer):
        x, residual = transformer_layer(layer_params[i], x, positions, residual)
        model._maybe_add_hidden_state(aux_hidden_states, i + 1, x, residual)

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({"hidden_states": x, "residual": residual})

    assert residual is not None
    x, _ = norm.forward_native(x, residual)

    if len(aux_hidden_states) > 0:
        return x, aux_hidden_states
    return x
