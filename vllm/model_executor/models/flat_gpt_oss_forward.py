# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn.functional as F

from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
from vllm.sequence import IntermediateTensors


def transformer_layer(
    layer_params: tuple[Any, ...],
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    layer_slot_mapping: torch.Tensor | None,
    attn_kv_cache: torch.Tensor,
    residual: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    (
        input_norm_weight,
        input_norm_eps,
        qkv_weight,
        qkv_bias,
        q_size,
        kv_size,
        rotary_cos_sin_cache,
        rotary_head_size,
        rotary_dim,
        attn_layer_name,
        attn_num_heads,
        attn_num_kv_heads,
        attn_head_size,
        attn_head_size_v,
        attn_k_scale,
        attn_v_scale,
        attn_q_scale,
        attn_query_uses_fp8,
        attn_kv_cache_uses_fp8,
        fp8_dtype,
        fp8_min,
        fp8_max,
        o_proj_weight,
        o_proj_bias,
        post_attention_norm_weight,
        post_attention_norm_eps,
        router_weight,
        router_bias,
        experts,
        hidden_size,
    ) = layer_params

    # TransformerBlock.input_layernorm
    original_dtype = hidden_states.dtype
    if residual is None:
        residual = hidden_states
        hidden_states = hidden_states.to(torch.float32)
    else:
        hidden_states = torch.add(hidden_states.to(torch.float32), residual)
        residual = hidden_states.to(original_dtype)
    variance = torch.mean(torch.pow(hidden_states, 2), dim=-1, keepdim=True)
    hidden_states = torch.mul(
        hidden_states, torch.rsqrt(torch.add(variance, input_norm_eps))
    )
    hidden_states = torch.mul(hidden_states.to(original_dtype), input_norm_weight)

    # TransformerBlock.attn.qkv_proj
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

    # TransformerBlock.attn.rotary_emb
    positions = torch.flatten(positions)
    num_tokens = positions.shape[0]
    cos_sin_cache = rotary_cos_sin_cache.to(dtype=q.dtype, device=q.device)
    cos_sin = torch.index_select(cos_sin_cache, 0, positions)
    cos, sin = torch.chunk(cos_sin, 2, dim=-1)
    cos = torch.unsqueeze(cos, -2)
    sin = torch.unsqueeze(sin, -2)

    q = torch.reshape(q, (num_tokens, -1, rotary_head_size))
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    q1, q2 = torch.chunk(q_rot, 2, dim=-1)
    q_rot = torch.cat(
        (
            torch.sub(torch.mul(q1, cos), torch.mul(q2, sin)),
            torch.add(torch.mul(q2, cos), torch.mul(q1, sin)),
        ),
        dim=-1,
    )
    q = torch.reshape(torch.cat((q_rot, q_pass), dim=-1), (num_tokens, q_size))

    k = torch.reshape(k, (num_tokens, -1, rotary_head_size))
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]
    k1, k2 = torch.chunk(k_rot, 2, dim=-1)
    k_rot = torch.cat(
        (
            torch.sub(torch.mul(k1, cos), torch.mul(k2, sin)),
            torch.add(torch.mul(k2, cos), torch.mul(k1, sin)),
        ),
        dim=-1,
    )
    k = torch.reshape(torch.cat((k_rot, k_pass), dim=-1), (num_tokens, kv_size))

    # TransformerBlock.attn.attn KV-cache write and attention
    attn_output_dtype = q.dtype
    q = torch.reshape(q, (num_tokens, attn_num_heads, attn_head_size))
    k = torch.reshape(k, (num_tokens, attn_num_kv_heads, attn_head_size))
    v = torch.reshape(v, (num_tokens, attn_num_kv_heads, attn_head_size_v))

    if layer_slot_mapping is not None and torch.numel(attn_kv_cache) != 0:
        valid_slots = torch.ge(layer_slot_mapping, 0)
        token_indices = torch.squeeze(torch.nonzero(valid_slots), -1)
        slots = torch.index_select(layer_slot_mapping, 0, token_indices)
        key_cache, value_cache = torch.unbind(attn_kv_cache, 1)
        if attn_kv_cache_uses_fp8:
            key_cache = torch.ops.aten.view.dtype(key_cache, fp8_dtype)
            value_cache = torch.ops.aten.view.dtype(value_cache, fp8_dtype)
        block_size = key_cache.shape[1]
        block_indices = torch.div(slots, block_size, rounding_mode="floor")
        block_offsets = torch.remainder(slots, block_size)
        cache_k = torch.index_select(k, 0, token_indices)
        cache_v = torch.index_select(v, 0, token_indices)
        if attn_kv_cache_uses_fp8:
            cache_k = torch.clamp(
                torch.div(cache_k.to(torch.float32), attn_k_scale),
                fp8_min,
                fp8_max,
            )
            cache_v = torch.clamp(
                torch.div(cache_v.to(torch.float32), attn_v_scale),
                fp8_min,
                fp8_max,
            )
            cache_k = cache_k.to(fp8_dtype)
            cache_v = cache_v.to(fp8_dtype)
        else:
            cache_k = cache_k.to(key_cache.dtype)
            cache_v = cache_v.to(value_cache.dtype)
        key_cache[block_indices, block_offsets] = cache_k
        value_cache[block_indices, block_offsets] = cache_v

    if attn_query_uses_fp8:
        q = torch.clamp(
            torch.div(q.to(torch.float32), attn_q_scale),
            fp8_min,
            fp8_max,
        )
        q = q.to(fp8_dtype)

    attn_output = torch.empty(
        (num_tokens, attn_num_heads, attn_head_size_v),
        dtype=attn_output_dtype,
        device=q.device,
    )
    torch.ops.vllm.unified_attention_with_output(
        q,
        k,
        v,
        attn_output,
        attn_layer_name,
        None,
        None,
        None,
    )
    attn_output = torch.reshape(attn_output, (num_tokens, q_size))

    # TransformerBlock.attn.o_proj
    hidden_states = F.linear(attn_output, o_proj_weight, o_proj_bias)

    # TransformerBlock.post_attention_layernorm
    original_dtype = hidden_states.dtype
    hidden_states = torch.add(hidden_states.to(torch.float32), residual)
    residual = hidden_states.to(original_dtype)
    variance = torch.mean(torch.pow(hidden_states, 2), dim=-1, keepdim=True)
    hidden_states = torch.mul(
        hidden_states, torch.rsqrt(torch.add(variance, post_attention_norm_eps))
    )
    hidden_states = torch.mul(
        hidden_states.to(original_dtype),
        post_attention_norm_weight,
    )

    # TransformerBlock.mlp.router
    router_logits = F.linear(hidden_states, router_weight, router_bias)

    # TransformerBlock.mlp.experts
    output = experts.forward_cuda(
        hidden_states=hidden_states,
        router_logits=router_logits,
    )[:, :hidden_size]

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
        fp8_min, fp8_max = get_fp8_min_max()
        layer_params = []
        for layer in model.layers:
            if layer is None:
                layer_params.append(None)
                continue
            qkv_proj = layer.attn.qkv_proj
            o_proj = layer.attn.o_proj
            router = layer.mlp.router
            attn = layer.attn.attn
            if layer.input_layernorm.variance_size_override is not None:
                raise ValueError("Flat GPT-OSS only supports full-size RMSNorm")
            if layer.post_attention_layernorm.variance_size_override is not None:
                raise ValueError("Flat GPT-OSS only supports full-size RMSNorm")
            if layer.mlp.is_sequence_parallel:
                raise ValueError("Flat GPT-OSS is specialized for non-SP MoE")
            if not layer.attn.rotary_emb.is_neox_style:
                raise ValueError("Flat GPT-OSS is specialized for Neox rotary")
            if attn.attn_backend.forward_includes_kv_cache_update:
                raise ValueError("Flat GPT-OSS expects an external KV-cache update")
            if attn.kv_sharing_target_layer_name is not None:
                raise ValueError("Flat GPT-OSS does not support KV sharing")
            if attn.calculate_kv_scales:
                raise ValueError("Flat GPT-OSS expects precomputed KV scales")
            if attn.impl._is_per_token_head_quant:
                raise ValueError("Flat GPT-OSS is specialized for tensor-scale fp8 KV")
            layer_params.append(
                (
                    layer.input_layernorm.weight.data,
                    layer.input_layernorm.variance_epsilon,
                    qkv_proj.weight,
                    None if qkv_proj.skip_bias_add else qkv_proj.bias,
                    layer.attn.q_size,
                    layer.attn.kv_size,
                    layer.attn.rotary_emb.cos_sin_cache,
                    layer.attn.rotary_emb.head_size,
                    layer.attn.rotary_emb.rotary_dim,
                    attn.layer_name,
                    attn.num_heads,
                    attn.num_kv_heads,
                    attn.head_size,
                    attn.head_size_v,
                    attn._k_scale,
                    attn._v_scale,
                    attn._q_scale,
                    attn.query_quant is not None,
                    attn.kv_cache_dtype.startswith("fp8"),
                    attn.impl.fp8_dtype,
                    fp8_min,
                    fp8_max,
                    o_proj.weight,
                    None
                    if (o_proj.tp_rank > 0 or o_proj.skip_bias_add)
                    else o_proj.bias,
                    layer.post_attention_layernorm.weight.data,
                    layer.post_attention_layernorm.variance_epsilon,
                    router.weight,
                    None if router.skip_bias_add else router.bias,
                    layer.mlp.experts,
                    layer.mlp.hidden_size,
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
            # GptOssModel.embedding
            x = embedding.forward_native(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        x = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    aux_hidden_states = model._maybe_add_hidden_state(
        [], model.start_layer, x, residual
    )
    forward_context = get_forward_context()
    slot_mapping = forward_context.slot_mapping
    assert isinstance(slot_mapping, dict)
    for i in range(model.start_layer, model.end_layer):
        layer_name = layer_params[i][9]
        layer_slot_mapping = slot_mapping.get(layer_name)
        attn_kv_cache = forward_context.no_compile_layers[layer_name].kv_cache
        # GptOssModel.layers[i]
        x, residual = transformer_layer(
            layer_params[i],
            x,
            positions,
            layer_slot_mapping,
            attn_kv_cache,
            residual,
        )
        model._maybe_add_hidden_state(aux_hidden_states, i + 1, x, residual)

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({"hidden_states": x, "residual": residual})

    assert residual is not None
    # GptOssModel.norm
    x, _ = norm.forward_native(x, residual)

    if len(aux_hidden_states) > 0:
        return x, aux_hidden_states
    return x
