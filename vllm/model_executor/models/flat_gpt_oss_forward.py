# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn.functional as F
from flashinfer import trtllm_fp4_block_scale_moe
from flashinfer.tllm_enums import SfLayout

from vllm import _custom_ops as ops
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
from vllm.model_executor.models.flat_gpt_oss_kernels import (
    fused_add_rms_norm_mxfp8_quant,
    moe_finalize_top4,
    rope_and_cache,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import direct_register_custom_op


def _flashinfer_mxfp8_quantize_linear(
    x: torch.Tensor,
    alignment: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer.quantization.fp8_quantization import (
        get_mxfp8_quantization_sm100_module,
    )

    return get_mxfp8_quantization_sm100_module().mxfp8_quantize_sm100(
        x,
        SfLayout.layout_linear,
        alignment,
        True,
    )


def _flashinfer_mxfp8_quantize_linear_fake(
    x: torch.Tensor,
    alignment: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    padded_k = (x.shape[-1] + alignment - 1) // alignment * alignment
    num_rows = x.numel() // x.shape[-1]
    return (
        torch.empty(
            (*x.shape[:-1], padded_k),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        ),
        torch.empty(
            (num_rows * padded_k // 32,),
            dtype=torch.uint8,
            device=x.device,
        ),
    )


direct_register_custom_op(
    op_name="flashinfer_mxfp8_quantize_linear",
    op_func=_flashinfer_mxfp8_quantize_linear,
    fake_impl=_flashinfer_mxfp8_quantize_linear_fake,
)


def _flashinfer_trtllm_fp4_block_scale_moe(
    routing_logits: torch.Tensor,
    output_like: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: torch.Tensor,
    gemm1_alpha: torch.Tensor,
    gemm1_beta: torch.Tensor,
    gemm1_clamp_limit: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: torch.Tensor,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routing_method_type: int,
    tune_max_num_tokens: int,
) -> torch.Tensor:
    gemm2_output, expert_weights, expanded_idx_to_permuted_idx = (
        trtllm_fp4_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=None,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            gemm1_weights=gemm1_weights,
            gemm1_weights_scale=gemm1_weights_scale,
            gemm1_bias=gemm1_bias,
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
            gemm1_clamp_limit=gemm1_clamp_limit,
            gemm2_weights=gemm2_weights,
            gemm2_weights_scale=gemm2_weights_scale,
            gemm2_bias=gemm2_bias,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=num_experts,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=intermediate_size,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
            routed_scaling_factor=None,
            routing_method_type=routing_method_type,
            do_finalize=False,
            enable_pdl=True,
            tune_max_num_tokens=tune_max_num_tokens,
            output=None,
        )
    )
    return moe_finalize_top4(
        gemm2_output,
        expert_weights,
        expanded_idx_to_permuted_idx,
        output_like,
    )


def _flashinfer_trtllm_fp4_block_scale_moe_fake(
    routing_logits: torch.Tensor,
    output_like: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: torch.Tensor,
    gemm1_alpha: torch.Tensor,
    gemm1_beta: torch.Tensor,
    gemm1_clamp_limit: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: torch.Tensor,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routing_method_type: int,
    tune_max_num_tokens: int,
) -> torch.Tensor:
    return torch.empty_like(output_like)


direct_register_custom_op(
    op_name="flashinfer_trtllm_fp4_block_scale_moe",
    op_func=_flashinfer_trtllm_fp4_block_scale_moe,
    fake_impl=_flashinfer_trtllm_fp4_block_scale_moe_fake,
)


def transformer_layer(
    layer_params: tuple[Any, ...],
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    _layer_slot_mapping: torch.Tensor | None,
    _attn_kv_cache: torch.Tensor,
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
        _rotary_dim,
        attn_layer_name,
        attn_num_heads,
        attn_num_kv_heads,
        attn_head_size,
        attn_head_size_v,
        attn_k_scale,
        attn_v_scale,
        attn_q_scale,
        attn_query_uses_fp8,
        attn_kv_cache_dtype,
        _fp8_dtype,
        _fp8_min,
        _fp8_max,
        o_proj_weight,
        o_proj_bias,
        post_attention_norm_weight,
        post_attention_norm_eps,
        router_weight,
        router_bias,
        moe_w1,
        moe_w2,
        moe_w1_scale,
        moe_w2_scale,
        moe_w1_bias,
        moe_w2_bias,
        moe_gemm1_alpha,
        moe_gemm1_beta,
        moe_gemm1_clamp_limit,
        moe_global_num_experts,
        moe_topk,
        moe_intermediate_size,
        moe_local_expert_offset,
        moe_local_num_experts,
        moe_routing_method_type,
        moe_tune_max_num_tokens,
        hidden_size,
    ) = layer_params

    # TransformerBlock.input_layernorm
    if residual is None:
        residual = hidden_states
        hidden_states = torch.empty_like(hidden_states)
        ops.rms_norm(
            hidden_states,
            residual,
            input_norm_weight,
            input_norm_eps,
        )
    else:
        ops.fused_add_rms_norm(
            hidden_states,
            residual,
            input_norm_weight,
            input_norm_eps,
        )

    # TransformerBlock.attn.qkv_proj
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

    # TransformerBlock.attn.rotary_emb
    positions = torch.flatten(positions)
    num_tokens = positions.shape[0]
    cos_sin_cache = rotary_cos_sin_cache
    attn_output_dtype = q.dtype
    q_already_quantized = False
    if _layer_slot_mapping is None:
        cos_sin_cache = rotary_cos_sin_cache.to(dtype=q.dtype, device=q.device)
        q = q.contiguous()
        k = k.contiguous()
        ops.rotary_embedding(
            positions,
            q,
            k,
            rotary_head_size,
            cos_sin_cache,
            True,
        )
        kv_cache_dummy_dep = torch.empty(
            0,
            dtype=_attn_kv_cache.dtype,
            device=_attn_kv_cache.device,
        )
    else:
        q_fp8 = None
        if attn_query_uses_fp8:
            q_fp8 = torch.empty(q.shape, dtype=_fp8_dtype, device=q.device)
            q_already_quantized = True
        kv_cache_dummy_dep = rope_and_cache(
            q,
            k,
            v,
            _attn_kv_cache,
            _layer_slot_mapping,
            positions,
            cos_sin_cache,
            attn_kv_cache_dtype,
            attn_k_scale,
            attn_v_scale,
            attn_num_heads,
            attn_num_kv_heads,
            attn_head_size,
            _rotary_dim,
            q_fp8,
            attn_q_scale,
        )
        if q_fp8 is not None:
            q = q_fp8

    # TransformerBlock.attn.attn KV-cache write and attention
    if attn_query_uses_fp8 and not q_already_quantized:
        q, _ = ops.scaled_fp8_quant(q, attn_q_scale)

    q = torch.reshape(q, (num_tokens, attn_num_heads, attn_head_size))
    k = torch.reshape(k, (num_tokens, attn_num_kv_heads, attn_head_size))
    v = torch.reshape(v, (num_tokens, attn_num_kv_heads, attn_head_size_v))

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
        kv_cache_dummy_dep,
    )
    attn_output = torch.reshape(attn_output, (num_tokens, q_size))

    # TransformerBlock.attn.o_proj
    hidden_states = F.linear(attn_output, o_proj_weight, o_proj_bias)

    # TransformerBlock.post_attention_layernorm + MoE MXFP8 activation quant.
    moe_x_quant, moe_x_scale = fused_add_rms_norm_mxfp8_quant(
        hidden_states,
        residual,
        post_attention_norm_weight,
        post_attention_norm_eps,
        256,
    )

    # TransformerBlock.mlp.router
    router_logits = F.linear(hidden_states, router_weight, router_bias)

    # TransformerBlock.mlp.experts.forward_cuda
    # FusedMoE.runner.forward -> MoEPrepareAndFinalizeNoDPEPMonolithic.prepare
    moe_x_scale = moe_x_scale.view(torch.float8_e4m3fn).reshape(
        *hidden_states.shape[:-1],
        -1,
    )

    # TrtLlmMxfp4ExpertsMonolithic.apply
    output = torch.ops.vllm.flashinfer_trtllm_fp4_block_scale_moe(
        router_logits.to(torch.bfloat16),
        hidden_states,
        moe_x_quant,
        moe_x_scale,
        moe_w1,
        moe_w1_scale,
        moe_w1_bias,
        moe_gemm1_alpha,
        moe_gemm1_beta,
        moe_gemm1_clamp_limit,
        moe_w2,
        moe_w2_scale,
        moe_w2_bias,
        moe_global_num_experts,
        moe_topk,
        moe_intermediate_size,
        moe_local_expert_offset,
        moe_local_num_experts,
        moe_routing_method_type,
        moe_tune_max_num_tokens,
    )
    output = output[:, :hidden_size]

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
            experts = layer.mlp.experts
            expert_method = experts.quant_method
            moe_kernel = expert_method.moe_kernel
            if moe_kernel is None or not expert_method.is_monolithic:
                raise ValueError(
                    "Flat GPT-OSS is specialized for monolithic MXFP4 MoE"
                )
            fused_experts = moe_kernel.fused_experts
            if fused_experts.__class__.__name__ != "TrtLlmMxfp4ExpertsMonolithic":
                raise ValueError(
                    "Flat GPT-OSS is specialized for FlashInfer TRTLLM MXFP4 MoE"
                )
            if not fused_experts.use_mxfp8_input:
                raise ValueError(
                    "Flat GPT-OSS is specialized for MXFP8 MoE activations"
                )
            if moe_kernel.output_is_reduced():
                raise ValueError("Flat GPT-OSS is specialized for no MoE all-reduce")
            moe_parallel_config = experts.moe_config.moe_parallel_config
            if (
                moe_parallel_config.use_all2all_kernels
                or moe_parallel_config.dp_size > 1
                or moe_parallel_config.ep_size > 1
            ):
                raise ValueError("Flat GPT-OSS is specialized for no DP/EP MoE")
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
                    attn.kv_cache_dtype,
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
                    experts.w13_weight,
                    experts.w2_weight,
                    fused_experts.w1_scale,
                    fused_experts.w2_scale,
                    fused_experts.w1_bias,
                    fused_experts.w2_bias,
                    fused_experts.gemm1_alpha,
                    fused_experts.gemm1_beta,
                    fused_experts.gemm1_clamp_limit,
                    experts.global_num_experts,
                    fused_experts.topk,
                    fused_experts.intermediate_size_per_partition,
                    fused_experts.ep_rank * fused_experts.local_num_experts,
                    fused_experts.local_num_experts,
                    fused_experts.routing_method_type,
                    1,
                    layer.mlp.hidden_size,
                )
            )
        cached_params = (
            model.embedding,
            layer_params,
            model.norm.weight.data,
            model.norm.variance_epsilon,
        )
        model._flat_gpt_oss_params = cached_params
    embedding, layer_params, norm_weight, norm_eps = cached_params

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
    ops.fused_add_rms_norm(x, residual, norm_weight, norm_eps)

    if len(aux_hidden_states) > 0:
        return x, aux_hidden_states
    return x
