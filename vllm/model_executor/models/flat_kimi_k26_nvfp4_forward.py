# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flat forward for the Kimi K2.6 NVFP4 target serve path."""

from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.distributed import (
    get_pp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLP
from vllm.sequence import IntermediateTensors


def transformer_layer(
    layer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    llama_4_scaling: torch.Tensor | None,
    moe_quant_config,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Input RMSNorm and residual setup.
    if residual is None:
        residual = hidden_states.clone()
    else:
        hidden_states = hidden_states + residual
        residual = hidden_states
    norm_dtype = hidden_states.dtype
    norm_float = hidden_states.float()
    norm_var = norm_float.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = (
        norm_float
        * torch.rsqrt(norm_var + layer.input_layernorm.variance_epsilon)
    ).to(norm_dtype)
    hidden_states = (hidden_states * layer.input_layernorm.weight).to(norm_dtype)

    # Attention.
    wrapper = layer.self_attn.mla_attn

    # Attention qkv/LoRA projections and q/kv RMSNorm.
    if wrapper.fused_qkv_a_proj.weight.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise NotImplementedError("Flat Kimi K2.6 attention qkv must be unquantized")
    qkv_lora = F.linear(
        hidden_states,
        wrapper.fused_qkv_a_proj.weight,
        wrapper.fused_qkv_a_proj.bias,
    )
    if wrapper.fused_qkv_a_proj.gather_output and wrapper.fused_qkv_a_proj.tp_size > 1:
        qkv_lora = tensor_model_parallel_all_gather(qkv_lora)
    q_c, kv_lora = qkv_lora.split(
        [wrapper.q_lora_rank, wrapper.kv_lora_rank + wrapper.qk_rope_head_dim],
        dim=-1,
    )
    q_dtype = q_c.dtype
    q_float = q_c.float()
    q_var = q_float.pow(2).mean(dim=-1, keepdim=True)
    q_c = (
        q_float * torch.rsqrt(q_var + wrapper.q_a_layernorm.variance_epsilon)
    ).to(q_dtype)
    q_c = (q_c * wrapper.q_a_layernorm.weight).to(q_dtype)

    if wrapper.q_b_proj.weight.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise NotImplementedError("Flat Kimi K2.6 attention q_b must be unquantized")
    q = F.linear(q_c, wrapper.q_b_proj.weight, wrapper.q_b_proj.bias)
    if wrapper.q_b_proj.gather_output and wrapper.q_b_proj.tp_size > 1:
        q = tensor_model_parallel_all_gather(q)

    kv_c, k_pe = kv_lora.split([wrapper.kv_lora_rank, wrapper.qk_rope_head_dim], dim=-1)
    kv_dtype = kv_c.dtype
    kv_float = kv_c.float()
    kv_var = kv_float.pow(2).mean(dim=-1, keepdim=True)
    kv_c_normed = (
        kv_float * torch.rsqrt(kv_var + wrapper.kv_a_layernorm.variance_epsilon)
    ).to(kv_dtype)
    kv_c_normed = (kv_c_normed * wrapper.kv_a_layernorm.weight).to(kv_dtype)

    q = q.view(-1, wrapper.num_heads, wrapper.qk_head_dim)
    k_pe = k_pe.unsqueeze(1)

    # DeepSeek YaRN RoPE.
    rope = wrapper.rotary_emb
    query_rot = q[..., wrapper.qk_nope_head_dim:][..., :rope.rotary_dim]
    key_rot = k_pe[..., :rope.rotary_dim]
    cos_sin = rope.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
    query_even = query_rot[..., ::2]
    query_odd = query_rot[..., 1::2]
    query_rotated = torch.stack((-query_odd, query_even), dim=-1).flatten(-2)
    key_even = key_rot[..., ::2]
    key_odd = key_rot[..., 1::2]
    key_rotated = torch.stack((-key_odd, key_even), dim=-1).flatten(-2)
    q[..., wrapper.qk_nope_head_dim:] = query_rot * cos + query_rotated * sin
    k_pe = key_rot * cos + key_rotated * sin

    if wrapper.indexer and wrapper.is_sparse and not wrapper.skip_topk:
        raise NotImplementedError("Kimi K2.6 flat path does not support sparse MLA")
    if llama_4_scaling is not None:
        q *= llama_4_scaling

    # KV-cache write and active FlashInfer MLA attention backend.
    mla = wrapper.mla_attn
    forward_context = get_forward_context()
    attn_metadata_raw = forward_context.attn_metadata
    if isinstance(attn_metadata_raw, dict):
        attn_metadata = attn_metadata_raw[mla.layer_name]
    elif isinstance(attn_metadata_raw, list):
        attn_metadata = attn_metadata_raw[0][mla.layer_name]
    else:
        attn_metadata = attn_metadata_raw

    slot_mapping = forward_context.slot_mapping
    assert isinstance(slot_mapping, dict)
    layer_slot_mapping = slot_mapping.get(mla.layer_name)
    if mla.kv_cache.numel() != 0 and layer_slot_mapping is not None:
        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            mla.kv_cache,
            layer_slot_mapping.flatten(),
            kv_cache_dtype=mla.kv_cache_dtype,
            scale=mla._k_scale,
        )

    attn_output = torch.empty(
        (hidden_states.shape[0], wrapper.num_heads * wrapper.v_head_dim),
        dtype=q.dtype,
        device=q.device,
    )
    if attn_metadata is not None:
        fp8_attention = (
            mla.kv_cache_dtype.startswith("fp8")
            or mla.kv_cache_dtype.endswith("per_token_head")
            or mla.kv_cache_dtype == "nvfp4"
        )
        kv_cache = mla.kv_cache
        if fp8_attention and mla.kv_cache_dtype != "fp8_ds_mla":
            kv_cache = kv_cache.view(torch.float8_e4m3fn)

        num_actual_toks = attn_metadata.num_actual_tokens
        output_actual = attn_output[:num_actual_toks]
        q = q[:num_actual_toks]
        kv_c_normed = kv_c_normed[:num_actual_toks]
        k_pe = k_pe[:num_actual_toks]

        assert (
            attn_metadata.num_decodes is not None
            and attn_metadata.num_prefills is not None
            and attn_metadata.num_decode_tokens is not None
        )
        num_mqa_tokens = attn_metadata.num_decode_tokens
        num_mha_tokens = q.size(0) - num_mqa_tokens

        if num_mha_tokens > 0:
            from flashinfer.prefill import trtllm_ragged_attention_deepseek

            prefill_metadata = attn_metadata.prefill
            assert prefill_metadata is not None
            assert prefill_metadata.prefill_backend is not None
            has_context = prefill_metadata.chunked_context is not None

            q_prefill = q[num_mqa_tokens:]
            if wrapper.kv_b_proj.weight.dtype not in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ):
                raise NotImplementedError(
                    "Flat Kimi K2.6 attention kv_b must be unquantized"
                )
            kv_nope = F.linear(
                kv_c_normed[num_mqa_tokens:],
                wrapper.kv_b_proj.weight,
                wrapper.kv_b_proj.bias,
            )
            if wrapper.kv_b_proj.gather_output and wrapper.kv_b_proj.tp_size > 1:
                kv_nope = tensor_model_parallel_all_gather(kv_nope)
            kv_nope = kv_nope.view(
                -1,
                wrapper.num_heads,
                wrapper.qk_nope_head_dim + wrapper.v_head_dim,
            )
            k_nope, v = kv_nope.split(
                [wrapper.qk_nope_head_dim, wrapper.v_head_dim],
                dim=-1,
            )
            k = torch.cat(
                [
                    k_nope,
                    k_pe[num_mqa_tokens:].expand(-1, wrapper.num_heads, -1),
                ],
                dim=-1,
            )
            use_fp8_prefill = prefill_metadata.q_data_type == torch.float8_e4m3fn
            if use_fp8_prefill:
                q_prefill = q_prefill.to(prefill_metadata.q_data_type)
                k = k.to(prefill_metadata.q_data_type)
                v = v.to(prefill_metadata.q_data_type)

            backend = prefill_metadata.prefill_backend
            ret = trtllm_ragged_attention_deepseek(
                query=q_prefill,
                key=k,
                value=v,
                workspace_buffer=backend._workspace_buffer,
                seq_lens=backend._query_seq_lens,
                max_q_len=prefill_metadata.max_query_len,
                max_kv_len=prefill_metadata.max_query_len,
                bmm1_scale=backend.scale,
                bmm2_scale=1.0,
                o_sf_scale=1.0,
                batch_size=backend._query_seq_lens.shape[0],
                window_left=-1,
                cum_seq_lens_q=prefill_metadata.query_start_loc,
                cum_seq_lens_kv=prefill_metadata.query_start_loc,
                enable_pdl=False,
                is_causal=True,
                return_lse=has_context,
                out=output_actual[num_mqa_tokens:].view(
                    -1,
                    wrapper.num_heads,
                    wrapper.v_head_dim,
                ),
            )
            if has_context:
                assert isinstance(ret, tuple)
                suffix_output = ret[0]
                suffix_lse = ret[1].transpose(0, 1).contiguous()
                assert prefill_metadata.chunked_context is not None
                workspace = prefill_metadata.chunked_context.workspace
                context_output = None
                merge_output = None

                for chunk_idx in range(len(prefill_metadata.chunked_context.seq_tot)):
                    toks = prefill_metadata.chunked_context.seq_tot[chunk_idx]
                    if not use_fp8_prefill:
                        ops.gather_and_maybe_dequant_cache(
                            src_cache=kv_cache,
                            dst=workspace,
                            block_table=prefill_metadata.block_table,
                            cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[
                                chunk_idx
                            ],
                            token_to_seq=prefill_metadata.chunked_context.token_to_seq[
                                chunk_idx
                            ],
                            num_tokens=(
                                prefill_metadata.chunked_context.chunk_total_token[
                                    chunk_idx
                                ]
                            ),
                            kv_cache_dtype=mla.kv_cache_dtype,
                            scale=mla._k_scale,
                            seq_starts=(
                                prefill_metadata.chunked_context.starts[chunk_idx]
                            ),
                        )
                    else:
                        ops.cp_gather_cache(
                            src_cache=kv_cache,
                            dst=workspace,
                            block_table=prefill_metadata.block_table,
                            cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[
                                chunk_idx
                            ],
                            batch_size=attn_metadata.num_prefills,
                            seq_starts=(
                                prefill_metadata.chunked_context.starts[chunk_idx]
                            ),
                        )

                    context_kv_c = workspace[:toks][..., :wrapper.kv_lora_rank]
                    if use_fp8_prefill:
                        context_kv_c = context_kv_c.to(prefill_metadata.q_data_type)
                    else:
                        context_kv_c = context_kv_c.to(wrapper.kv_b_proj.weight.dtype)
                    context_k_pe = workspace[
                        :toks, ..., wrapper.kv_lora_rank:
                    ].unsqueeze(1)
                    context_kv_nope = F.linear(
                        context_kv_c,
                        wrapper.kv_b_proj.weight,
                        wrapper.kv_b_proj.bias,
                    )
                    if wrapper.kv_b_proj.gather_output and wrapper.kv_b_proj.tp_size > 1:
                        context_kv_nope = tensor_model_parallel_all_gather(
                            context_kv_nope
                        )
                    context_kv_nope = context_kv_nope.view(
                        -1,
                        wrapper.num_heads,
                        wrapper.qk_nope_head_dim + wrapper.v_head_dim,
                    )
                    if use_fp8_prefill:
                        context_kv_nope = context_kv_nope.to(
                            prefill_metadata.q_data_type
                        )
                        context_k_pe = context_k_pe.to(prefill_metadata.q_data_type)
                    context_k_nope, context_v = context_kv_nope.split(
                        [wrapper.qk_nope_head_dim, wrapper.v_head_dim],
                        dim=-1,
                    )
                    context_k = torch.empty(
                        (
                            context_k_nope.shape[0],
                            context_k_nope.shape[1],
                            context_k_nope.shape[2] + context_k_pe.shape[-1],
                        ),
                        dtype=context_k_nope.dtype,
                        device=context_k_nope.device,
                    )
                    context_k[..., :wrapper.qk_nope_head_dim] = context_k_nope
                    context_k[..., wrapper.qk_nope_head_dim:] = context_k_pe

                    chunk_output, chunk_lse = trtllm_ragged_attention_deepseek(
                        query=q_prefill,
                        key=context_k,
                        value=context_v,
                        workspace_buffer=backend._workspace_buffer,
                        seq_lens=prefill_metadata.chunked_context.seq_lens[
                            chunk_idx
                        ],
                        max_q_len=prefill_metadata.max_query_len,
                        max_kv_len=prefill_metadata.chunked_context.max_seq_lens[
                            chunk_idx
                        ],
                        bmm1_scale=backend.scale,
                        bmm2_scale=1.0,
                        o_sf_scale=1.0,
                        batch_size=prefill_metadata.chunked_context.seq_lens[
                            chunk_idx
                        ].shape[0],
                        window_left=-1,
                        cum_seq_lens_q=prefill_metadata.query_start_loc,
                        cum_seq_lens_kv=prefill_metadata.chunked_context.cu_seq_lens[
                            chunk_idx
                        ],
                        enable_pdl=False,
                        is_causal=False,
                        return_lse=True,
                        out=torch.empty(
                            q_prefill.shape[0],
                            q_prefill.shape[1],
                            context_v.shape[2],
                            device=q_prefill.device,
                            dtype=prefill_metadata.output_dtype,
                        ),
                    )
                    chunk_lse = chunk_lse.transpose(0, 1).contiguous()
                    if context_output is None:
                        context_output = chunk_output
                        context_lse = chunk_lse
                    else:
                        if merge_output is None:
                            merge_output = torch.empty_like(context_output)
                            merge_lse = torch.empty_like(context_lse)
                        merge_attn_states(
                            output=merge_output,
                            output_lse=merge_lse,
                            prefix_output=context_output,
                            prefix_lse=context_lse,
                            suffix_output=chunk_output,
                            suffix_lse=chunk_lse,
                        )
                        context_output, merge_output = merge_output, context_output
                        context_lse, merge_lse = merge_lse, context_lse

                assert context_output is not None
                output_view = output_actual[num_mqa_tokens:].view(
                    -1,
                    wrapper.num_heads,
                    wrapper.v_head_dim,
                )
                merge_attn_states(
                    output=output_view,
                    prefix_output=context_output,
                    prefix_lse=context_lse,
                    suffix_output=suffix_output,
                    suffix_lse=suffix_lse,
                    prefill_tokens_with_context=(
                        prefill_metadata.chunked_context.prefill_tokens_with_context
                    ),
                )
            elif isinstance(ret, torch.Tensor):
                output_actual[num_mqa_tokens:].copy_(ret.flatten(start_dim=-2))

        if num_mqa_tokens > 0:
            from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

            mqa_q = q[:num_mqa_tokens]
            mqa_q_nope, mqa_q_pe = mqa_q.split(
                [wrapper.qk_nope_head_dim, wrapper.qk_rope_head_dim],
                dim=-1,
            )
            mqa_q_nope = mqa_q_nope.transpose(0, 1)
            n_heads, batch_tokens, _ = mqa_q_nope.shape
            _, _, latent_rank = mla.W_UK_T.shape
            mqa_ql_nope = mqa_q_nope.new_empty((n_heads, batch_tokens, latent_rank))
            torch.bmm(mqa_q_nope, mla.W_UK_T, out=mqa_ql_nope)
            mqa_ql_nope = mqa_ql_nope.transpose(0, 1)
            if fp8_attention and mla.impl.supports_quant_query_input:
                decode_q = torch.cat([mqa_ql_nope, mqa_q_pe], dim=-1)
                decode_q_flat = decode_q.reshape(decode_q.shape[0], -1)
                decode_q_flat, _ = ops.scaled_fp8_quant(
                    decode_q_flat,
                    mla._q_scale,
                    group_shape=(-1, -1),
                )
                mqa_q = decode_q_flat.view(decode_q.shape)
            else:
                mqa_q = torch.cat([mqa_ql_nope, mqa_q_pe], dim=-1)
            if attn_metadata.num_decode_tokens % attn_metadata.num_decodes != 0:
                mqa_q = mqa_q.unsqueeze(1)
            else:
                mqa_q = mqa_q.view(
                    attn_metadata.num_decodes,
                    -1,
                    mqa_q.shape[-2],
                    mqa_q.shape[-1],
                )

            bmm1_scale = mla.impl.scale
            bmm2_scale = 1.0
            if fp8_attention:
                bmm1_scale *= mla._q_scale_float * mla._k_scale_float
                bmm2_scale *= mla._k_scale_float

            assert attn_metadata.decode is not None
            attn_out = trtllm_batch_decode_with_kv_cache_mla(
                query=mqa_q,
                kv_cache=kv_cache.unsqueeze(1),
                workspace_buffer=mla.impl._workspace_buffer,
                qk_nope_head_dim=wrapper.qk_nope_head_dim,
                kv_lora_rank=wrapper.kv_lora_rank,
                qk_rope_head_dim=wrapper.qk_rope_head_dim,
                block_tables=attn_metadata.decode.block_table,
                seq_lens=attn_metadata.decode.seq_lens,
                max_seq_len=attn_metadata.max_seq_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
            )
            attn_out = attn_out.view(-1, attn_out.shape[-2], attn_out.shape[-1])
            attn_out = attn_out.view(
                -1,
                wrapper.num_heads,
                wrapper.kv_lora_rank,
            ).transpose(0, 1)
            v_out = output_actual[:num_mqa_tokens].view(
                -1,
                wrapper.num_heads,
                wrapper.v_head_dim,
            )
            torch.bmm(attn_out, mla.W_UV, out=v_out.transpose(0, 1))
    else:
        attn_output.fill_(0)

    if wrapper.o_proj.weight.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise NotImplementedError("Flat Kimi K2.6 attention o_proj must be unquantized")
    if wrapper.o_proj.input_is_parallel:
        o_input = attn_output
    else:
        o_input = torch.chunk(attn_output, wrapper.o_proj.tp_size, dim=-1)[
            wrapper.o_proj.tp_rank
        ].contiguous()
    hidden_states = F.linear(
        o_input,
        wrapper.o_proj.weight,
        wrapper.o_proj.bias,
    )
    if wrapper.o_proj.reduce_results and wrapper.o_proj.tp_size > 1:
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)

    if hidden_states.dtype == torch.float16:
        hidden_states *= 1.0 / layer.routed_scaling_factor
        if layer.layer_idx == 0:
            residual *= 1.0 / layer.routed_scaling_factor

    # Post-attention RMSNorm.
    hidden_states = hidden_states + residual
    residual = hidden_states
    norm_dtype = hidden_states.dtype
    norm_float = hidden_states.float()
    norm_var = norm_float.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = (
        norm_float
        * torch.rsqrt(norm_var + layer.post_attention_layernorm.variance_epsilon)
    ).to(norm_dtype)
    hidden_states = (hidden_states * layer.post_attention_layernorm.weight).to(
        norm_dtype
    )

    # MLP router/experts.
    if isinstance(layer.mlp, DeepseekV2MLP):
        mlp = layer.mlp
        if mlp.gate_up_proj.weight.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise NotImplementedError("Flat Kimi K2.6 dense MLP must be unquantized")
        gate_up = F.linear(
            hidden_states,
            mlp.gate_up_proj.weight,
            mlp.gate_up_proj.bias,
        )
        if mlp.gate_up_proj.gather_output and mlp.gate_up_proj.tp_size > 1:
            gate_up = tensor_model_parallel_all_gather(gate_up)
        gate, up = gate_up.chunk(2, dim=-1)
        mlp_intermediate = F.silu(gate) * up
        if mlp.down_proj.weight.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise NotImplementedError("Flat Kimi K2.6 dense MLP must be unquantized")
        if mlp.down_proj.input_is_parallel:
            mlp_input = mlp_intermediate
        else:
            mlp_input = torch.chunk(
                mlp_intermediate,
                mlp.down_proj.tp_size,
                dim=-1,
            )[mlp.down_proj.tp_rank].contiguous()
        hidden_states = F.linear(
            mlp_input,
            mlp.down_proj.weight,
            mlp.down_proj.bias,
        )
        if mlp.down_proj.reduce_results and mlp.down_proj.tp_size > 1:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        if hidden_states.dtype == torch.float16:
            hidden_states *= 1.0 / layer.routed_scaling_factor
    else:
        moe = layer.mlp
        routed = moe.experts
        quant_config = moe_quant_config
        assert quant_config is not None
        assert routed.quant_method.is_monolithic

        # Shared experts.
        shared_output = None
        if moe.shared_experts is not None:
            shared = moe.shared_experts
            if shared.gate_up_proj.weight.dtype not in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ):
                raise NotImplementedError(
                    "Flat Kimi K2.6 shared experts must be unquantized"
                )
            shared_gate_up = F.linear(
                hidden_states,
                shared.gate_up_proj.weight,
                shared.gate_up_proj.bias,
            )
            if shared.gate_up_proj.gather_output and shared.gate_up_proj.tp_size > 1:
                shared_gate_up = tensor_model_parallel_all_gather(shared_gate_up)
            shared_gate, shared_up = shared_gate_up.chunk(2, dim=-1)
            shared_intermediate = F.silu(shared_gate) * shared_up
            if shared.down_proj.weight.dtype not in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ):
                raise NotImplementedError(
                    "Flat Kimi K2.6 shared experts must be unquantized"
                )
            if shared.down_proj.input_is_parallel:
                shared_input = shared_intermediate
            else:
                shared_input = torch.chunk(
                    shared_intermediate,
                    shared.down_proj.tp_size,
                    dim=-1,
                )[shared.down_proj.tp_rank].contiguous()
            shared_output = F.linear(
                shared_input,
                shared.down_proj.weight,
                shared.down_proj.bias,
            )
            if shared.down_proj.reduce_results and shared.down_proj.tp_size > 1:
                shared_output = tensor_model_parallel_all_reduce(shared_output)

        # Router.
        router_logits = F.linear(hidden_states, moe.gate.weight)
        if moe.gate.out_dtype is not None:
            router_logits = router_logits.to(moe.gate.out_dtype)

        # NVFP4 activation quantization and FlashInfer TRT-LLM FP4 routed experts.
        if quant_config.quant_dtype != "nvfp4":
            raise NotImplementedError("Flat Kimi K2.6 target expects NVFP4 MoE")
        input_sf = (
            quant_config.a1_gscale
            if quant_config.use_nvfp4_w4a4
            else quant_config.a1_scale
        )
        a1q, a1q_scale = ops.scaled_fp4_quant(
            hidden_states,
            input_sf,
            is_sf_swizzled_layout=quant_config.is_scale_swizzled,
        )
        assert a1q_scale is not None
        assert quant_config.w1_scale is not None
        assert quant_config.w2_scale is not None

        import flashinfer

        correction_bias = moe.gate.e_score_correction_bias
        if correction_bias is not None:
            correction_bias = correction_bias.to(torch.bfloat16)
        if routed.activation.value != "silu":
            raise NotImplementedError("Flat Kimi K2.6 target expects SwiGLU MoE")

        fused_output = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
            routing_logits=router_logits,
            routing_bias=correction_bias,
            hidden_states=a1q,
            hidden_states_scale=a1q_scale.view(torch.float8_e4m3fn).reshape(
                *a1q.shape[:-1],
                -1,
            ),
            gemm1_weights=routed.w13_weight,
            gemm1_weights_scale=quant_config.w1_scale.view(torch.float8_e4m3fn),
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=routed.w2_weight,
            gemm2_weights_scale=quant_config.w2_scale.view(torch.float8_e4m3fn),
            gemm2_bias=None,
            output1_scale_scalar=routed.g1_scale_c,
            output1_scale_gate_scalar=quant_config.g1_alphas,
            output2_scale_scalar=quant_config.g2_alphas,
            num_experts=routed.global_num_experts,
            top_k=routed.top_k,
            n_group=(routed.num_expert_group or 0),
            topk_group=(routed.topk_group or 0),
            intermediate_size=routed.moe_config.intermediate_size_per_partition,
            local_expert_offset=routed.ep_rank * routed.local_num_experts,
            local_num_experts=routed.local_num_experts,
            routed_scaling_factor=routed.routed_scaling_factor,
            routing_method_type=RoutingMethodType.DeepSeekV3,
            do_finalize=True,
            activation_type=3,
        )[0]

        fused_output = fused_output[..., :hidden_states.shape[-1]]
        if moe.routed_scaling_factor != 1.0:
            fused_output = fused_output * moe.routed_scaling_factor

        hidden_states = (
            fused_output if shared_output is None else shared_output + fused_output
        )
        if (
            not routed.moe_config.is_sequence_parallel
            and (routed.moe_config.tp_size > 1 or routed.moe_config.ep_size > 1)
        ):
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)

    return hidden_states, residual


def flat_forward(
    model,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors:
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            if input_ids is None:
                raise ValueError(
                    "Either input_ids or inputs_embeds must be provided "
                    "to FlatDeepseekV2Model.forward"
                )
            # Vocab-parallel embedding.
            embed = model.embed_tokens
            if embed.tp_size > 1:
                org_mask = (input_ids >= embed.shard_indices.org_vocab_start_index) & (
                    input_ids < embed.shard_indices.org_vocab_end_index
                )
                added_mask = (
                    input_ids >= embed.shard_indices.added_vocab_start_index
                ) & (input_ids < embed.shard_indices.added_vocab_end_index)
                added_offset = (
                    embed.shard_indices.added_vocab_start_index
                    - (
                        embed.shard_indices.org_vocab_end_index
                        - embed.shard_indices.org_vocab_start_index
                    )
                    - embed.shard_indices.num_org_vocab_padding
                )
                valid_offset = (
                    embed.shard_indices.org_vocab_start_index * org_mask
                    + added_offset * added_mask
                )
                vocab_mask = org_mask | added_mask
                masked_input = vocab_mask * (input_ids - valid_offset)
            else:
                masked_input = input_ids
                vocab_mask = None
            hidden_states = F.embedding(masked_input.long(), embed.weight)
            if vocab_mask is not None:
                hidden_states.masked_fill_((~vocab_mask).unsqueeze(-1), 0)
                hidden_states = tensor_model_parallel_all_reduce(hidden_states)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    llama_4_scaling_config = getattr(model.config, "llama_4_scaling", None)
    if llama_4_scaling_config is not None:
        raise NotImplementedError("Flat Kimi K2.6 target does not use llama_4 scaling")
    llama_4_scaling = None

    if not hasattr(model, "_flat_kimi_moe_quant_configs"):
        flat_moe_quant_configs = {}
        for layer_idx, cached_layer in enumerate(model.layers):
            if isinstance(cached_layer.mlp, DeepseekV2MLP):
                flat_moe_quant_configs[layer_idx] = None
            else:
                routed = cached_layer.mlp.experts
                if routed.quant_method.moe_quant_config is None:
                    routed.quant_method.moe_quant_config = (
                        routed.quant_method.get_fused_moe_quant_config(routed)
                    )
                flat_moe_quant_configs[layer_idx] = (
                    routed.quant_method.moe_quant_config
                )
        model._flat_kimi_moe_quant_configs = flat_moe_quant_configs

    aux_hidden_states = []
    layer_iter: Iterable = islice(model.layers, model.start_layer, model.end_layer)
    for idx, layer in enumerate(layer_iter, start=model.start_layer):
        if idx in model.aux_hidden_state_layers:
            aux_hidden_states.append(hidden_states + residual)
        hidden_states, residual = transformer_layer(
            layer,
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
            model._flat_kimi_moe_quant_configs[idx],
        )

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

    # Final norm.
    hidden_states = hidden_states + residual
    norm_dtype = hidden_states.dtype
    norm_float = hidden_states.float()
    norm_var = norm_float.pow(2).mean(dim=-1, keepdim=True)
    hidden_states = (
        norm_float * torch.rsqrt(norm_var + model.norm.variance_epsilon)
    ).to(norm_dtype)
    hidden_states = (hidden_states * model.norm.weight).to(norm_dtype)
    if len(aux_hidden_states) > 0:
        return hidden_states, aux_hidden_states
    return hidden_states
