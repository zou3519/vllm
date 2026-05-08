# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flat forward pass for DeepSeek V3.2 NVFP4."""

from itertools import islice

import torch
import deep_gemm
import torch.nn.functional as F

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsMonolithic,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)


_FI_SPARSE_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024
_fi_sparse_workspace: torch.Tensor | None = None


def transformer_layer(
    layer,
    positions,
    hidden_states,
    residual,
    forward_context,
    tp_group_name,
):
    global _fi_sparse_workspace

    # Input RMSNorm and residual.
    if residual is None:
        residual = hidden_states.clone()
        norm = layer.input_layernorm
        hidden_states = torch.empty_like(hidden_states)
        ops.rms_norm(
            hidden_states,
            residual,
            norm.weight.data,
            norm.variance_epsilon,
        )
    else:
        norm = layer.input_layernorm
        ops.fused_add_rms_norm(
            hidden_states,
            residual,
            norm.weight.data,
            norm.variance_epsilon,
        )

    # MLA fused q/kv A projection.
    attn = layer.self_attn
    wrapper = attn.mla_attn
    fused_qkv_a_proj = wrapper.fused_qkv_a_proj
    if getattr(fused_qkv_a_proj, "_use_min_latency_gemm", False):
        qkv_lora = torch.ops.vllm.min_latency_fused_qkv_a_proj(
            hidden_states, fused_qkv_a_proj.weight
        )
    else:
        bias = fused_qkv_a_proj.bias if not fused_qkv_a_proj.skip_bias_add else None
        if hasattr(fused_qkv_a_proj, "weight_global_scale") and hasattr(
            fused_qkv_a_proj, "input_global_scale_inv"
        ):
            qkv_shape = [
                *hidden_states.shape[:-1],
                fused_qkv_a_proj.output_size_per_partition,
            ]
            x_fp4, x_blockscale = ops.scaled_fp4_quant(
                hidden_states,
                fused_qkv_a_proj.input_global_scale_inv,
                is_sf_swizzled_layout=True,
                backend=fused_qkv_a_proj.quant_method.backend.value,
            )
            if fused_qkv_a_proj.weights_padding_cols > 0:
                x_fp4 = F.pad(
                    x_fp4, (0, fused_qkv_a_proj.weights_padding_cols)
                ).contiguous()
            backend_name = fused_qkv_a_proj.quant_method.backend.value[
                len("flashinfer-") :
            ]
            block_scale_a = x_blockscale.view(torch.uint8)
            block_scale_b = fused_qkv_a_proj.weight_scale.view(torch.uint8)
            qkv_lora = torch.ops.vllm.flashinfer_mm_fp4(
                x_fp4,
                fused_qkv_a_proj.weight.t(),
                block_scale_a,
                block_scale_b.t(),
                fused_qkv_a_proj.alpha,
                hidden_states.dtype,
                backend_name == "trtllm" and x_fp4.shape[0] <= 32,
                backend_name,
            )
            if qkv_lora.shape[-1] != fused_qkv_a_proj.output_size_per_partition:
                qkv_lora = qkv_lora[
                    ..., : fused_qkv_a_proj.output_size_per_partition
                ].contiguous()
            qkv_lora = qkv_lora.view(*qkv_shape)
            if bias is not None:
                qkv_lora = qkv_lora + bias
        else:
            qkv_lora = F.linear(hidden_states, fused_qkv_a_proj.weight, bias)

    q_c, kv_lora = qkv_lora.split(
        [wrapper.q_lora_rank, wrapper.kv_lora_rank + wrapper.qk_rope_head_dim],
        dim=-1,
    )

    # q_a RMSNorm.
    norm = wrapper.q_a_layernorm
    q_c_normed = torch.empty_like(q_c)
    ops.rms_norm(
        q_c_normed,
        q_c,
        norm.weight.data,
        norm.variance_epsilon,
    )
    q_c = q_c_normed

    # q_b projection.
    q_b_proj = wrapper.q_b_proj
    bias = q_b_proj.bias if not q_b_proj.skip_bias_add else None
    if hasattr(q_b_proj, "weight_global_scale") and hasattr(
        q_b_proj, "input_global_scale_inv"
    ):
        q_shape = [*q_c.shape[:-1], q_b_proj.output_size_per_partition]
        x_fp4, x_blockscale = ops.scaled_fp4_quant(
            q_c,
            q_b_proj.input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend=q_b_proj.quant_method.backend.value,
        )
        if q_b_proj.weights_padding_cols > 0:
            x_fp4 = F.pad(x_fp4, (0, q_b_proj.weights_padding_cols)).contiguous()
        backend_name = q_b_proj.quant_method.backend.value[len("flashinfer-") :]
        q = torch.ops.vllm.flashinfer_mm_fp4(
            x_fp4,
            q_b_proj.weight.t(),
            x_blockscale.view(torch.uint8),
            q_b_proj.weight_scale.view(torch.uint8).t(),
            q_b_proj.alpha,
            q_c.dtype,
            backend_name == "trtllm" and x_fp4.shape[0] <= 32,
            backend_name,
        )
        if q.shape[-1] != q_b_proj.output_size_per_partition:
            q = q[..., : q_b_proj.output_size_per_partition].contiguous()
        q = q.view(*q_shape)
        if bias is not None:
            q = q + bias
    else:
        q = F.linear(q_c, q_b_proj.weight, bias)
    if q_b_proj.gather_output and q_b_proj.tp_size > 1:
        gathered_q = [torch.empty_like(q) for _ in range(q_b_proj.tp_size)]
        torch.distributed.all_gather(gathered_q, q)
        q = torch.cat(gathered_q, dim=-1)

    # kv_a RMSNorm.
    kv_c, k_pe = kv_lora.split(
        [wrapper.kv_lora_rank, wrapper.qk_rope_head_dim], dim=-1
    )
    norm = wrapper.kv_a_layernorm
    kv_c_normed = torch.empty_like(kv_c)
    ops.rms_norm(
        kv_c_normed,
        kv_c,
        norm.weight.data,
        norm.variance_epsilon,
    )

    # MLA RoPE.
    q = q.view(-1, wrapper.num_heads, wrapper.qk_head_dim)
    k_pe = k_pe.unsqueeze(1)
    rotary = wrapper.rotary_emb
    q_rot = q[..., wrapper.qk_nope_head_dim :]
    cos_sin_cache = rotary.cos_sin_cache
    if cos_sin_cache.device != q.device or cos_sin_cache.dtype != q.dtype:
        cos_sin_cache = cos_sin_cache.to(q.device, dtype=q.dtype)
    ops.rotary_embedding(
        positions.flatten(),
        q_rot,
        k_pe,
        wrapper.qk_rope_head_dim,
        cos_sin_cache,
        False,
    )

    # Sparse indexer q projection.
    if wrapper.indexer and wrapper.is_sparse:
        indexer = wrapper.indexer
        wq_b = indexer.wq_b
        bias = wq_b.bias if not wq_b.skip_bias_add else None
        if hasattr(wq_b, "weight_global_scale") and hasattr(
            wq_b, "input_global_scale_inv"
        ):
            index_q_shape = [*q_c.shape[:-1], wq_b.output_size_per_partition]
            x_fp4, x_blockscale = ops.scaled_fp4_quant(
                q_c,
                wq_b.input_global_scale_inv,
                is_sf_swizzled_layout=True,
                backend=wq_b.quant_method.backend.value,
            )
            if wq_b.weights_padding_cols > 0:
                x_fp4 = F.pad(x_fp4, (0, wq_b.weights_padding_cols)).contiguous()
            backend_name = wq_b.quant_method.backend.value[len("flashinfer-") :]
            index_q = torch.ops.vllm.flashinfer_mm_fp4(
                x_fp4,
                wq_b.weight.t(),
                x_blockscale.view(torch.uint8),
                wq_b.weight_scale.view(torch.uint8).t(),
                wq_b.alpha,
                q_c.dtype,
                backend_name == "trtllm" and x_fp4.shape[0] <= 32,
                backend_name,
            )
            if index_q.shape[-1] != wq_b.output_size_per_partition:
                index_q = index_q[..., : wq_b.output_size_per_partition].contiguous()
            index_q = index_q.view(*index_q_shape)
            if bias is not None:
                index_q = index_q + bias
        else:
            index_q = F.linear(q_c, wq_b.weight, bias)
        index_q = index_q.view(-1, indexer.n_head, indexer.head_dim)
        q_pe, q_nope = torch.split(
            index_q,
            [indexer.rope_dim, indexer.head_dim - indexer.rope_dim],
            dim=-1,
        )

        # Sparse indexer fused wk + weights projection.
        if indexer.is_fp4_ckpt:
            wk_weights_proj = indexer.wk_weights_proj
            bias = (
                wk_weights_proj.bias
                if not wk_weights_proj.skip_bias_add
                else None
            )
            if hasattr(wk_weights_proj, "weight_global_scale") and hasattr(
                wk_weights_proj, "input_global_scale_inv"
            ):
                kw_shape = [
                    *hidden_states.shape[:-1],
                    wk_weights_proj.output_size_per_partition,
                ]
                x_fp4, x_blockscale = ops.scaled_fp4_quant(
                    hidden_states,
                    wk_weights_proj.input_global_scale_inv,
                    is_sf_swizzled_layout=True,
                    backend=wk_weights_proj.quant_method.backend.value,
                )
                if wk_weights_proj.weights_padding_cols > 0:
                    x_fp4 = F.pad(
                        x_fp4, (0, wk_weights_proj.weights_padding_cols)
                    ).contiguous()
                backend_name = wk_weights_proj.quant_method.backend.value[
                    len("flashinfer-") :
                ]
                kw = torch.ops.vllm.flashinfer_mm_fp4(
                    x_fp4,
                    wk_weights_proj.weight.t(),
                    x_blockscale.view(torch.uint8),
                    wk_weights_proj.weight_scale.view(torch.uint8).t(),
                    wk_weights_proj.alpha,
                    hidden_states.dtype,
                    backend_name == "trtllm" and x_fp4.shape[0] <= 32,
                    backend_name,
                )
                if kw.shape[-1] != wk_weights_proj.output_size_per_partition:
                    kw = kw[
                        ..., : wk_weights_proj.output_size_per_partition
                    ].contiguous()
                kw = kw.view(*kw_shape)
                if bias is not None:
                    kw = kw + bias
            else:
                kw = F.linear(hidden_states, wk_weights_proj.weight, bias)
            index_k = kw[:, : indexer.head_dim]
            index_weights = kw[:, indexer.head_dim :]
        else:
            wk = indexer.wk
            bias = wk.bias if not wk.skip_bias_add else None
            if hasattr(wk, "weight_global_scale") and hasattr(
                wk, "input_global_scale_inv"
            ):
                index_k_shape = [
                    *hidden_states.shape[:-1],
                    wk.output_size_per_partition,
                ]
                x_fp4, x_blockscale = ops.scaled_fp4_quant(
                    hidden_states,
                    wk.input_global_scale_inv,
                    is_sf_swizzled_layout=True,
                    backend=wk.quant_method.backend.value,
                )
                if wk.weights_padding_cols > 0:
                    x_fp4 = F.pad(x_fp4, (0, wk.weights_padding_cols)).contiguous()
                backend_name = wk.quant_method.backend.value[len("flashinfer-") :]
                index_k = torch.ops.vllm.flashinfer_mm_fp4(
                    x_fp4,
                    wk.weight.t(),
                    x_blockscale.view(torch.uint8),
                    wk.weight_scale.view(torch.uint8).t(),
                    wk.alpha,
                    hidden_states.dtype,
                    backend_name == "trtllm" and x_fp4.shape[0] <= 32,
                    backend_name,
                )
                if index_k.shape[-1] != wk.output_size_per_partition:
                    index_k = index_k[..., : wk.output_size_per_partition].contiguous()
                index_k = index_k.view(*index_k_shape)
                if bias is not None:
                    index_k = index_k + bias
            else:
                index_k = F.linear(hidden_states, wk.weight, bias)

            weights_proj = indexer.weights_proj
            bias = weights_proj.bias if not weights_proj.skip_bias_add else None
            index_weights = F.linear(hidden_states, weights_proj.weight, bias)

        # Sparse indexer K LayerNorm and RoPE.
        index_k = F.layer_norm(
            index_k.float(),
            (indexer.k_norm.dim,),
            indexer.k_norm.weight,
            indexer.k_norm.bias,
            indexer.k_norm.eps,
        ).type_as(index_k)
        k_pe_index, k_nope = torch.split(
            index_k,
            [indexer.rope_dim, indexer.head_dim - indexer.rope_dim],
            dim=-1,
        )
        rotary = wrapper.indexer_rope_emb
        k_pe_index = k_pe_index.unsqueeze(1)
        cos_sin_cache = rotary.cos_sin_cache
        if cos_sin_cache.device != q_pe.device or cos_sin_cache.dtype != q_pe.dtype:
            cos_sin_cache = cos_sin_cache.to(q_pe.device, dtype=q_pe.dtype)
        ops.rotary_embedding(
            positions.flatten(),
            q_pe,
            k_pe_index,
            indexer.rope_dim,
            cos_sin_cache,
            True,
        )
        index_q = torch.cat([q_pe, q_nope], dim=-1)
        index_k = torch.cat([k_pe_index.squeeze(-2), k_nope], dim=-1)

        # Sparse indexer fp8 q quantization and weight scaling.
        q_flat = index_q.view(-1, indexer.head_dim)
        q_fp8 = torch.empty(
            q_flat.shape,
            device=q_flat.device,
            dtype=torch.float8_e4m3fn,
        )
        q_scale = torch.empty(
            q_flat.shape[:-1] + (q_flat.shape[-1] // indexer.quant_block_size,),
            device=q_flat.device,
            dtype=torch.float32,
        )
        torch.ops._C.per_token_group_fp8_quant(
            q_flat,
            q_fp8,
            q_scale,
            indexer.quant_block_size,
            1e-10,
            -448.0,
            448.0,
            indexer.scale_fmt is not None,
            False,
            False,
        )
        q_fp8 = q_fp8.view(-1, indexer.n_head, indexer.head_dim)
        q_scale = q_scale.view(-1, indexer.n_head, 1)
        index_weights = (
            index_weights.unsqueeze(-1)
            * q_scale
            * indexer.softmax_scale
            * indexer.n_head**-0.5
        ).squeeze(-1)

        # Sparse indexer: profile allocation path.
        attn_metadata = forward_context.attn_metadata
        fp8_dtype = torch.float8_e4m3fn
        if not isinstance(attn_metadata, dict):
            _ = torch.empty(
                (indexer.max_total_seq_len, indexer.head_dim),
                dtype=torch.float8_e4m3fn,
                device=hidden_states.device,
            )
            _ = torch.empty(
                (indexer.max_total_seq_len, 4),
                dtype=torch.uint8,
                device=hidden_states.device,
            )
            max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
            _ = torch.empty(
                max_logits_elems, dtype=torch.uint8, device=hidden_states.device
            )
        else:
            # Sparse indexer: fp8 K quantization and KV-cache write.
            index_metadata = attn_metadata[indexer.k_cache.prefix]
            assert isinstance(index_metadata, DeepseekV32IndexerMetadata)
            slot_mapping = index_metadata.slot_mapping
            has_decode = index_metadata.num_decodes > 0
            has_prefill = index_metadata.num_prefills > 0
            num_decode_tokens = index_metadata.num_decode_tokens
            index_k = index_k[: slot_mapping.shape[0]]
            ops.indexer_k_quant_and_cache(
                index_k,
                indexer.k_cache.kv_cache,
                slot_mapping,
                indexer.quant_block_size,
                indexer.scale_fmt,
            )

            topk_indices_buffer = indexer.topk_indices_buffer

            # Sparse indexer: prefill MQA logits and per-row top-k.
            if has_prefill:
                assert index_metadata.prefill is not None
                k_fp8_full = torch.empty(
                    (indexer.max_total_seq_len, indexer.head_dim),
                    dtype=fp8_dtype,
                    device=hidden_states.device,
                )
                k_scale_full = torch.empty(
                    (indexer.max_total_seq_len, 4),
                    dtype=torch.uint8,
                    device=hidden_states.device,
                )
                for chunk in index_metadata.prefill.chunks:
                    k_fp8 = k_fp8_full[: chunk.total_seq_lens]
                    k_scale = k_scale_full[: chunk.total_seq_lens]
                    if not chunk.skip_kv_gather:
                        ops.cp_gather_indexer_k_quant_cache(
                            indexer.k_cache.kv_cache,
                            k_fp8,
                            k_scale,
                            chunk.block_table,
                            chunk.cu_seq_lens,
                        )

                    q_prefill = q_fp8[
                        chunk.token_start : chunk.token_end
                    ].to(torch.float32)
                    k_prefill = k_fp8.to(torch.float32) * k_scale.view(
                        torch.float32
                    ).flatten().unsqueeze(-1)
                    weights_prefill = index_weights[
                        chunk.token_start : chunk.token_end
                    ]
                    logits = torch.einsum(
                        "mhd,nd,mh->mn",
                        q_prefill,
                        k_prefill,
                        weights_prefill,
                    )
                    topk_indices = topk_indices_buffer[
                        chunk.token_start : chunk.token_end, : indexer.topk_tokens
                    ]
                    torch.ops._C.top_k_per_row_prefill(
                        logits,
                        chunk.cu_seqlen_ks,
                        chunk.cu_seqlen_ke,
                        topk_indices,
                        logits.shape[0],
                        logits.stride(0),
                        logits.stride(1),
                        indexer.topk_tokens,
                    )

            # Sparse indexer: decode paged MQA logits and top-k.
            if has_decode:
                decode_metadata = index_metadata.decode
                assert decode_metadata is not None
                decode_lens = decode_metadata.decode_lens
                assert not decode_metadata.requires_padding
                padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_fp8.shape[1:]
                )

                batch_size = padded_q_fp8_decode_tokens.shape[0]
                next_n = padded_q_fp8_decode_tokens.shape[1]
                num_padded_tokens = batch_size * next_n
                logits = deep_gemm.fp8_paged_mqa_logits(
                    padded_q_fp8_decode_tokens,
                    indexer.k_cache.kv_cache.unsqueeze(-2),
                    index_weights[:num_padded_tokens],
                    decode_metadata.seq_lens,
                    decode_metadata.block_table,
                    decode_metadata.schedule_metadata,
                    indexer.max_model_len,
                    clean_logits=False,
                )
                topk_indices = topk_indices_buffer[
                    :num_padded_tokens, : indexer.topk_tokens
                ]
                if decode_metadata.use_large_context_topk:
                    assert next_n == 1
                    lengths = decode_metadata.seq_lens
                    torch.ops._C.large_context_topk(
                        logits, topk_indices, lengths, None
                    )
                else:
                    torch.ops._C.top_k_per_row_decode(
                        logits,
                        next_n,
                        decode_metadata.seq_lens,
                        topk_indices,
                        logits.shape[0],
                        logits.stride(0),
                        logits.stride(1),
                        indexer.topk_tokens,
                    )

    # MLA KV-cache write.
    mla = wrapper.mla_attn
    if mla.calculate_kv_scales:
        torch.ops.vllm.maybe_calc_kv_scales(q, kv_c_normed, k_pe, mla.layer_name)
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[mla.layer_name]
    slot_mapping = forward_context.slot_mapping
    assert isinstance(slot_mapping, dict)
    if mla.kv_cache.numel() != 0:
        layer_slot_mapping = slot_mapping.get(mla.layer_name)
        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            mla.kv_cache,
            layer_slot_mapping.flatten(),
            kv_cache_dtype=mla.kv_cache_dtype,
            scale=mla._k_scale,
        )

    output_shape = (hidden_states.shape[0], wrapper.num_heads * wrapper.v_head_dim)
    attn_output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
    if attn_metadata is None:
        attn_output.fill_(0)
    else:
        num_actual_toks = attn_metadata.num_actual_tokens
        q = q[:num_actual_toks]
        attn_output_actual = attn_output[:num_actual_toks]

        # MLA decode q nope projection using preprocessed W_UK_T.
        mqa_q_nope, mqa_q_pe = q.split(
            [wrapper.qk_nope_head_dim, wrapper.qk_rope_head_dim], dim=-1
        )
        mqa_q_nope = mqa_q_nope.transpose(0, 1)
        if mla.q_pad_num_heads is not None:
            bsz, heads, rope_dim = mqa_q_pe.shape
            mqa_pe_padded = mqa_q_pe.new_empty(
                (bsz, mla.q_pad_num_heads, rope_dim)
            )
            mqa_pe_padded.resize_((bsz, heads, rope_dim))
            mqa_pe_padded.copy_(mqa_q_pe)
            mqa_q_pe = mqa_pe_padded

        heads, batch, _ = mqa_q_nope.shape
        _, _, lora_rank = mla.W_UK_T.shape
        if mla.q_pad_num_heads is not None:
            mqa_ql_nope = mqa_q_nope.new_empty(
                (mla.q_pad_num_heads, batch, lora_rank)
            )
            mqa_ql_nope.resize_((heads, batch, lora_rank))
        else:
            mqa_ql_nope = mqa_q_nope.new_empty((heads, batch, lora_rank))
        torch.bmm(mqa_q_nope, mla.W_UK_T, out=mqa_ql_nope)
        mqa_ql_nope = mqa_ql_nope.transpose(0, 1)

        decode_q0 = torch.cat((mqa_ql_nope, mqa_q_pe), dim=-1)
        decode_q_flat = decode_q0.reshape(decode_q0.shape[0], -1)
        mqa_q, _ = ops.scaled_fp8_quant(
            decode_q_flat,
            mla._q_scale,
            group_shape=(-1, -1),
        )
        mqa_q = mqa_q.view(decode_q0.shape)

        # FlashInfer sparse MLA decode.
        impl = mla.impl
        num_actual_toks = mqa_q.shape[0]
        assert impl.topk_indices_buffer is not None
        topk_indices = impl.topk_indices_buffer[:num_actual_toks]
        req_id = attn_metadata.req_id_per_token[:num_actual_toks]
        block_table = attn_metadata.block_table
        block_size = attn_metadata.block_size
        topk_indices_physical, seq_lens = triton_convert_req_index_to_global_index(
            req_id,
            block_table,
            topk_indices,
            BLOCK_SIZE=block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )
        if impl._workspace_buffer is None:
            if _fi_sparse_workspace is None:
                _fi_sparse_workspace = torch.zeros(
                    _FI_SPARSE_WORKSPACE_BUFFER_SIZE,
                    dtype=torch.uint8,
                    device=mqa_q.device,
                )
            impl._workspace_buffer = _fi_sparse_workspace
        if impl.bmm1_scale is None:
            impl.bmm1_scale = impl.scale
            impl.bmm1_scale *= mla._q_scale_float * mla._k_scale_float
        if impl.bmm2_scale is None:
            impl.bmm2_scale = 1.0
            impl.bmm2_scale *= mla._k_scale_float

        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

        kv_cache_fp8 = mla.kv_cache.view(torch.float8_e4m3fn)
        sparse_out = trtllm_batch_decode_with_kv_cache_mla(
            query=mqa_q.unsqueeze(1),
            kv_cache=kv_cache_fp8.unsqueeze(1),
            workspace_buffer=impl._workspace_buffer,
            qk_nope_head_dim=impl.qk_nope_head_dim,
            kv_lora_rank=impl.kv_lora_rank,
            qk_rope_head_dim=impl.qk_rope_head_dim,
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=seq_lens,
            max_seq_len=attn_metadata.topk_tokens,
            bmm1_scale=impl.bmm1_scale,
            bmm2_scale=impl.bmm2_scale,
            sparse_mla_top_k=attn_metadata.topk_tokens,
        )
        sparse_out = sparse_out.view(-1, sparse_out.shape[-2], sparse_out.shape[-1])

        # MLA v-up projection.
        x = sparse_out.view(-1, mla.num_heads, mla.kv_lora_rank).transpose(0, 1)
        out_view = attn_output_actual.view(-1, mla.num_heads, mla.v_head_dim)
        out_t = out_view.transpose(0, 1)
        torch.bmm(x, mla.W_UV, out=out_t)
        out_new = out_t.transpose(0, 1).reshape(-1, mla.num_heads * mla.v_head_dim)
        n_heads, batch, v_dim = out_t.shape
        out_t.resize_((batch, n_heads * v_dim))
        out_t.copy_(out_new)

    # MLA o projection.
    o_proj = wrapper.o_proj
    if o_proj.input_is_parallel:
        input_parallel = attn_output
    else:
        input_parallel = torch.chunk(attn_output, o_proj.tp_size, dim=-1)[
            o_proj.tp_rank
        ].contiguous()
    bias = None if (o_proj.tp_rank > 0 or o_proj.skip_bias_add) else o_proj.bias
    if hasattr(o_proj, "weight_global_scale") and hasattr(
        o_proj, "input_global_scale_inv"
    ):
        o_shape = [*input_parallel.shape[:-1], o_proj.output_size_per_partition]
        x_fp4, x_blockscale = ops.scaled_fp4_quant(
            input_parallel,
            o_proj.input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend=o_proj.quant_method.backend.value,
        )
        if o_proj.weights_padding_cols > 0:
            x_fp4 = F.pad(x_fp4, (0, o_proj.weights_padding_cols)).contiguous()
        backend_name = o_proj.quant_method.backend.value[len("flashinfer-") :]
        hidden_states = torch.ops.vllm.flashinfer_mm_fp4(
            x_fp4,
            o_proj.weight.t(),
            x_blockscale.view(torch.uint8),
            o_proj.weight_scale.view(torch.uint8).t(),
            o_proj.alpha,
            input_parallel.dtype,
            backend_name == "trtllm" and x_fp4.shape[0] <= 32,
            backend_name,
        )
        if hidden_states.shape[-1] != o_proj.output_size_per_partition:
            hidden_states = hidden_states[
                ..., : o_proj.output_size_per_partition
            ].contiguous()
        hidden_states = hidden_states.view(*o_shape)
        if bias is not None:
            hidden_states = hidden_states + bias
    else:
        hidden_states = F.linear(input_parallel, o_proj.weight, bias)
    if o_proj.reduce_results and o_proj.tp_size > 1:
        hidden_states = torch.ops.vllm.all_reduce(
            hidden_states, group_name=tp_group_name
        )

    # Post-attention RMSNorm and residual.
    norm = layer.post_attention_layernorm
    ops.fused_add_rms_norm(
        hidden_states,
        residual,
        norm.weight.data,
        norm.variance_epsilon,
    )

    # MLP or MoE.
    if isinstance(layer.mlp, DeepseekV2MoE):
        moe = layer.mlp
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Optional shared experts.
        shared_output = None
        if moe.shared_experts is not None:
            shared_mlp = moe.shared_experts
            gate_up_proj = shared_mlp.gate_up_proj
            bias = gate_up_proj.bias if not gate_up_proj.skip_bias_add else None
            if hasattr(gate_up_proj, "weight_global_scale") and hasattr(
                gate_up_proj, "input_global_scale_inv"
            ):
                gate_up_shape = [
                    *hidden_states.shape[:-1],
                    gate_up_proj.output_size_per_partition,
                ]
                x_fp4, x_blockscale = ops.scaled_fp4_quant(
                    hidden_states,
                    gate_up_proj.input_global_scale_inv,
                    is_sf_swizzled_layout=True,
                    backend=gate_up_proj.quant_method.backend.value,
                )
                if gate_up_proj.weights_padding_cols > 0:
                    x_fp4 = F.pad(
                        x_fp4, (0, gate_up_proj.weights_padding_cols)
                    ).contiguous()
                backend_name = gate_up_proj.quant_method.backend.value[
                    len("flashinfer-") :
                ]
                gate_up = torch.ops.vllm.flashinfer_mm_fp4(
                    x_fp4,
                    gate_up_proj.weight.t(),
                    x_blockscale.view(torch.uint8),
                    gate_up_proj.weight_scale.view(torch.uint8).t(),
                    gate_up_proj.alpha,
                    hidden_states.dtype,
                    backend_name == "trtllm" and x_fp4.shape[0] <= 32,
                    backend_name,
                )
                if gate_up.shape[-1] != gate_up_proj.output_size_per_partition:
                    gate_up = gate_up[
                        ..., : gate_up_proj.output_size_per_partition
                    ].contiguous()
                gate_up = gate_up.view(*gate_up_shape)
                if bias is not None:
                    gate_up = gate_up + bias
            else:
                gate_up = F.linear(hidden_states, gate_up_proj.weight, bias)
            if gate_up_proj.gather_output and gate_up_proj.tp_size > 1:
                gathered_gate_up = [
                    torch.empty_like(gate_up) for _ in range(gate_up_proj.tp_size)
                ]
                torch.distributed.all_gather(gathered_gate_up, gate_up)
                gate_up = torch.cat(gathered_gate_up, dim=-1)
            shared_output = F.silu(gate_up[..., : gate_up.shape[-1] // 2]) * gate_up[
                ..., gate_up.shape[-1] // 2 :
            ]
            down_proj = shared_mlp.down_proj
            input_parallel = shared_output
            if not down_proj.input_is_parallel:
                input_parallel = torch.chunk(
                    shared_output, down_proj.tp_size, dim=-1
                )[down_proj.tp_rank].contiguous()
            bias = (
                None
                if (down_proj.tp_rank > 0 or down_proj.skip_bias_add)
                else down_proj.bias
            )
            if hasattr(down_proj, "weight_global_scale") and hasattr(
                down_proj, "input_global_scale_inv"
            ):
                shared_shape = [
                    *input_parallel.shape[:-1],
                    down_proj.output_size_per_partition,
                ]
                x_fp4, x_blockscale = ops.scaled_fp4_quant(
                    input_parallel,
                    down_proj.input_global_scale_inv,
                    is_sf_swizzled_layout=True,
                    backend=down_proj.quant_method.backend.value,
                )
                if down_proj.weights_padding_cols > 0:
                    x_fp4 = F.pad(
                        x_fp4, (0, down_proj.weights_padding_cols)
                    ).contiguous()
                backend_name = down_proj.quant_method.backend.value[
                    len("flashinfer-") :
                ]
                shared_output = torch.ops.vllm.flashinfer_mm_fp4(
                    x_fp4,
                    down_proj.weight.t(),
                    x_blockscale.view(torch.uint8),
                    down_proj.weight_scale.view(torch.uint8).t(),
                    down_proj.alpha,
                    input_parallel.dtype,
                    backend_name == "trtllm" and x_fp4.shape[0] <= 32,
                    backend_name,
                )
                if shared_output.shape[-1] != down_proj.output_size_per_partition:
                    shared_output = shared_output[
                        ..., : down_proj.output_size_per_partition
                    ].contiguous()
                shared_output = shared_output.view(*shared_shape)
                if bias is not None:
                    shared_output = shared_output + bias
            else:
                shared_output = F.linear(input_parallel, down_proj.weight, bias)
            if down_proj.reduce_results and down_proj.tp_size > 1:
                shared_output = torch.ops.vllm.all_reduce(
                    shared_output, group_name=tp_group_name
                )

        # Router.
        gate = moe.gate
        if gate.allow_dsv3_router_gemm and hidden_states.shape[0] <= 16:
            router_logits = ops.dsv3_router_gemm(
                hidden_states=hidden_states,
                router_weight=gate.weight,
                output_dtype=gate.out_dtype,
            )
        elif gate.allow_cublas_router_gemm and hidden_states.dtype == torch.bfloat16:
            router_logits = ops.router_gemm_bf16_fp32(hidden_states, gate.weight)
        else:
            gate_input = hidden_states
            if gate.out_dtype is not None and gate_input.dtype != gate.weight.dtype:
                gate_input = gate_input.to(gate.weight.dtype)
            bias = gate.bias if not gate.skip_bias_add else None
            router_logits = F.linear(gate_input, gate.weight, bias)
            if gate.out_dtype is not None and router_logits.dtype != gate.out_dtype:
                router_logits = router_logits.to(gate.out_dtype)

        # FlashInfer TRTLLM NVFP4 monolithic MoE.
        quant_method = moe.experts.quant_method
        kernel = quant_method.moe_kernel
        assert kernel is not None and kernel.is_monolithic
        assert isinstance(kernel.fused_experts, TrtLlmNvFp4ExpertsMonolithic)
        fused_experts = kernel.fused_experts
        quant_config = fused_experts.quant_config
        assert not fused_experts.expects_unquantized_inputs
        input_sf = quant_config.a1_gscale
        assert quant_config.use_nvfp4_w4a4
        assert quant_config.quant_dtype == "nvfp4"
        assert quant_config.block_shape is None
        a1q, a1q_scale = ops.scaled_fp4_quant(
            hidden_states,
            input_sf,
            is_sf_swizzled_layout=quant_config.is_nvfp4_scale_swizzled,
        )
        assert fused_experts.routing_method_type == RoutingMethodType.DeepSeekV3
        router_logits = router_logits.to(torch.float32)
        e_score_correction_bias = moe.experts.e_score_correction_bias
        if e_score_correction_bias is not None:
            e_score_correction_bias = e_score_correction_bias.to(torch.bfloat16)
        assert a1q_scale is not None
        assert quant_config.w1_scale is not None
        assert quant_config.w2_scale is not None

        import flashinfer

        assert moe.experts.activation.value == "silu"
        activation_type = 3
        final_hidden_states = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
            routing_logits=router_logits,
            routing_bias=e_score_correction_bias,
            hidden_states=a1q,
            hidden_states_scale=a1q_scale.view(torch.float8_e4m3fn).reshape(
                *a1q.shape[:-1], -1
            ),
            gemm1_weights=moe.experts.w13_weight,
            gemm1_weights_scale=quant_config.w1_scale.view(torch.float8_e4m3fn),
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=moe.experts.w2_weight,
            gemm2_weights_scale=quant_config.w2_scale.view(torch.float8_e4m3fn),
            gemm2_bias=None,
            output1_scale_scalar=fused_experts.g1_scale_c,
            output1_scale_gate_scalar=quant_config.g1_alphas,
            output2_scale_scalar=quant_config.g2_alphas,
            num_experts=moe.experts.global_num_experts,
            top_k=fused_experts.topk,
            n_group=(moe.experts.num_expert_group or 0),
            topk_group=(moe.experts.topk_group or 0),
            intermediate_size=fused_experts.intermediate_size_per_partition,
            local_expert_offset=(
                fused_experts.ep_rank * fused_experts.local_num_experts
            ),
            local_num_experts=fused_experts.local_num_experts,
            routed_scaling_factor=moe.experts.routed_scaling_factor,
            routing_method_type=fused_experts.routing_method_type,
            do_finalize=True,
            activation_type=activation_type,
        )[0]

        final_hidden_states *= moe.routed_scaling_factor

        if shared_output is not None:
            final_hidden_states += shared_output

        if moe.tp_size > 1:
            final_hidden_states = torch.ops.vllm.all_reduce(
                final_hidden_states, group_name=tp_group_name
            )
        hidden_states = final_hidden_states.view(num_tokens, hidden_dim)
    else:
        mlp = layer.mlp
        gate_up_proj = mlp.gate_up_proj
        bias = gate_up_proj.bias if not gate_up_proj.skip_bias_add else None
        if hasattr(gate_up_proj, "weight_global_scale") and hasattr(
            gate_up_proj, "input_global_scale_inv"
        ):
            gate_up_shape = [
                *hidden_states.shape[:-1],
                gate_up_proj.output_size_per_partition,
            ]
            x_fp4, x_blockscale = ops.scaled_fp4_quant(
                hidden_states,
                gate_up_proj.input_global_scale_inv,
                is_sf_swizzled_layout=True,
                backend=gate_up_proj.quant_method.backend.value,
            )
            if gate_up_proj.weights_padding_cols > 0:
                x_fp4 = F.pad(
                    x_fp4, (0, gate_up_proj.weights_padding_cols)
                ).contiguous()
            backend_name = gate_up_proj.quant_method.backend.value[
                len("flashinfer-") :
            ]
            gate_up = torch.ops.vllm.flashinfer_mm_fp4(
                x_fp4,
                gate_up_proj.weight.t(),
                x_blockscale.view(torch.uint8),
                gate_up_proj.weight_scale.view(torch.uint8).t(),
                gate_up_proj.alpha,
                hidden_states.dtype,
                backend_name == "trtllm" and x_fp4.shape[0] <= 32,
                backend_name,
            )
            if gate_up.shape[-1] != gate_up_proj.output_size_per_partition:
                gate_up = gate_up[
                    ..., : gate_up_proj.output_size_per_partition
                ].contiguous()
            gate_up = gate_up.view(*gate_up_shape)
            if bias is not None:
                gate_up = gate_up + bias
        else:
            gate_up = F.linear(hidden_states, gate_up_proj.weight, bias)
        if gate_up_proj.gather_output and gate_up_proj.tp_size > 1:
            gathered_gate_up = [
                torch.empty_like(gate_up) for _ in range(gate_up_proj.tp_size)
            ]
            torch.distributed.all_gather(gathered_gate_up, gate_up)
            gate_up = torch.cat(gathered_gate_up, dim=-1)
        hidden_states = F.silu(gate_up[..., : gate_up.shape[-1] // 2]) * gate_up[
            ..., gate_up.shape[-1] // 2 :
        ]
        down_proj = mlp.down_proj
        input_parallel = hidden_states
        if not down_proj.input_is_parallel:
            input_parallel = torch.chunk(hidden_states, down_proj.tp_size, dim=-1)[
                down_proj.tp_rank
            ].contiguous()
        bias = (
            None
            if (down_proj.tp_rank > 0 or down_proj.skip_bias_add)
            else down_proj.bias
        )
        if hasattr(down_proj, "weight_global_scale") and hasattr(
            down_proj, "input_global_scale_inv"
        ):
            down_shape = [
                *input_parallel.shape[:-1],
                down_proj.output_size_per_partition,
            ]
            x_fp4, x_blockscale = ops.scaled_fp4_quant(
                input_parallel,
                down_proj.input_global_scale_inv,
                is_sf_swizzled_layout=True,
                backend=down_proj.quant_method.backend.value,
            )
            if down_proj.weights_padding_cols > 0:
                x_fp4 = F.pad(
                    x_fp4, (0, down_proj.weights_padding_cols)
                ).contiguous()
            backend_name = down_proj.quant_method.backend.value[len("flashinfer-") :]
            hidden_states = torch.ops.vllm.flashinfer_mm_fp4(
                x_fp4,
                down_proj.weight.t(),
                x_blockscale.view(torch.uint8),
                down_proj.weight_scale.view(torch.uint8).t(),
                down_proj.alpha,
                input_parallel.dtype,
                backend_name == "trtllm" and x_fp4.shape[0] <= 32,
                backend_name,
            )
            if hidden_states.shape[-1] != down_proj.output_size_per_partition:
                hidden_states = hidden_states[
                    ..., : down_proj.output_size_per_partition
                ].contiguous()
            hidden_states = hidden_states.view(*down_shape)
            if bias is not None:
                hidden_states = hidden_states + bias
        else:
            hidden_states = F.linear(input_parallel, down_proj.weight, bias)
        if down_proj.reduce_results and down_proj.tp_size > 1:
            hidden_states = torch.ops.vllm.all_reduce(
                hidden_states, group_name=tp_group_name
            )

    return hidden_states, residual


def flat_forward(
    model,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None = None,
):
    # Embedding.
    if inputs_embeds is not None:
        hidden_states = inputs_embeds
    else:
        if input_ids is None:
            raise ValueError(
                "Either input_ids or inputs_embeds must be provided "
                "to FlatDeepseekV32Model.forward"
            )
        hidden_states = model.embed_tokens(input_ids)
    residual = None

    # Decoder stack.
    forward_context = get_forward_context()
    tp_group_name = get_tp_group().unique_name
    aux_hidden_states = []
    for idx, layer in enumerate(
        islice(model.layers, model.start_layer, model.end_layer),
        start=model.start_layer,
    ):
        if idx in model.aux_hidden_state_layers:
            aux_hidden_states.append(hidden_states + residual)
        hidden_states, residual = transformer_layer(
            layer,
            positions,
            hidden_states,
            residual,
            forward_context,
            tp_group_name,
        )

    # Final RMSNorm.
    norm = model.norm
    ops.fused_add_rms_norm(
        hidden_states,
        residual,
        norm.weight.data,
        norm.variance_epsilon,
    )
    del residual

    if len(aux_hidden_states) > 0:
        return hidden_states, aux_hidden_states
    return hidden_states
