# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flat forward path for DeepSeek V4.

The original module tree is still constructed so the existing vLLM loader can
populate all weights.  Forward avoids decoder-layer/module dispatch and instead
walks explicit tensors and selected backend ops in a single Python path.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Any

import torch
import torch.nn.functional as F

from vllm.distributed import (
    get_pp_group,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    get_masked_input_and_mask,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _per_token_group_quant_fp8,
    get_fp8_min_max,
    w8a8_triton_block_scaled_mm,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.sparse_attn_indexer import kv_cache_as_quant_view
from vllm.models.deepseek_v4.attention import PREFILL_CHUNK_SIZE
from vllm.models.deepseek_v4.compressor import _save_partial_states_kernel
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    fused_indexer_q_rope_quant,
    quantize_and_insert_k_cache,
)
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    _fused_inv_rope_fp8_quant_per_head,
)
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4Model,
    _deepseek_v4_stage_mega_moe_inputs_kernel,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.triton_utils import triton
from vllm.utils.deep_gemm import (
    fp8_einsum,
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    get_tma_aligned_size,
)
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager


@dataclass(slots=True)
class FlatDeepseekV4LayerParams:
    layer: Any
    mla: Any
    ffn: Any
    shared_experts: Any | None
    attn_norm_weight: torch.Tensor
    hc_attn_fn: torch.Tensor
    hc_attn_scale: torch.Tensor
    hc_attn_base: torch.Tensor
    hc_ffn_fn: torch.Tensor
    hc_ffn_scale: torch.Tensor
    hc_ffn_base: torch.Tensor
    moe_max_num_tokens: int
    moe_symm_buffer: Any | None
    moe_l1_weights: tuple[torch.Tensor, torch.Tensor] | None
    moe_l2_weights: tuple[torch.Tensor, torch.Tensor] | None


@dataclass(slots=True)
class FlatDeepseekV4Params:
    layers: list[FlatDeepseekV4LayerParams]
    embed_tokens: Any
    norm_weight: torch.Tensor | None
    hc_head_fn: torch.Tensor
    hc_head_scale: torch.Tensor
    hc_head_base: torch.Tensor


def extract_all_layer_params(model: "FlatDeepseekV4Model") -> FlatDeepseekV4Params:
    """Collect the tensors/modules needed by the flat path.

    Module objects are retained only as parameter containers for quant methods
    and backend static-context lookups.  Their ``forward`` methods are not used
    by the flat decoder path.
    """

    layers: list[FlatDeepseekV4LayerParams] = []
    for layer in islice(model.layers, model.start_layer, model.end_layer):
        ffn = layer.ffn
        experts = ffn.experts
        if model.use_mega_moe:
            experts.finalize_weights()
            assert experts._transformed_l1_weights is not None
            assert experts._transformed_l2_weights is not None
            moe_symm_buffer = experts.get_symm_buffer()
            moe_l1_weights = experts._transformed_l1_weights
            moe_l2_weights = experts._transformed_l2_weights
            moe_max_num_tokens = experts.max_num_tokens
        else:
            moe_symm_buffer = None
            moe_l1_weights = None
            moe_l2_weights = None
            moe_max_num_tokens = 0
        layers.append(
            FlatDeepseekV4LayerParams(
                layer=layer,
                mla=layer.attn.mla_attn,
                ffn=ffn,
                shared_experts=ffn.shared_experts,
                attn_norm_weight=layer.attn_norm.weight,
                hc_attn_fn=layer.hc_attn_fn,
                hc_attn_scale=layer.hc_attn_scale,
                hc_attn_base=layer.hc_attn_base,
                hc_ffn_fn=layer.hc_ffn_fn,
                hc_ffn_scale=layer.hc_ffn_scale,
                hc_ffn_base=layer.hc_ffn_base,
                moe_max_num_tokens=moe_max_num_tokens,
                moe_symm_buffer=moe_symm_buffer,
                moe_l1_weights=moe_l1_weights,
                moe_l2_weights=moe_l2_weights,
            )
        )

    return FlatDeepseekV4Params(
        layers=layers,
        embed_tokens=model.embed_tokens if get_pp_group().is_first_rank else None,
        norm_weight=model.norm.weight if get_pp_group().is_last_rank else None,
        hc_head_fn=model.hc_head_fn,
        hc_head_scale=model.hc_head_scale,
        hc_head_base=model.hc_head_base,
    )

def transformer_layer(
    layer_params: FlatDeepseekV4LayerParams,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    input_ids: torch.Tensor | None,
    residual: torch.Tensor | None,
    post_mix: torch.Tensor | None,
    res_mix: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One flat DeepSeek V4 decoder layer."""

    layer = layer_params.layer
    if residual is None:
        # HC attention pre-mix.
        residual = hidden_states
        hc_mult = hidden_states.shape[-2]
        hidden_size = hidden_states.shape[-1]
        residual_flat = hidden_states.view(-1, hc_mult, hidden_size)
        outer_shape = hidden_states.shape[:-2]
        num_tokens = residual_flat.shape[0]
        hc_x = residual_flat.view(num_tokens, hc_mult * hidden_size).to(torch.float32)
        mixes = torch.matmul(hc_x, layer_params.hc_attn_fn.t())
        sqrsum = hc_x.square().sum(dim=-1, keepdim=True)
        mixes = mixes * torch.rsqrt(
            sqrsum / (hc_mult * hidden_size) + layer.rms_norm_eps
        )
        pre_mix = (
            torch.sigmoid(
                mixes[:, :hc_mult] * layer_params.hc_attn_scale[0]
                + layer_params.hc_attn_base[:hc_mult]
            )
            + layer.hc_eps
        )
        post_mix = (
            torch.sigmoid(
                mixes[:, hc_mult : 2 * hc_mult] * layer_params.hc_attn_scale[1]
                + layer_params.hc_attn_base[hc_mult : 2 * hc_mult]
            )
            * layer.hc_post_alpha
        )
        comb_logits = mixes[:, 2 * hc_mult :].view(num_tokens, hc_mult, hc_mult)
        comb_logits = comb_logits * layer_params.hc_attn_scale[
            2
        ] + layer_params.hc_attn_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
        res_mix = torch.softmax(comb_logits, dim=-1) + layer.hc_eps
        res_mix = res_mix / (res_mix.sum(dim=-2, keepdim=True) + layer.hc_eps)
        for _ in range(layer.hc_sinkhorn_iters - 1):
            res_mix = res_mix / (res_mix.sum(dim=-1, keepdim=True) + layer.hc_eps)
            res_mix = res_mix / (res_mix.sum(dim=-2, keepdim=True) + layer.hc_eps)
        hidden_states = torch.sum(
            pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1
        ).to(torch.bfloat16)
        hidden_states = hidden_states.view(*outer_shape, hidden_size)
        post_mix = post_mix.view(*outer_shape, hc_mult, 1)
        res_mix = res_mix.view(*outer_shape, hc_mult, hc_mult)
    else:
        # HC previous post-mix + attention pre-mix.
        assert post_mix is not None
        assert res_mix is not None
        residual = (
            torch.einsum(
                "...ij,...ih->...jh",
                res_mix.to(torch.float32),
                residual.to(torch.float32),
            )
            + post_mix.to(torch.float32) * hidden_states.unsqueeze(-2).to(torch.float32)
        ).to(residual.dtype)
        hc_mult = residual.shape[-2]
        hidden_size = residual.shape[-1]
        residual_flat = residual.view(-1, hc_mult, hidden_size)
        outer_shape = residual.shape[:-2]
        num_tokens = residual_flat.shape[0]
        hc_x = residual_flat.view(num_tokens, hc_mult * hidden_size).to(torch.float32)
        mixes = torch.matmul(hc_x, layer_params.hc_attn_fn.t())
        sqrsum = hc_x.square().sum(dim=-1, keepdim=True)
        mixes = mixes * torch.rsqrt(
            sqrsum / (hc_mult * hidden_size) + layer.rms_norm_eps
        )
        pre_mix = (
            torch.sigmoid(
                mixes[:, :hc_mult] * layer_params.hc_attn_scale[0]
                + layer_params.hc_attn_base[:hc_mult]
            )
            + layer.hc_eps
        )
        post_mix = (
            torch.sigmoid(
                mixes[:, hc_mult : 2 * hc_mult] * layer_params.hc_attn_scale[1]
                + layer_params.hc_attn_base[hc_mult : 2 * hc_mult]
            )
            * layer.hc_post_alpha
        )
        comb_logits = mixes[:, 2 * hc_mult :].view(num_tokens, hc_mult, hc_mult)
        comb_logits = comb_logits * layer_params.hc_attn_scale[
            2
        ] + layer_params.hc_attn_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
        res_mix = torch.softmax(comb_logits, dim=-1) + layer.hc_eps
        res_mix = res_mix / (res_mix.sum(dim=-2, keepdim=True) + layer.hc_eps)
        for _ in range(layer.hc_sinkhorn_iters - 1):
            res_mix = res_mix / (res_mix.sum(dim=-1, keepdim=True) + layer.hc_eps)
            res_mix = res_mix / (res_mix.sum(dim=-2, keepdim=True) + layer.hc_eps)
        hidden_states = torch.sum(
            pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1
        ).to(torch.bfloat16)
        hidden_states = hidden_states.view(*outer_shape, hidden_size)
        post_mix = post_mix.view(*outer_shape, hc_mult, 1)
        res_mix = res_mix.view(*outer_shape, hc_mult, hc_mult)

    # Attention RMSNorm and MLA.
    variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden_states = (
        hidden_states
        * torch.rsqrt(variance + layer.rms_norm_eps)
        * layer_params.attn_norm_weight
    ).to(hidden_states.dtype)
    mla = layer_params.mla
    num_tokens = hidden_states.shape[0]
    o_padded = torch.empty(
        (num_tokens, mla.padded_heads, mla.head_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    fused_wqa_wkv = mla.fused_wqa_wkv
    fused_wqa_wkv_input = hidden_states.contiguous().view(
        -1, hidden_states.shape[-1]
    )
    if fused_wqa_wkv.weight.dtype == torch.float8_e4m3fn:
        fused_wqa_wkv_q = torch.empty_like(
            fused_wqa_wkv_input, dtype=torch.float8_e4m3fn
        )
        fused_wqa_wkv_scale = torch.empty(
            fused_wqa_wkv_input.shape[:-1]
            + (fused_wqa_wkv_input.shape[-1] // 128,),
            dtype=torch.float32,
            device=fused_wqa_wkv_input.device,
        )
        fp8_min, fp8_max = get_fp8_min_max()
        _per_token_group_quant_fp8[(fused_wqa_wkv_input.numel() // 128,)](
            fused_wqa_wkv_input,
            fused_wqa_wkv_q,
            fused_wqa_wkv_scale,
            128,
            fused_wqa_wkv_input.shape[1],
            fused_wqa_wkv_input.stride(0),
            1e-10,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            use_ue8m0=True,
            BLOCK=128,
            num_warps=1,
            num_stages=1,
        )
        fused_wqa_wkv_weight_scale = getattr(
            fused_wqa_wkv, "weight_scale_inv", None
        )
        if fused_wqa_wkv_weight_scale is None:
            fused_wqa_wkv_weight_scale = fused_wqa_wkv.weight_scale
        qr_kv = w8a8_triton_block_scaled_mm(
            fused_wqa_wkv_q,
            fused_wqa_wkv.weight,
            fused_wqa_wkv_scale,
            fused_wqa_wkv_weight_scale,
            [128, 128],
            output_dtype=torch.bfloat16,
        ).view(*hidden_states.shape[:-1], fused_wqa_wkv.weight.shape[0])
    else:
        fused_wqa_wkv_bias = (
            None
            if (fused_wqa_wkv.tp_rank > 0 or fused_wqa_wkv.skip_bias_add)
            else fused_wqa_wkv.bias
        )
        qr_kv = F.linear(
            hidden_states,
            fused_wqa_wkv.weight,
            fused_wqa_wkv_bias,
        )
    if fused_wqa_wkv.gather_output and fused_wqa_wkv.tp_size > 1:
        qr_kv = tensor_model_parallel_all_gather(qr_kv)

    kv_score = None
    indexer_kv_score = None
    indexer_weights = None
    if mla.compressor is not None:
        kv_score = torch.mm(
            hidden_states,
            mla.compressor.fused_wkv_wgate.weight.T,
            out_dtype=torch.float32,
        )
    if mla.indexer is not None:
        indexer = mla.indexer
        indexer_weights = F.linear(
            hidden_states,
            indexer.weights_proj.weight,
            indexer.weights_proj.bias,
        )
        indexer_kv_score = torch.mm(
            hidden_states,
            indexer.compressor.fused_wkv_wgate.weight.T,
            out_dtype=torch.float32,
        )

    qr, kv = qr_kv.split([mla.q_lora_rank, mla.head_dim], dim=-1)
    q_variance = qr.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    qr = (qr * torch.rsqrt(q_variance + mla.eps) * mla.q_norm.weight.data).to(
        qr.dtype
    )
    kv_variance = kv.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    kv = (kv * torch.rsqrt(kv_variance + mla.eps) * mla.kv_norm.weight.data).to(
        kv.dtype
    )

    wq_b = mla.wq_b
    wq_b_input = qr.contiguous().view(-1, qr.shape[-1])
    if wq_b.weight.dtype == torch.float8_e4m3fn:
        wq_b_q = torch.empty_like(wq_b_input, dtype=torch.float8_e4m3fn)
        wq_b_scale = torch.empty(
            wq_b_input.shape[:-1] + (wq_b_input.shape[-1] // 128,),
            dtype=torch.float32,
            device=wq_b_input.device,
        )
        fp8_min, fp8_max = get_fp8_min_max()
        _per_token_group_quant_fp8[(wq_b_input.numel() // 128,)](
            wq_b_input,
            wq_b_q,
            wq_b_scale,
            128,
            wq_b_input.shape[1],
            wq_b_input.stride(0),
            1e-10,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            use_ue8m0=True,
            BLOCK=128,
            num_warps=1,
            num_stages=1,
        )
        wq_b_weight_scale = getattr(wq_b, "weight_scale_inv", None)
        if wq_b_weight_scale is None:
            wq_b_weight_scale = wq_b.weight_scale
        q = w8a8_triton_block_scaled_mm(
            wq_b_q,
            wq_b.weight,
            wq_b_scale,
            wq_b_weight_scale,
            [128, 128],
            output_dtype=torch.bfloat16,
        ).view(*qr.shape[:-1], wq_b.weight.shape[0])
    else:
        wq_b_bias = None if (wq_b.tp_rank > 0 or wq_b.skip_bias_add) else wq_b.bias
        q = F.linear(qr, wq_b.weight, wq_b_bias)
    if wq_b.gather_output and wq_b.tp_size > 1:
        q = tensor_model_parallel_all_gather(q)
    q = q.view(-1, mla.n_local_heads, mla.head_dim)

    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        q_head_variance = q.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        q = (q * torch.rsqrt(q_head_variance + mla.eps)).to(q.dtype)

        cos_sin = mla.rotary_emb.cos_sin_cache[positions.to(torch.int64)]
        half_rope = mla.rope_head_dim // 2
        cos = cos_sin[:, :half_rope].view(num_tokens, 1, half_rope)
        sin = cos_sin[:, half_rope:].view(num_tokens, 1, half_rope)
        q_nope = q[..., : mla.nope_head_dim]
        q_rope = q[..., mla.nope_head_dim :]
        q_even = q_rope[..., 0::2].to(torch.float32)
        q_odd = q_rope[..., 1::2].to(torch.float32)
        q_rope_rot = torch.empty_like(q_rope)
        q_rope_rot[..., 0::2] = (q_even * cos - q_odd * sin).to(q.dtype)
        q_rope_rot[..., 1::2] = (q_even * sin + q_odd * cos).to(q.dtype)
        q = torch.cat((q_nope, q_rope_rot), dim=-1)

        kv_nope = kv[..., : mla.nope_head_dim]
        kv_rope = kv[..., mla.nope_head_dim :]
        kv_cos = cos_sin[:, :half_rope]
        kv_sin = cos_sin[:, half_rope:]
        kv_even = kv_rope[..., 0::2].to(torch.float32)
        kv_odd = kv_rope[..., 1::2].to(torch.float32)
        kv_rope_rot = torch.empty_like(kv_rope)
        kv_rope_rot[..., 0::2] = (kv_even * kv_cos - kv_odd * kv_sin).to(kv.dtype)
        kv_rope_rot[..., 1::2] = (kv_even * kv_sin + kv_odd * kv_cos).to(kv.dtype)
        kv_rot = torch.cat((kv_nope, kv_rope_rot), dim=-1)

        swa_metadata = attn_metadata.get(mla.swa_cache_layer.prefix)
        assert swa_metadata is not None
        swa_kv_cache = mla.swa_cache_layer.kv_cache
        quantize_and_insert_k_cache(
            kv_rot,
            swa_kv_cache.view(swa_kv_cache.shape[0], -1),
            swa_metadata.slot_mapping,
            swa_metadata.block_size,
        )

        if mla.indexer is not None:
            assert mla.compressor is not None
            assert kv_score is not None
            assert indexer_kv_score is not None
            assert indexer_weights is not None
            compressor = mla.compressor
            kv_comp, score_comp = kv_score.split(
                [
                    compressor.coff * compressor.head_dim,
                    compressor.coff * compressor.head_dim,
                ],
                dim=-1,
            )
            state_metadata = attn_metadata[compressor.state_cache.prefix]
            token_to_req_indices = state_metadata.token_to_req_indices
            slot_mapping = state_metadata.slot_mapping
            num_actual = slot_mapping.shape[0]
            block_table = state_metadata.block_table
            block_size = state_metadata.block_size
            state_cache = compressor.state_cache.kv_cache
            state_width = state_cache.shape[-1] // 2
            pdl_kwargs = {} if current_platform.is_rocm() else {"launch_pdl": False}
            _save_partial_states_kernel[(num_actual,)](
                kv_comp,
                kv_comp.stride(0),
                score_comp,
                score_comp.stride(0),
                compressor.ape,
                compressor.ape.stride(0),
                positions,
                state_cache,
                state_cache.stride(0),
                state_cache.stride(1),
                slot_mapping,
                block_size,
                HEAD_SIZE=kv_comp.shape[-1],
                TRITON_BLOCK_SIZE=triton.next_power_of_2(kv_comp.shape[-1]),
                STATE_WIDTH=state_width,
                COMPRESS_RATIO=compressor.compress_ratio,
                **pdl_kwargs,
            )
            k_cache_metadata = attn_metadata[compressor.k_cache_prefix]
            kv_cache = compressor._static_forward_context[
                compressor.k_cache_prefix
            ].kv_cache
            compressor._fused_kernel[(num_actual,)](
                state_cache,
                state_cache.stride(0),
                state_cache.stride(1),
                token_to_req_indices,
                positions,
                slot_mapping,
                block_table,
                block_table.stride(0),
                block_size,
                compressor.norm.weight,
                compressor.rms_norm_eps,
                mla.rotary_emb.cos_sin_cache,
                mla.rotary_emb.cos_sin_cache.stride(0),
                kv_cache,
                k_cache_metadata.slot_mapping,
                kv_cache.shape[1],
                HEAD_SIZE=compressor.head_dim,
                TRITON_BLOCK_SIZE=triton.next_power_of_2(compressor.head_dim),
                STATE_WIDTH=state_width,
                COMPRESS_RATIO=compressor.compress_ratio,
                OVERLAP=compressor.overlap,
                ROPE_HEAD_DIM=compressor.rope_head_dim,
                FP8_MAX=448.0,
                QUANT_BLOCK=compressor._quant_block,
                TOKEN_STRIDE=compressor._token_stride,
                SCALE_DIM=compressor._scale_dim,
                KV_BLOCK_STRIDE=kv_cache.stride(0),
                num_warps=compressor._num_warps,
                **pdl_kwargs,
            )

            indexer = mla.indexer
            indexer_wq_b = indexer.wq_b
            indexer_wq_b_input = qr.contiguous().view(-1, qr.shape[-1])
            if indexer_wq_b.weight.dtype == torch.float8_e4m3fn:
                indexer_q_in = torch.empty_like(
                    indexer_wq_b_input, dtype=torch.float8_e4m3fn
                )
                indexer_q_scale = torch.empty(
                    indexer_wq_b_input.shape[:-1]
                    + (indexer_wq_b_input.shape[-1] // 128,),
                    dtype=torch.float32,
                    device=indexer_wq_b_input.device,
                )
                fp8_min, fp8_max = get_fp8_min_max()
                _per_token_group_quant_fp8[(indexer_wq_b_input.numel() // 128,)](
                    indexer_wq_b_input,
                    indexer_q_in,
                    indexer_q_scale,
                    128,
                    indexer_wq_b_input.shape[1],
                    indexer_wq_b_input.stride(0),
                    1e-10,
                    fp8_min=fp8_min,
                    fp8_max=fp8_max,
                    use_ue8m0=True,
                    BLOCK=128,
                    num_warps=1,
                    num_stages=1,
                )
                indexer_wq_b_weight_scale = getattr(
                    indexer_wq_b, "weight_scale_inv", None
                )
                if indexer_wq_b_weight_scale is None:
                    indexer_wq_b_weight_scale = indexer_wq_b.weight_scale
                indexer_q = w8a8_triton_block_scaled_mm(
                    indexer_q_in,
                    indexer_wq_b.weight,
                    indexer_q_scale,
                    indexer_wq_b_weight_scale,
                    [128, 128],
                    output_dtype=torch.bfloat16,
                ).view(*qr.shape[:-1], indexer_wq_b.weight.shape[0])
            else:
                indexer_wq_b_bias = (
                    None
                    if (indexer_wq_b.tp_rank > 0 or indexer_wq_b.skip_bias_add)
                    else indexer_wq_b.bias
                )
                indexer_q = F.linear(qr, indexer_wq_b.weight, indexer_wq_b_bias)
            if indexer_wq_b.gather_output and indexer_wq_b.tp_size > 1:
                indexer_q = tensor_model_parallel_all_gather(indexer_q)
            indexer_q = indexer_q.view(-1, indexer.n_head, indexer.head_dim)

            indexer_compressor = indexer.compressor
            indexer_kv_comp, indexer_score_comp = indexer_kv_score.split(
                [
                    indexer_compressor.coff * indexer_compressor.head_dim,
                    indexer_compressor.coff * indexer_compressor.head_dim,
                ],
                dim=-1,
            )
            indexer_state_metadata = attn_metadata[
                indexer_compressor.state_cache.prefix
            ]
            indexer_token_to_req_indices = (
                indexer_state_metadata.token_to_req_indices
            )
            indexer_slot_mapping = indexer_state_metadata.slot_mapping
            indexer_num_actual = indexer_slot_mapping.shape[0]
            indexer_block_table = indexer_state_metadata.block_table
            indexer_block_size = indexer_state_metadata.block_size
            indexer_state_cache = indexer_compressor.state_cache.kv_cache
            indexer_state_width = indexer_state_cache.shape[-1] // 2
            _save_partial_states_kernel[(indexer_num_actual,)](
                indexer_kv_comp,
                indexer_kv_comp.stride(0),
                indexer_score_comp,
                indexer_score_comp.stride(0),
                indexer_compressor.ape,
                indexer_compressor.ape.stride(0),
                positions,
                indexer_state_cache,
                indexer_state_cache.stride(0),
                indexer_state_cache.stride(1),
                indexer_slot_mapping,
                indexer_block_size,
                HEAD_SIZE=indexer_kv_comp.shape[-1],
                TRITON_BLOCK_SIZE=triton.next_power_of_2(indexer_kv_comp.shape[-1]),
                STATE_WIDTH=indexer_state_width,
                COMPRESS_RATIO=indexer_compressor.compress_ratio,
                **pdl_kwargs,
            )
            indexer_k_cache_metadata = attn_metadata[
                indexer_compressor.k_cache_prefix
            ]
            indexer_kv_cache = indexer_compressor._static_forward_context[
                indexer_compressor.k_cache_prefix
            ].kv_cache
            indexer_compressor._fused_kernel[(indexer_num_actual,)](
                indexer_state_cache,
                indexer_state_cache.stride(0),
                indexer_state_cache.stride(1),
                indexer_token_to_req_indices,
                positions,
                indexer_slot_mapping,
                indexer_block_table,
                indexer_block_table.stride(0),
                indexer_block_size,
                indexer_compressor.norm.weight,
                indexer_compressor.rms_norm_eps,
                mla.indexer_rotary_emb.cos_sin_cache,
                mla.indexer_rotary_emb.cos_sin_cache.stride(0),
                indexer_kv_cache,
                indexer_k_cache_metadata.slot_mapping,
                indexer_kv_cache.shape[1],
                HEAD_SIZE=indexer_compressor.head_dim,
                TRITON_BLOCK_SIZE=triton.next_power_of_2(indexer_compressor.head_dim),
                STATE_WIDTH=indexer_state_width,
                COMPRESS_RATIO=indexer_compressor.compress_ratio,
                OVERLAP=indexer_compressor.overlap,
                ROPE_HEAD_DIM=indexer_compressor.rope_head_dim,
                FP8_MAX=448.0,
                QUANT_BLOCK=indexer_compressor._quant_block,
                TOKEN_STRIDE=indexer_compressor._token_stride,
                SCALE_DIM=indexer_compressor._scale_dim,
                KV_BLOCK_STRIDE=indexer_kv_cache.stride(0),
                num_warps=indexer_compressor._num_warps,
                **pdl_kwargs,
            )
            indexer_q_quant, folded_indexer_weights = fused_indexer_q_rope_quant(
                positions,
                indexer_q,
                mla.indexer_rotary_emb.cos_sin_cache,
                indexer_weights,
                indexer.softmax_scale,
                indexer.n_head**-0.5,
                use_fp4=indexer.use_fp4_kv,
            )
            if isinstance(indexer_q_quant, tuple):
                indexer_q_values, indexer_q_scale = indexer_q_quant
            else:
                indexer_q_values = indexer_q_quant
                indexer_q_scale = None
            indexer_metadata = attn_metadata[indexer.k_cache.prefix]
            indexer_slot_mapping = indexer_metadata.slot_mapping
            indexer_has_decode = indexer_metadata.num_decodes > 0
            indexer_has_prefill = indexer_metadata.num_prefills > 0
            indexer_num_decode_tokens = indexer_metadata.num_decode_tokens
            indexer.topk_indices_buffer[: hidden_states.shape[0]] = -1
            if indexer_has_prefill:
                prefill_metadata = indexer_metadata.prefill
                assert prefill_metadata is not None
                workspace_manager = current_workspace_manager()
                if indexer.use_fp4_kv:
                    value_spec = (
                        (
                            indexer.max_total_seq_len,
                            indexer.head_dim // 2,
                        ),
                        torch.uint8,
                    )
                    scale_spec = (
                        (
                            indexer.max_total_seq_len,
                            indexer.head_dim // 32,
                        ),
                        torch.uint8,
                    )
                else:
                    value_spec = (
                        (indexer.max_total_seq_len, indexer.head_dim),
                        current_platform.fp8_dtype(),
                    )
                    scale_spec = ((indexer.max_total_seq_len, 4), torch.uint8)
                k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
                    value_spec,
                    scale_spec,
                )
                for chunk in prefill_metadata.chunks:
                    k_quant = k_quant_full[: chunk.total_seq_lens]
                    k_scale = k_scale_full[: chunk.total_seq_lens]
                    if not chunk.skip_kv_gather:
                        cache = indexer.k_cache.kv_cache
                        cache_block_size = cache.shape[1]
                        value_width = k_quant.shape[1]
                        scale_width = k_scale.shape[1]
                        dst = 0
                        for req_idx in range(chunk.block_table.shape[0]):
                            seq_start = int(chunk.cu_seq_lens[req_idx].item())
                            seq_end = int(chunk.cu_seq_lens[req_idx + 1].item())
                            for token_idx in range(seq_end - seq_start):
                                block_idx = int(
                                    chunk.block_table[
                                        req_idx, token_idx // cache_block_size
                                    ].item()
                                )
                                block_offset = token_idx % cache_block_size
                                cache_token = cache[block_idx, block_offset]
                                k_quant[dst].copy_(cache_token[:value_width])
                                k_scale[dst].copy_(
                                    cache_token[
                                        value_width : value_width + scale_width
                                    ]
                                )
                                dst += 1
                    q_slice = indexer_q_values[chunk.token_start : chunk.token_end]
                    q_scale_slice = (
                        indexer_q_scale[chunk.token_start : chunk.token_end]
                        if indexer_q_scale is not None
                        else None
                    )
                    if indexer.use_fp4_kv:
                        q_slice_cast = q_slice.view(torch.int8)
                        k_quant_cast = k_quant.view(torch.int8)
                        k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
                    else:
                        q_slice_cast = q_slice
                        k_quant_cast = k_quant
                        k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        folded_indexer_weights[
                            chunk.token_start : chunk.token_end
                        ],
                        chunk.cu_seqlen_ks,
                        chunk.cu_seqlen_ke,
                        clean_logits=False,
                    )
                    topk_indices = indexer.topk_indices_buffer[
                        chunk.token_start : chunk.token_end,
                        : indexer.topk_tokens,
                    ]
                    for row_idx in range(logits.shape[0]):
                        row_end = int(chunk.cu_seqlen_ke[row_idx].item())
                        row_k = min(indexer.topk_tokens, row_end)
                        topk_indices[row_idx, :row_k] = torch.topk(
                            logits[row_idx, :row_end],
                            row_k,
                            dim=-1,
                        )[1].to(topk_indices.dtype)
            if indexer_has_decode:
                decode_metadata = indexer_metadata.decode
                assert decode_metadata is not None
                indexer_kv_cache_quant = kv_cache_as_quant_view(
                    indexer.k_cache.kv_cache,
                    indexer.head_dim,
                    indexer.use_fp4_kv,
                )
                decode_lens = decode_metadata.decode_lens
                if decode_metadata.requires_padding:
                    if indexer_q_scale is not None:
                        padded_q_values = pack_seq_triton(
                            indexer_q_values[:indexer_num_decode_tokens],
                            decode_lens,
                            pad_value=0,
                        )
                        padded_q_scale = pack_seq_triton(
                            indexer_q_scale[:indexer_num_decode_tokens],
                            decode_lens,
                            pad_value=0,
                        )
                    else:
                        padded_q_values = pack_seq_triton(
                            indexer_q_values[:indexer_num_decode_tokens],
                            decode_lens,
                        )
                        padded_q_scale = None
                else:
                    padded_q_values = indexer_q_values[
                        :indexer_num_decode_tokens
                    ].reshape(
                        decode_lens.shape[0],
                        -1,
                        *indexer_q_values.shape[1:],
                    )
                    if indexer_q_scale is not None:
                        padded_q_scale = indexer_q_scale[
                            :indexer_num_decode_tokens
                        ].reshape(
                            decode_lens.shape[0],
                            -1,
                            *indexer_q_scale.shape[1:],
                        )
                    else:
                        padded_q_scale = None
                batch_size = padded_q_values.shape[0]
                next_n = padded_q_values.shape[1]
                num_padded_tokens = batch_size * next_n
                seq_lens = decode_metadata.seq_lens[:batch_size]
                padded_q_cast = (
                    padded_q_values.view(torch.int8)
                    if indexer.use_fp4_kv
                    else padded_q_values
                )
                logits = fp8_fp4_paged_mqa_logits(
                    (padded_q_cast, padded_q_scale),
                    indexer_kv_cache_quant,
                    folded_indexer_weights[:num_padded_tokens],
                    seq_lens,
                    decode_metadata.block_table,
                    decode_metadata.schedule_metadata,
                    max_model_len=indexer.max_model_len,
                    clean_logits=False,
                )
                topk_indices = indexer.topk_indices_buffer[
                    :num_padded_tokens, : indexer.topk_tokens
                ]
                row_ids = torch.arange(logits.shape[0], device=logits.device)
                row_batch = row_ids // next_n
                row_offset = row_ids % next_n
                row_ends = seq_lens[row_batch] - next_n + row_offset + 1
                for row_idx in range(logits.shape[0]):
                    row_end = int(row_ends[row_idx].item())
                    row_k = min(indexer.topk_tokens, row_end)
                    topk_indices[row_idx, :row_k] = torch.topk(
                        logits[row_idx, :row_end],
                        row_k,
                        dim=-1,
                    )[1].to(topk_indices.dtype)
                if decode_metadata.requires_padding:
                    topk_indices = unpack_seq_triton(
                        topk_indices.reshape(
                            batch_size,
                            -1,
                            topk_indices.shape[-1],
                        ),
                        decode_lens,
                    )
                    indexer.topk_indices_buffer[
                        : topk_indices.shape[0], : topk_indices.shape[-1]
                    ] = topk_indices
        elif mla.compressor is not None:
            assert kv_score is not None
            compressor = mla.compressor
            kv_comp, score_comp = kv_score.split(
                [
                    compressor.coff * compressor.head_dim,
                    compressor.coff * compressor.head_dim,
                ],
                dim=-1,
            )
            state_metadata = attn_metadata[compressor.state_cache.prefix]
            token_to_req_indices = state_metadata.token_to_req_indices
            slot_mapping = state_metadata.slot_mapping
            num_actual = slot_mapping.shape[0]
            block_table = state_metadata.block_table
            block_size = state_metadata.block_size
            state_cache = compressor.state_cache.kv_cache
            state_width = state_cache.shape[-1] // 2
            pdl_kwargs = {} if current_platform.is_rocm() else {"launch_pdl": False}
            _save_partial_states_kernel[(num_actual,)](
                kv_comp,
                kv_comp.stride(0),
                score_comp,
                score_comp.stride(0),
                compressor.ape,
                compressor.ape.stride(0),
                positions,
                state_cache,
                state_cache.stride(0),
                state_cache.stride(1),
                slot_mapping,
                block_size,
                HEAD_SIZE=kv_comp.shape[-1],
                TRITON_BLOCK_SIZE=triton.next_power_of_2(kv_comp.shape[-1]),
                STATE_WIDTH=state_width,
                COMPRESS_RATIO=compressor.compress_ratio,
                **pdl_kwargs,
            )
            k_cache_metadata = attn_metadata[compressor.k_cache_prefix]
            kv_cache = compressor._static_forward_context[
                compressor.k_cache_prefix
            ].kv_cache
            compressor._fused_kernel[(num_actual,)](
                state_cache,
                state_cache.stride(0),
                state_cache.stride(1),
                token_to_req_indices,
                positions,
                slot_mapping,
                block_table,
                block_table.stride(0),
                block_size,
                compressor.norm.weight,
                compressor.rms_norm_eps,
                mla.rotary_emb.cos_sin_cache,
                mla.rotary_emb.cos_sin_cache.stride(0),
                kv_cache,
                k_cache_metadata.slot_mapping,
                kv_cache.shape[1],
                HEAD_SIZE=compressor.head_dim,
                TRITON_BLOCK_SIZE=triton.next_power_of_2(compressor.head_dim),
                STATE_WIDTH=state_width,
                COMPRESS_RATIO=compressor.compress_ratio,
                OVERLAP=compressor.overlap,
                ROPE_HEAD_DIM=compressor.rope_head_dim,
                FP8_MAX=448.0,
                QUANT_BLOCK=compressor._quant_block,
                TOKEN_STRIDE=compressor._token_stride,
                SCALE_DIM=compressor._scale_dim,
                KV_BLOCK_STRIDE=kv_cache.stride(0),
                num_warps=compressor._num_warps,
                **pdl_kwargs,
            )
    else:
        sub = mla.mla_attn
        swa_only = sub.compress_ratio <= 1
        n_compressed = (
            0
            if swa_only
            else (sub.max_model_len + sub.compress_ratio - 1) // sub.compress_ratio
        )
        workspace_width = n_compressed + sub.window_size + sub.max_num_batched_tokens
        current_workspace_manager().get_simultaneous(
            ((PREFILL_CHUNK_SIZE, workspace_width, q.shape[-1]), torch.bfloat16),
        )
        o_padded.zero_()
        attn_o = o_padded[:, : mla.n_local_heads, :]
        heads_per_group = mla.n_local_heads // mla.n_local_groups
        quant_group_size = 128
        chunks_per_head = mla.head_dim // quant_group_size
        num_scale_blocks = heads_per_group * mla.head_dim // quant_group_size
        tma_aligned_t = get_tma_aligned_size(num_tokens, 4)
        scale_inner = (
            (num_scale_blocks + 3) // 4
            if mla._tma_aligned_scales
            else num_scale_blocks
        )
        o_fp8_buf = torch.empty(
            (mla.n_local_groups, num_tokens, heads_per_group * mla.head_dim),
            dtype=torch.float8_e4m3fn,
            device=attn_o.device,
        )
        o_scale_dtype = torch.int32 if mla._tma_aligned_scales else torch.float32
        o_scale_buf = torch.empty(
            mla.n_local_groups * scale_inner * tma_aligned_t,
            dtype=o_scale_dtype,
            device=attn_o.device,
        ).as_strided(
            (mla.n_local_groups, num_tokens, scale_inner),
            (scale_inner * tma_aligned_t, 1, tma_aligned_t),
        )
        pdl_kwargs = {} if current_platform.is_rocm() else {"launch_pdl": False}
        _fused_inv_rope_fp8_quant_per_head[
            (tma_aligned_t, mla.n_local_groups * heads_per_group)
        ](
            attn_o,
            positions,
            mla.rotary_emb.cos_sin_cache,
            o_fp8_buf,
            o_scale_buf,
            num_tokens,
            heads_per_group=heads_per_group,
            o_stride_token=attn_o.stride(0),
            o_stride_head=attn_o.stride(1),
            cache_stride_pos=mla.rotary_emb.cos_sin_cache.stride(0),
            fp8_stride_group=o_fp8_buf.stride(0),
            fp8_stride_token=o_fp8_buf.stride(1),
            scale_stride_group=o_scale_buf.stride(0),
            scale_stride_k=o_scale_buf.stride(2),
            fp8_max=torch.finfo(torch.float8_e4m3fn).max,
            eps=1e-10,
            QUANT_GROUP_SIZE=quant_group_size,
            CHUNKS_PER_HEAD=chunks_per_head,
            ROPE_START=mla.nope_head_dim % quant_group_size,
            HALF_ROPE=mla.rope_head_dim // 2,
            TMA_ALIGNED_SCALES=mla._tma_aligned_scales,
            num_stages=1,
            **pdl_kwargs,
            num_warps=1,
        )
        o_fp8 = o_fp8_buf.transpose(0, 1)
        o_scale = o_scale_buf.transpose(0, 1)
        z = torch.empty(
            (num_tokens, mla.n_local_groups, mla.o_lora_rank),
            device=attn_o.device,
            dtype=torch.bfloat16,
        )
        fp8_einsum(
            "bhr,hdr->bhd",
            (o_fp8, o_scale),
            (mla.wo_a.weight, mla.wo_a.weight_scale_inv),
            z,
            recipe=tuple(mla._einsum_recipe),
        )
        wo_b = mla.wo_b
        if wo_b.input_is_parallel:
            wo_b_input = z.flatten(1)
        else:
            split_input = split_tensor_along_last_dim(z.flatten(1), wo_b.tp_size)
            wo_b_input = split_input[wo_b.tp_rank].contiguous()
        wo_b_bias = None if (wo_b.tp_rank > 0 or wo_b.skip_bias_add) else wo_b.bias
        wo_b_input_2d = wo_b_input.contiguous().view(-1, wo_b_input.shape[-1])
        wo_b_q = torch.empty_like(wo_b_input_2d, dtype=torch.float8_e4m3fn)
        wo_b_scale = torch.empty(
            wo_b_input_2d.shape[:-1] + (wo_b_input_2d.shape[-1] // 128,),
            dtype=torch.float32,
            device=wo_b_input_2d.device,
        )
        fp8_min, fp8_max = get_fp8_min_max()
        _per_token_group_quant_fp8[(wo_b_input_2d.numel() // 128,)](
            wo_b_input_2d,
            wo_b_q,
            wo_b_scale,
            128,
            wo_b_input_2d.shape[1],
            wo_b_input_2d.stride(0),
            1e-10,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            use_ue8m0=True,
            BLOCK=128,
            num_warps=1,
            num_stages=1,
        )
        wo_b_weight_scale = getattr(wo_b, "weight_scale_inv", None)
        if wo_b_weight_scale is None:
            wo_b_weight_scale = wo_b.weight_scale
        hidden_states = w8a8_triton_block_scaled_mm(
            wo_b_q,
            wo_b.weight,
            wo_b_scale,
            wo_b_weight_scale,
            [128, 128],
            output_dtype=torch.bfloat16,
        ).view(*wo_b_input.shape[:-1], wo_b.weight.shape[0])
        if wo_b_bias is not None:
            hidden_states = hidden_states + wo_b_bias
        if wo_b.reduce_results and wo_b.tp_size > 1:
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)

        # The dummy profiling path has zero attention output.
        residual = (
            torch.einsum(
                "...ij,...ih->...jh",
                res_mix.to(torch.float32),
                residual.to(torch.float32),
            )
            + post_mix.to(torch.float32) * hidden_states.unsqueeze(-2).to(torch.float32)
        ).to(residual.dtype)
        hc_mult = residual.shape[-2]
        hidden_size = residual.shape[-1]
        residual_flat = residual.view(-1, hc_mult, hidden_size)
        outer_shape = residual.shape[:-2]
        num_tokens = residual_flat.shape[0]
        hc_x = residual_flat.view(num_tokens, hc_mult * hidden_size).to(torch.float32)
        mixes = torch.matmul(hc_x, layer_params.hc_ffn_fn.t())
        sqrsum = hc_x.square().sum(dim=-1, keepdim=True)
        mixes = mixes * torch.rsqrt(
            sqrsum / (hc_mult * hidden_size) + layer.rms_norm_eps
        )
        pre_mix = (
            torch.sigmoid(
                mixes[:, :hc_mult] * layer_params.hc_ffn_scale[0]
                + layer_params.hc_ffn_base[:hc_mult]
            )
            + layer.hc_eps
        )
        post_mix = (
            torch.sigmoid(
                mixes[:, hc_mult : 2 * hc_mult] * layer_params.hc_ffn_scale[1]
                + layer_params.hc_ffn_base[hc_mult : 2 * hc_mult]
            )
            * layer.hc_post_alpha
        )
        comb_logits = mixes[:, 2 * hc_mult :].view(num_tokens, hc_mult, hc_mult)
        comb_logits = comb_logits * layer_params.hc_ffn_scale[
            2
        ] + layer_params.hc_ffn_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
        res_mix = torch.softmax(comb_logits, dim=-1) + layer.hc_eps
        res_mix = res_mix / (res_mix.sum(dim=-2, keepdim=True) + layer.hc_eps)
        for _ in range(layer.hc_sinkhorn_iters - 1):
            res_mix = res_mix / (res_mix.sum(dim=-1, keepdim=True) + layer.hc_eps)
            res_mix = res_mix / (res_mix.sum(dim=-2, keepdim=True) + layer.hc_eps)
        hidden_states = torch.sum(
            pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1
        ).to(torch.bfloat16)
        hidden_states = hidden_states.view(*outer_shape, hidden_size)
        post_mix = post_mix.view(*outer_shape, hc_mult, 1)
        res_mix = res_mix.view(*outer_shape, hc_mult, hc_mult)
        return hidden_states, residual, post_mix, res_mix

    if mla.n_local_heads < mla.padded_heads:
        q = F.pad(q, (0, 0, 0, mla.padded_heads - mla.n_local_heads), value=0.0)
    sub = mla.mla_attn
    flashmla_metadata = attn_metadata.get(sub.prefix)
    swa_metadata = attn_metadata.get(sub.swa_cache_layer.prefix)
    assert swa_metadata is not None
    swa_only = sub.compress_ratio <= 1
    self_kv_cache = sub.kv_cache if not swa_only else None
    swa_kv_cache = sub.swa_cache_layer.kv_cache
    num_decodes = swa_metadata.num_decodes
    num_prefills = swa_metadata.num_prefills
    num_decode_tokens = swa_metadata.num_decode_tokens
    if num_prefills > 0:
        prefill_q = q[num_decode_tokens:]
        prefill_output = o_padded[num_decode_tokens:]
        prefill_swa_only = flashmla_metadata is None
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]
        if not prefill_swa_only:
            if sub.compress_ratio == 4:
                assert sub.topk_indices_buffer is not None
                topk_indices = sub.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                assert flashmla_metadata is not None
                topk_indices = flashmla_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
            n_compressed = (
                sub.max_model_len + sub.compress_ratio - 1
            ) // sub.compress_ratio
        else:
            assert sub.topk_indices_buffer is not None
            topk_indices = sub.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            n_compressed = 0
        workspace_width = n_compressed + sub.window_size + sub.max_num_batched_tokens
        num_chunks = (num_prefills + PREFILL_CHUNK_SIZE - 1) // PREFILL_CHUNK_SIZE
        kv_workspace = current_workspace_manager().get_simultaneous(
            ((PREFILL_CHUNK_SIZE, workspace_width, prefill_q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not prefill_swa_only:
                assert flashmla_metadata is not None
                block_table = flashmla_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv_workspace[:chunk_size],
                    self_kv_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // sub.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=flashmla_metadata.block_size // sub.compress_ratio,
                    offset=0,
                )
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv_workspace[:chunk_size],
                swa_kv_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=n_compressed,
            )
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                sub.window_size,
                sub.compress_ratio,
                top_k,
                workspace_width,
                n_compressed,
            )
            flash_mla_sparse_fwd(
                q=prefill_q[query_start:query_end],
                kv=kv_workspace.view(-1, 1, prefill_q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=sub.scale,
                attn_sink=sub.attn_sink,
                topk_length=combined_lens,
                out=prefill_output[query_start:query_end],
            )
    if num_decodes > 0:
        decode_q = q[:num_decode_tokens]
        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert flashmla_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = flashmla_metadata.block_size // sub.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if sub.compress_ratio == 4:
                assert sub.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    sub.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    flashmla_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                topk_indices = flashmla_metadata.c128a_global_decode_topk_indices
                topk_lens = flashmla_metadata.c128a_decode_topk_lens
        if sub.compress_ratio <= 1:
            tile_metadata = swa_metadata.tile_sched_swaonly
        elif sub.compress_ratio == 4:
            tile_metadata = swa_metadata.tile_sched_c4a
        elif sub.compress_ratio == 128:
            tile_metadata = swa_metadata.tile_sched_c128a
        else:
            raise ValueError(
                f"Unsupported compress_ratio={sub.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        assert tile_metadata is not None
        flash_mla_with_kvcache(
            q=decode_q.unsqueeze(1),
            k_cache=sub.swa_cache_layer.kv_cache.unsqueeze(-2),
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=tile_metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_metadata.decode_swa_indices,
            topk_length=swa_metadata.decode_swa_lens,
            softmax_scale=sub.scale,
            attn_sink=sub.attn_sink,
            extra_k_cache=self_kv_cache.unsqueeze(-2) if not swa_only else None,
            extra_indices_in_kvcache=topk_indices,
            extra_topk_length=topk_lens,
            out=o_padded[:num_decode_tokens].unsqueeze(1),
        )
    attn_o = o_padded[:, : mla.n_local_heads, :]
    heads_per_group = mla.n_local_heads // mla.n_local_groups
    quant_group_size = 128
    chunks_per_head = mla.head_dim // quant_group_size
    num_scale_blocks = heads_per_group * mla.head_dim // quant_group_size
    tma_aligned_t = get_tma_aligned_size(num_tokens, 4)
    scale_inner = (
        (num_scale_blocks + 3) // 4
        if mla._tma_aligned_scales
        else num_scale_blocks
    )
    o_fp8_buf = torch.empty(
        (mla.n_local_groups, num_tokens, heads_per_group * mla.head_dim),
        dtype=torch.float8_e4m3fn,
        device=attn_o.device,
    )
    o_scale_dtype = torch.int32 if mla._tma_aligned_scales else torch.float32
    o_scale_buf = torch.empty(
        mla.n_local_groups * scale_inner * tma_aligned_t,
        dtype=o_scale_dtype,
        device=attn_o.device,
    ).as_strided(
        (mla.n_local_groups, num_tokens, scale_inner),
        (scale_inner * tma_aligned_t, 1, tma_aligned_t),
    )
    pdl_kwargs = {} if current_platform.is_rocm() else {"launch_pdl": False}
    _fused_inv_rope_fp8_quant_per_head[
        (tma_aligned_t, mla.n_local_groups * heads_per_group)
    ](
        attn_o,
        positions,
        mla.rotary_emb.cos_sin_cache,
        o_fp8_buf,
        o_scale_buf,
        num_tokens,
        heads_per_group=heads_per_group,
        o_stride_token=attn_o.stride(0),
        o_stride_head=attn_o.stride(1),
        cache_stride_pos=mla.rotary_emb.cos_sin_cache.stride(0),
        fp8_stride_group=o_fp8_buf.stride(0),
        fp8_stride_token=o_fp8_buf.stride(1),
        scale_stride_group=o_scale_buf.stride(0),
        scale_stride_k=o_scale_buf.stride(2),
        fp8_max=torch.finfo(torch.float8_e4m3fn).max,
        eps=1e-10,
        QUANT_GROUP_SIZE=quant_group_size,
        CHUNKS_PER_HEAD=chunks_per_head,
        ROPE_START=mla.nope_head_dim % quant_group_size,
        HALF_ROPE=mla.rope_head_dim // 2,
        TMA_ALIGNED_SCALES=mla._tma_aligned_scales,
        num_stages=1,
        **pdl_kwargs,
        num_warps=1,
    )
    o_fp8 = o_fp8_buf.transpose(0, 1)
    o_scale = o_scale_buf.transpose(0, 1)
    z = torch.empty(
        (num_tokens, mla.n_local_groups, mla.o_lora_rank),
        device=attn_o.device,
        dtype=torch.bfloat16,
    )
    fp8_einsum(
        "bhr,hdr->bhd",
        (o_fp8, o_scale),
        (mla.wo_a.weight, mla.wo_a.weight_scale_inv),
        z,
        recipe=tuple(mla._einsum_recipe),
    )
    wo_b = mla.wo_b
    if wo_b.input_is_parallel:
        wo_b_input = z.flatten(1)
    else:
        split_input = split_tensor_along_last_dim(z.flatten(1), wo_b.tp_size)
        wo_b_input = split_input[wo_b.tp_rank].contiguous()
    wo_b_bias = None if (wo_b.tp_rank > 0 or wo_b.skip_bias_add) else wo_b.bias
    wo_b_input_2d = wo_b_input.contiguous().view(-1, wo_b_input.shape[-1])
    wo_b_q = torch.empty_like(wo_b_input_2d, dtype=torch.float8_e4m3fn)
    wo_b_scale = torch.empty(
        wo_b_input_2d.shape[:-1] + (wo_b_input_2d.shape[-1] // 128,),
        dtype=torch.float32,
        device=wo_b_input_2d.device,
    )
    fp8_min, fp8_max = get_fp8_min_max()
    _per_token_group_quant_fp8[(wo_b_input_2d.numel() // 128,)](
        wo_b_input_2d,
        wo_b_q,
        wo_b_scale,
        128,
        wo_b_input_2d.shape[1],
        wo_b_input_2d.stride(0),
        1e-10,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        use_ue8m0=True,
        BLOCK=128,
        num_warps=1,
        num_stages=1,
    )
    wo_b_weight_scale = getattr(wo_b, "weight_scale_inv", None)
    if wo_b_weight_scale is None:
        wo_b_weight_scale = wo_b.weight_scale
    hidden_states = w8a8_triton_block_scaled_mm(
        wo_b_q,
        wo_b.weight,
        wo_b_scale,
        wo_b_weight_scale,
        [128, 128],
        output_dtype=torch.bfloat16,
    ).view(*wo_b_input.shape[:-1], wo_b.weight.shape[0])
    if wo_b_bias is not None:
        hidden_states = hidden_states + wo_b_bias
    if wo_b.reduce_results and wo_b.tp_size > 1:
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)

    # HC attention post-mix + FFN pre-mix.
    residual = (
        torch.einsum(
            "...ij,...ih->...jh",
            res_mix.to(torch.float32),
            residual.to(torch.float32),
        )
        + post_mix.to(torch.float32) * hidden_states.unsqueeze(-2).to(torch.float32)
    ).to(residual.dtype)
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    outer_shape = residual.shape[:-2]
    num_tokens = residual_flat.shape[0]
    hc_x = residual_flat.view(num_tokens, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(hc_x, layer_params.hc_ffn_fn.t())
    sqrsum = hc_x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(
        sqrsum / (hc_mult * hidden_size) + layer.rms_norm_eps
    )
    pre_mix = (
        torch.sigmoid(
            mixes[:, :hc_mult] * layer_params.hc_ffn_scale[0]
            + layer_params.hc_ffn_base[:hc_mult]
        )
        + layer.hc_eps
    )
    post_mix = (
        torch.sigmoid(
            mixes[:, hc_mult : 2 * hc_mult] * layer_params.hc_ffn_scale[1]
            + layer_params.hc_ffn_base[hc_mult : 2 * hc_mult]
        )
        * layer.hc_post_alpha
    )
    comb_logits = mixes[:, 2 * hc_mult :].view(num_tokens, hc_mult, hc_mult)
    comb_logits = comb_logits * layer_params.hc_ffn_scale[
        2
    ] + layer_params.hc_ffn_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
    res_mix = torch.softmax(comb_logits, dim=-1) + layer.hc_eps
    res_mix = res_mix / (res_mix.sum(dim=-2, keepdim=True) + layer.hc_eps)
    for _ in range(layer.hc_sinkhorn_iters - 1):
        res_mix = res_mix / (res_mix.sum(dim=-1, keepdim=True) + layer.hc_eps)
        res_mix = res_mix / (res_mix.sum(dim=-2, keepdim=True) + layer.hc_eps)
    hidden_states = torch.sum(
        pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1
    ).to(torch.bfloat16)
    hidden_states = hidden_states.view(*outer_shape, hidden_size)
    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    res_mix = res_mix.view(*outer_shape, hc_mult, hc_mult)

    # MoE router/experts.
    ffn = layer_params.ffn
    norm_gate = ffn.norm_gate
    if norm_gate.tid2eid is not None and input_ids is None:
        raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")
    org_shape = hidden_states.shape
    variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    normed_x = (
        hidden_states
        * torch.rsqrt(variance + norm_gate.rms_eps)
        * norm_gate.norm.weight
    ).to(hidden_states.dtype)
    gate = norm_gate.gate
    if gate.allow_cublas_router_gemm and normed_x.dtype == torch.bfloat16:
        router_logits = torch.mm(normed_x, gate.weight.T, out_dtype=torch.float32)
    else:
        gate_x = normed_x
        if gate.out_dtype is not None and gate_x.dtype != gate.weight.dtype:
            gate_x = gate_x.to(gate.weight.dtype)
        router_logits = F.linear(gate_x, gate.weight, gate.bias)
        if gate.out_dtype is not None and router_logits.dtype != gate.out_dtype:
            router_logits = router_logits.to(gate.out_dtype)
    if ffn.scoring_func == "sqrtsoftplus":
        scores = F.softplus(router_logits).sqrt()
    elif ffn.scoring_func == "softmax":
        scores = router_logits.softmax(dim=-1)
    elif ffn.scoring_func == "sigmoid":
        scores = router_logits.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {ffn.scoring_func}")
    if norm_gate.e_score_correction_bias is not None:
        scores_for_choice = scores + norm_gate.e_score_correction_bias.data.unsqueeze(0)
    else:
        scores_for_choice = scores
    if norm_gate.tid2eid is not None:
        assert input_ids is not None
        topk_ids = norm_gate.tid2eid[input_ids]
    else:
        topk_ids = torch.topk(
            scores_for_choice,
            k=ffn.n_activated_experts,
            dim=-1,
            sorted=False,
        )[1]
    topk_weights = scores.gather(1, topk_ids)
    if ffn.renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(torch.float32)
    if ffn.routed_scaling_factor != 1.0:
        topk_weights *= ffn.routed_scaling_factor
    topk_ids = topk_ids.to(ffn.hash_indices_dtype)

    import vllm.third_party.deep_gemm as deep_gemm

    if normed_x.shape[0] > layer_params.moe_max_num_tokens:
        raise ValueError(
            f"DeepSeek V4 MegaMoE got {normed_x.shape[0]} tokens, "
            "but the symmetric buffer was sized for "
            f"{layer_params.moe_max_num_tokens}."
        )
    hidden_states = torch.empty_like(normed_x, dtype=torch.bfloat16)
    symm_buffer = layer_params.moe_symm_buffer
    assert symm_buffer is not None
    num_tokens = normed_x.shape[0]
    hidden_size = normed_x.shape[1]
    if num_tokens != 0 and hidden_size % 128 != 0:
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging requires hidden_size to be "
            "a multiple of 128."
        )
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging requires topk_weights and "
            "topk_ids to have the same shape."
        )
    if num_tokens != 0:
        block_k = 128
        grid = (num_tokens, triton.cdiv(hidden_size, block_k))
        block_topk = triton.next_power_of_2(topk_ids.shape[1])
        _deepseek_v4_stage_mega_moe_inputs_kernel[grid](
            normed_x,
            symm_buffer.x[:num_tokens],
            symm_buffer.x_sf[:num_tokens],
            topk_ids,
            topk_weights,
            symm_buffer.topk_idx[:num_tokens],
            symm_buffer.topk_weights[:num_tokens],
            normed_x.stride(0),
            normed_x.stride(1),
            symm_buffer.x.stride(0),
            symm_buffer.x.stride(1),
            symm_buffer.x_sf.stride(0),
            symm_buffer.x_sf.stride(1),
            topk_ids.stride(0),
            topk_ids.stride(1),
            topk_weights.stride(0),
            topk_weights.stride(1),
            symm_buffer.topk_idx.stride(0),
            symm_buffer.topk_idx.stride(1),
            symm_buffer.topk_weights.stride(0),
            symm_buffer.topk_weights.stride(1),
            hidden_size,
            topk_ids.shape[1],
            BLOCK_K=block_k,
            GROUP_K=32,
            BLOCK_TOPK=block_topk,
            num_warps=4,
        )
    assert layer_params.moe_l1_weights is not None
    assert layer_params.moe_l2_weights is not None
    activation_clamp = float(ffn.swiglu_limit) if ffn.swiglu_limit is not None else None
    deep_gemm.fp8_fp4_mega_moe(
        hidden_states,
        layer_params.moe_l1_weights,
        layer_params.moe_l2_weights,
        symm_buffer,
        activation_clamp=activation_clamp,
        fast_math=True,
    )
    if layer_params.shared_experts is not None:
        shared = layer_params.shared_experts
        gate_up_proj = shared.gate_up_proj
        gate_up_input_2d = normed_x.contiguous().view(-1, normed_x.shape[-1])
        gate_up_q = torch.empty_like(gate_up_input_2d, dtype=torch.float8_e4m3fn)
        gate_up_scale = torch.empty(
            gate_up_input_2d.shape[:-1] + (gate_up_input_2d.shape[-1] // 128,),
            dtype=torch.float32,
            device=gate_up_input_2d.device,
        )
        fp8_min, fp8_max = get_fp8_min_max()
        _per_token_group_quant_fp8[(gate_up_input_2d.numel() // 128,)](
            gate_up_input_2d,
            gate_up_q,
            gate_up_scale,
            128,
            gate_up_input_2d.shape[1],
            gate_up_input_2d.stride(0),
            1e-10,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            use_ue8m0=True,
            BLOCK=128,
            num_warps=1,
            num_stages=1,
        )
        gate_up_weight_scale = getattr(gate_up_proj, "weight_scale_inv", None)
        if gate_up_weight_scale is None:
            gate_up_weight_scale = gate_up_proj.weight_scale
        gate_up = w8a8_triton_block_scaled_mm(
            gate_up_q,
            gate_up_proj.weight,
            gate_up_scale,
            gate_up_weight_scale,
            [128, 128],
            output_dtype=torch.bfloat16,
        ).view(*normed_x.shape[:-1], gate_up_proj.weight.shape[0])
        if gate_up_proj.bias is not None:
            gate_up = gate_up + gate_up_proj.bias
        d = gate_up.shape[-1] // 2
        if shared.swiglu_limit is None:
            shared_hidden = F.silu(gate_up[..., :d]) * gate_up[..., d:]
        else:
            gate_part = torch.clamp(gate_up[..., :d], max=float(shared.swiglu_limit))
            up_part = torch.clamp(
                gate_up[..., d:],
                min=-float(shared.swiglu_limit),
                max=float(shared.swiglu_limit),
            )
            shared_hidden = F.silu(gate_part) * up_part
        down = shared.down_proj
        if down.input_is_parallel:
            down_input = shared_hidden
        else:
            split_input = split_tensor_along_last_dim(shared_hidden, down.tp_size)
            down_input = split_input[down.tp_rank].contiguous()
        down_bias = None if (down.tp_rank > 0 or down.skip_bias_add) else down.bias
        down_input_2d = down_input.contiguous().view(-1, down_input.shape[-1])
        down_q = torch.empty_like(down_input_2d, dtype=torch.float8_e4m3fn)
        down_scale = torch.empty(
            down_input_2d.shape[:-1] + (down_input_2d.shape[-1] // 128,),
            dtype=torch.float32,
            device=down_input_2d.device,
        )
        fp8_min, fp8_max = get_fp8_min_max()
        _per_token_group_quant_fp8[(down_input_2d.numel() // 128,)](
            down_input_2d,
            down_q,
            down_scale,
            128,
            down_input_2d.shape[1],
            down_input_2d.stride(0),
            1e-10,
            fp8_min=fp8_min,
            fp8_max=fp8_max,
            use_ue8m0=True,
            BLOCK=128,
            num_warps=1,
            num_stages=1,
        )
        down_weight_scale = getattr(down, "weight_scale_inv", None)
        if down_weight_scale is None:
            down_weight_scale = down.weight_scale
        shared_output = w8a8_triton_block_scaled_mm(
            down_q,
            down.weight,
            down_scale,
            down_weight_scale,
            [128, 128],
            output_dtype=torch.bfloat16,
        ).view(*down_input.shape[:-1], down.weight.shape[0])
        if down_bias is not None:
            shared_output = shared_output + down_bias
        if down.reduce_results and down.tp_size > 1:
            shared_output = tensor_model_parallel_all_reduce(shared_output)
        hidden_states += shared_output
    hidden_states = hidden_states.view(org_shape)
    return hidden_states, residual, post_mix, res_mix


def flat_forward(
    model: "FlatDeepseekV4Model",
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors:
    params = getattr(model, "_flat_deepseek_v4_params", None)
    if params is None:
        params = extract_all_layer_params(model)
        model._flat_deepseek_v4_params = params

    if get_pp_group().is_first_rank:
        # Embedding.
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            embed_tokens = params.embed_tokens
            if embed_tokens.tp_size > 1:
                masked_input, input_mask = get_masked_input_and_mask(
                    input_ids,
                    embed_tokens.shard_indices.org_vocab_start_index,
                    embed_tokens.shard_indices.org_vocab_end_index,
                    embed_tokens.shard_indices.num_org_vocab_padding,
                    embed_tokens.shard_indices.added_vocab_start_index,
                    embed_tokens.shard_indices.added_vocab_end_index,
                )
            else:
                masked_input = input_ids
                input_mask = None
            output_parallel = F.embedding(masked_input.long(), embed_tokens.weight)
            if input_mask is not None:
                output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
            hidden_states = tensor_model_parallel_all_reduce(output_parallel)
        hidden_states = hidden_states.unsqueeze(-2).repeat(1, model.hc_mult, 1)
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]

    if model.use_mega_moe:
        input_ids = input_ids.to(torch.int64)

    residual, post_mix, res_mix = None, None, None
    last_layer_params = None
    for layer_params in params.layers:
        last_layer_params = layer_params
        hidden_states, residual, post_mix, res_mix = transformer_layer(
            layer_params,
            hidden_states,
            positions,
            input_ids,
            residual,
            post_mix,
            res_mix,
        )

    if last_layer_params is not None and current_platform.is_cuda():
        # Final HC post-mix after the last layer.
        hidden_states = (
            torch.einsum(
                "...ij,...ih->...jh",
                res_mix.to(torch.float32),
                residual.to(torch.float32),
            )
            + post_mix.to(torch.float32)
            * hidden_states.unsqueeze(-2).to(torch.float32)
        ).to(residual.dtype)

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({"hidden_states": hidden_states})

    # Final HC head and RMSNorm.
    num_tokens = hidden_states.shape[0]
    model._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

    hc_mult, hidden_size = hidden_states.shape[-2:]
    outer_shape = hidden_states.shape[:-2]
    hs_flat = hidden_states.view(-1, hc_mult, hidden_size)
    hs_flattened = hs_flat.flatten(-2)
    hs_variance = hs_flattened.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hs_normed = (
        hs_flattened * torch.rsqrt(hs_variance + model.rms_norm_eps)
    ).to(hs_flattened.dtype)
    mixes = F.linear(hs_normed.float(), params.hc_head_fn)
    pre_mix = torch.sigmoid(
        mixes * params.hc_head_scale + params.hc_head_base
    ) + model.hc_eps
    hidden_states = torch.sum(
        pre_mix.unsqueeze(-1) * hs_flat.to(torch.float32), dim=1
    ).to(torch.bfloat16)
    hidden_states = hidden_states.view(*outer_shape, hidden_size)
    assert params.norm_weight is not None
    final_variance = hidden_states.to(torch.float32).pow(2).mean(
        dim=-1, keepdim=True
    )
    hidden_states = (
        hidden_states
        * torch.rsqrt(final_variance + model.rms_norm_eps)
        * params.norm_weight
    ).to(hidden_states.dtype)
    return hidden_states


class FlatDeepseekV4Model(DeepseekV4Model):
    """DeepSeek V4 backbone using ``flat_forward``."""

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return flat_forward(
            self,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )
