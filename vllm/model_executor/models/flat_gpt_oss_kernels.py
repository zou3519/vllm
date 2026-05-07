# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import is_quantized_kv_cache


@triton.jit
def _fused_add_rms_norm_mxfp8_quant_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    quant_output_ptr,
    scale_output_ptr,
    input_stride: tl.int64,
    residual_stride: tl.int64,
    eps: tl.float32,
    hidden_size: tl.constexpr,
    padded_hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCALE_BLOCKS: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_SIZE)
    hidden_mask = offs < hidden_size

    input_base = row * input_stride
    residual_base = row * residual_stride
    x = tl.load(input_ptr + input_base + offs, mask=hidden_mask,
                other=0.0).to(tl.float32)
    residual = tl.load(
        residual_ptr + residual_base + offs,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)
    added = x + residual
    variance = tl.sum(added * added, axis=0) / hidden_size
    norm_scale = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offs, mask=hidden_mask,
                     other=0.0).to(tl.float32)
    normed = added * norm_scale * weight

    tl.store(input_ptr + input_base + offs, normed, mask=hidden_mask)
    tl.store(residual_ptr + residual_base + offs, added, mask=hidden_mask)

    quant_mask = offs < padded_hidden_size
    quant_normed = tl.where(hidden_mask, normed, 0.0)
    quant_blocks = tl.reshape(quant_normed, [SCALE_BLOCKS, 32])
    quant_valid = tl.reshape(hidden_mask, [SCALE_BLOCKS, 32])
    block_amax = tl.max(tl.where(quant_valid, tl.abs(quant_blocks), 0.0),
                        axis=1)

    dequant_scale = block_amax / 448.0
    dequant_scale_exponent = (
        dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF
    ) & 0x7F800000
    dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
    quant_scale = tl.where(dequant_scale_rounded == 0.0, 0.0,
                           1.0 / dequant_scale_rounded)
    quantized = quant_blocks * tl.expand_dims(quant_scale, axis=1)
    quantized = tl.reshape(quantized, [BLOCK_SIZE])
    quantized = tl.where(quant_mask, quantized, 0.0)
    tl.store(
        quant_output_ptr + row * padded_hidden_size + offs,
        quantized,
        mask=quant_mask,
    )

    scale_idx = tl.arange(0, SCALE_BLOCKS)
    scale_mask = scale_idx < padded_hidden_size // 32
    scale_bytes = (dequant_scale_exponent >> 23).to(tl.uint8)
    tl.store(
        scale_output_ptr + row * (padded_hidden_size // 32) + scale_idx,
        scale_bytes,
        mask=scale_mask,
    )


def fused_add_rms_norm_mxfp8_quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    alignment: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = input.shape[-1]
    if hidden_size % 32 != 0:
        raise ValueError("Flat GPT-OSS MXFP8 quantization requires 32-wide blocks")
    padded_hidden_size = (hidden_size + alignment - 1) // alignment * alignment
    num_rows = input.numel() // hidden_size
    quant_output = torch.empty(
        (*input.shape[:-1], padded_hidden_size),
        dtype=torch.float8_e4m3fn,
        device=input.device,
    )
    scale_output = torch.empty(
        (num_rows * padded_hidden_size // 32,),
        dtype=torch.uint8,
        device=input.device,
    )

    block_size = triton.next_power_of_2(padded_hidden_size)
    scale_blocks = block_size // 32
    _fused_add_rms_norm_mxfp8_quant_kernel[(num_rows,)](
        input,
        residual,
        weight,
        quant_output,
        scale_output,
        input.stride(-2),
        residual.stride(-2),
        eps,
        hidden_size,
        padded_hidden_size,
        block_size,
        scale_blocks,
        num_warps=8,
    )
    return quant_output, scale_output


@triton.jit
def _rope_and_cache_kernel(
    query_ptr,
    query_out_ptr,
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    q_scale_ptr,
    k_scale_ptr,
    v_scale_ptr,
    query_stride: tl.int64,
    query_out_stride: tl.int64,
    key_stride: tl.int64,
    value_stride: tl.int64,
    key_cache_block_stride: tl.int64,
    key_cache_slot_stride: tl.int64,
    key_cache_head_stride: tl.int64,
    key_cache_dim_stride: tl.int64,
    value_cache_block_stride: tl.int64,
    value_cache_slot_stride: tl.int64,
    value_cache_head_stride: tl.int64,
    value_cache_dim_stride: tl.int64,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    rotary_dim: tl.constexpr,
    QUERY_FP8: tl.constexpr,
    FP8_KV_CACHE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    tile_idx = tl.program_id(axis=1)
    offs = tile_idx * TILE_SIZE + tl.arange(0, TILE_SIZE)

    embed_dim: tl.constexpr = rotary_dim // 2
    q_rot_elems: tl.constexpr = num_q_heads * rotary_dim
    kv_elems: tl.constexpr = num_kv_heads * head_size

    pos = tl.load(positions_ptr + token_idx).to(tl.int64)

    q_mask = offs < q_rot_elems
    q_head = offs // rotary_dim
    q_dim = offs % rotary_dim
    q_pair = q_dim % embed_dim
    q_x_dim = tl.where(q_dim < embed_dim, q_dim, q_dim - embed_dim)
    q_y_dim = q_x_dim + embed_dim
    q_base = token_idx * query_stride + q_head * head_size
    q_x = tl.load(query_ptr + q_base + q_x_dim, mask=q_mask, other=0.0)
    q_y = tl.load(query_ptr + q_base + q_y_dim, mask=q_mask, other=0.0)
    q_cos = tl.load(cos_sin_cache_ptr + pos * rotary_dim + q_pair, mask=q_mask)
    q_sin = tl.load(
        cos_sin_cache_ptr + pos * rotary_dim + embed_dim + q_pair, mask=q_mask
    )
    q_out = tl.where(q_dim < embed_dim, q_x * q_cos - q_y * q_sin,
                     q_y * q_cos + q_x * q_sin)
    if QUERY_FP8:
        q_out_base = token_idx * query_out_stride + q_head * head_size
        tl.store(
            query_out_ptr + q_out_base + q_dim,
            q_out / tl.load(q_scale_ptr),
            mask=q_mask,
        )
    else:
        tl.store(query_ptr + q_base + q_dim, q_out, mask=q_mask)

    if tile_idx != 0:
        return

    kv_mask = offs < kv_elems
    kv_head = offs // head_size
    kv_dim = offs % head_size
    k_base = token_idx * key_stride + kv_head * head_size
    v_base = token_idx * value_stride + kv_head * head_size

    k_raw = tl.load(key_ptr + k_base + kv_dim, mask=kv_mask, other=0.0)
    k_pair = kv_dim % embed_dim
    k_x_dim = tl.where(kv_dim < embed_dim, kv_dim, kv_dim - embed_dim)
    k_y_dim = k_x_dim + embed_dim
    k_x = tl.load(
        key_ptr + k_base + k_x_dim,
        mask=kv_mask & (kv_dim < rotary_dim),
        other=0.0,
    )
    k_y = tl.load(
        key_ptr + k_base + k_y_dim,
        mask=kv_mask & (kv_dim < rotary_dim),
        other=0.0,
    )
    k_cos = tl.load(
        cos_sin_cache_ptr + pos * rotary_dim + k_pair,
        mask=kv_mask & (kv_dim < rotary_dim),
    )
    k_sin = tl.load(
        cos_sin_cache_ptr + pos * rotary_dim + embed_dim + k_pair,
        mask=kv_mask & (kv_dim < rotary_dim),
    )
    k_rot = tl.where(kv_dim < embed_dim, k_x * k_cos - k_y * k_sin,
                     k_y * k_cos + k_x * k_sin)
    k_out = tl.where(kv_dim < rotary_dim, k_rot, k_raw)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    cache_k = (
        block_idx * key_cache_block_stride
        + block_offset * key_cache_slot_stride
        + kv_head * key_cache_head_stride
        + kv_dim * key_cache_dim_stride
    )
    cache_v = (
        block_idx * value_cache_block_stride
        + block_offset * value_cache_slot_stride
        + kv_head * value_cache_head_stride
        + kv_dim * value_cache_dim_stride
    )

    v_out = tl.load(value_ptr + v_base + kv_dim, mask=kv_mask, other=0.0)
    if FP8_KV_CACHE:
        k_store = k_out / tl.load(k_scale_ptr)
        v_store = v_out / tl.load(v_scale_ptr)
    else:
        k_store = k_out
        v_store = v_out

    tl.store(key_cache_ptr + cache_k, k_store, mask=kv_mask)
    tl.store(value_cache_ptr + cache_v, v_store, mask=kv_mask)


def rope_and_cache(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    rotary_dim: int,
    query_output: torch.Tensor | None = None,
    query_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    key_cache, value_cache = kv_cache.unbind(1)

    fp8_kv_cache = is_quantized_kv_cache(kv_cache_dtype)
    if fp8_kv_cache:
        fp8_dtype = current_platform.fp8_dtype()
        if key_cache.dtype != fp8_dtype:
            key_cache = key_cache.view(fp8_dtype)
            value_cache = value_cache.view(fp8_dtype)

    if key_cache.shape[1] == num_kv_heads:
        block_size = key_cache.shape[2]
        key_slot_stride = key_cache.stride(2)
        key_head_stride = key_cache.stride(1)
        key_dim_stride = key_cache.stride(3)
        value_slot_stride = value_cache.stride(2)
        value_head_stride = value_cache.stride(1)
        value_dim_stride = value_cache.stride(3)
    else:
        block_size = key_cache.shape[1]
        key_slot_stride = key_cache.stride(1)
        key_head_stride = key_cache.stride(2)
        key_dim_stride = key_cache.stride(3)
        value_slot_stride = value_cache.stride(1)
        value_head_stride = value_cache.stride(2)
        value_dim_stride = value_cache.stride(3)

    n = max(num_q_heads * rotary_dim, num_kv_heads * head_size)
    tile_size = triton.next_power_of_2(n)
    grid = (slot_mapping.shape[0], triton.cdiv(n, tile_size))
    query_fp8 = query_output is not None
    if query_output is None:
        query_output = query
    if query_scale is None:
        query_scale = k_scale

    _rope_and_cache_kernel[grid](
        query,
        query_output,
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        query_scale,
        k_scale,
        v_scale,
        query.stride(0),
        query_output.stride(0),
        key.stride(0),
        value.stride(0),
        key_cache.stride(0),
        key_slot_stride,
        key_head_stride,
        key_dim_stride,
        value_cache.stride(0),
        value_slot_stride,
        value_head_stride,
        value_dim_stride,
        num_q_heads,
        num_kv_heads,
        head_size,
        block_size,
        rotary_dim,
        query_fp8,
        fp8_kv_cache,
        tile_size,
        num_warps=8,
        num_stages=4,
    )
    return kv_cache
