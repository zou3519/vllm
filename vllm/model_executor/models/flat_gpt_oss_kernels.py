# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import is_quantized_kv_cache


@triton.jit
def _rope_and_cache_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    query_stride: tl.int64,
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
    tl.store(query_ptr + q_base + q_dim, q_out, mask=q_mask)

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
    tl.store(key_ptr + k_base + kv_dim, k_out, mask=kv_mask)

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
    tile_size = min(2048, triton.next_power_of_2(n))
    grid = (slot_mapping.shape[0], triton.cdiv(n, tile_size))

    _rope_and_cache_kernel[grid](
        query,
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        k_scale,
        v_scale,
        query.stride(0),
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
        fp8_kv_cache,
        tile_size,
        num_warps=8,
        num_stages=4,
    )
    return torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)
