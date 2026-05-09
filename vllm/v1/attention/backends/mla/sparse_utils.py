# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for sparse MLA backends."""

import torch

from vllm.triton_utils import tl, triton


# Kernel with prefill workspace support and valid count tracking
@triton.jit
def _index_k_norm_rope_cache_kernel(
    index_k_ptr,  # [num_tokens, head_dim]
    positions_ptr,  # [num_tokens]
    cos_sin_cache_ptr,  # [max_position, rope_dim]
    norm_weight_ptr,  # [head_dim]
    norm_bias_ptr,  # [head_dim]
    slot_mapping_ptr,  # [num_tokens]
    kv_cache_fp8_ptr,  # flat fp8 view of [num_blocks, block_size, cache_stride]
    kv_cache_f32_ptr,  # flat float32 view of same storage for scale writes
    num_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    input_stride0,
    input_stride1,
    cache_block_size: tl.constexpr,
    cache_stride: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < head_dim

    x = tl.load(
        index_k_ptr + token * input_stride0 + cols * input_stride1,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / head_dim
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / head_dim
    inv_std = tl.rsqrt(var + EPS)
    weight = tl.load(norm_weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(norm_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    vals = centered * inv_std * weight + bias

    half_rope = rope_dim // 2
    pos = tl.load(positions_ptr + token)
    rope_pair = tl.where(cols < half_rope, cols, cols - half_rope)
    rope_mask = cols < rope_dim
    cos = tl.load(
        cos_sin_cache_ptr + pos * rope_dim + rope_pair,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + pos * rope_dim + half_rope + rope_pair,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    x_pair = tl.load(
        index_k_ptr + token * input_stride0 + rope_pair * input_stride1,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    y_pair = tl.load(
        index_k_ptr
        + token * input_stride0
        + (rope_pair + half_rope) * input_stride1,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    mean_pair = mean
    inv_pair = inv_std
    x_weight = tl.load(norm_weight_ptr + rope_pair, mask=rope_mask, other=0.0).to(
        tl.float32
    )
    y_weight = tl.load(
        norm_weight_ptr + rope_pair + half_rope,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    x_bias = tl.load(norm_bias_ptr + rope_pair, mask=rope_mask, other=0.0).to(
        tl.float32
    )
    y_bias = tl.load(
        norm_bias_ptr + rope_pair + half_rope,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    x_norm = (x_pair - mean_pair) * inv_pair * x_weight + x_bias
    y_norm = (y_pair - mean_pair) * inv_pair * y_weight + y_bias
    x_rot = x_norm * cos - y_norm * sin
    y_rot = y_norm * cos + x_norm * sin
    vals = tl.where(cols < half_rope, x_rot, vals)
    vals = tl.where((cols >= half_rope) & (cols < rope_dim), y_rot, vals)

    absmax = tl.max(tl.abs(tl.where(mask, vals, 0.0)), axis=0)
    scale = tl.maximum(absmax, 1.0e-4) / 448.0
    scale = tl.exp2(tl.ceil(tl.log2(scale)))
    q_vals = vals / scale
    q_vals = tl.minimum(tl.maximum(q_vals, -448.0), 448.0)

    slot = tl.load(slot_mapping_ptr + token)
    valid = slot >= 0
    block_idx = slot // cache_block_size
    block_offset = slot - block_idx * cache_block_size
    base = block_idx * cache_block_size * cache_stride
    data_offset = base + block_offset * head_dim + cols
    tl.store(kv_cache_fp8_ptr + data_offset, q_vals, mask=mask & valid)
    scale_byte_offset = base + cache_block_size * head_dim + block_offset * 4
    tl.store(kv_cache_f32_ptr + scale_byte_offset // 4, scale, mask=valid)


@triton.jit
def _index_q_rope_quant_weights_kernel(
    index_q_ptr,  # [num_tokens, n_heads, head_dim]
    positions_ptr,  # [num_tokens]
    cos_sin_cache_ptr,  # [max_position, rope_dim]
    index_weights_ptr,  # [num_tokens, n_heads]
    q_fp8_ptr,  # [num_tokens, n_heads, head_dim]
    scaled_weights_ptr,  # [num_tokens, n_heads]
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    index_q_stride0,
    index_q_stride1,
    index_q_stride2,
    weights_stride0,
    weights_stride1,
    q_fp8_stride0,
    q_fp8_stride1,
    q_fp8_stride2,
    scaled_weights_stride0,
    scaled_weights_stride1,
    FACTOR: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < head_dim

    vals = tl.load(
        index_q_ptr
        + token * index_q_stride0
        + head * index_q_stride1
        + cols * index_q_stride2,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    half_rope = rope_dim // 2
    pos = tl.load(positions_ptr + token)
    rope_pair = tl.where(cols < half_rope, cols, cols - half_rope)
    rope_mask = cols < rope_dim
    cos = tl.load(
        cos_sin_cache_ptr + pos * rope_dim + rope_pair,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + pos * rope_dim + half_rope + rope_pair,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    x_vals = tl.load(
        index_q_ptr
        + token * index_q_stride0
        + head * index_q_stride1
        + rope_pair * index_q_stride2,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    y_vals = tl.load(
        index_q_ptr
        + token * index_q_stride0
        + head * index_q_stride1
        + (rope_pair + half_rope) * index_q_stride2,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    x_rot = x_vals * cos - y_vals * sin
    y_rot = y_vals * cos + x_vals * sin
    vals = tl.where(cols < half_rope, x_rot, vals)
    vals = tl.where((cols >= half_rope) & (cols < rope_dim), y_rot, vals)

    absmax = tl.max(tl.abs(tl.where(mask, vals, 0.0)), axis=0)
    scale = absmax / 448.0
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(scale), 1.0e-10))))
    q_vals = vals / scale
    q_vals = tl.minimum(tl.maximum(q_vals, -448.0), 448.0)
    tl.store(
        q_fp8_ptr
        + token * q_fp8_stride0
        + head * q_fp8_stride1
        + cols * q_fp8_stride2,
        q_vals,
        mask=mask,
    )
    weight = tl.load(
        index_weights_ptr + token * weights_stride0 + head * weights_stride1
    ).to(tl.float32)
    tl.store(
        scaled_weights_ptr
        + token * scaled_weights_stride0
        + head * scaled_weights_stride1,
        weight * scale * FACTOR,
    )


@triton.jit
def _index_qk_rope_quant_cache_kernel(
    index_q_ptr,  # [num_tokens, n_heads, head_dim]
    index_k_ptr,  # [num_k_tokens, head_dim]
    positions_ptr,  # [num_tokens]
    cos_sin_cache_ptr,  # [max_position, rope_dim]
    index_weights_ptr,  # [num_tokens, n_heads]
    q_fp8_ptr,  # [num_tokens, n_heads, head_dim]
    scaled_weights_ptr,  # [num_tokens, n_heads]
    norm_weight_ptr,  # [head_dim]
    norm_bias_ptr,  # [head_dim]
    slot_mapping_ptr,  # [num_k_tokens]
    kv_cache_fp8_ptr,  # flat fp8 view of [num_blocks, block_size, cache_stride]
    kv_cache_f32_ptr,  # flat float32 view of same storage for scale writes
    num_k_tokens,
    n_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    index_q_stride0,
    index_q_stride1,
    index_q_stride2,
    index_k_stride0,
    index_k_stride1,
    weights_stride0,
    weights_stride1,
    q_fp8_stride0,
    q_fp8_stride1,
    q_fp8_stride2,
    scaled_weights_stride0,
    scaled_weights_stride1,
    cache_block_size: tl.constexpr,
    cache_stride: tl.constexpr,
    EPS: tl.constexpr,
    FACTOR: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    lane = tl.program_id(1)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < head_dim

    half_rope = rope_dim // 2
    pos = tl.load(positions_ptr + token)
    rope_pair = tl.where(cols < half_rope, cols, cols - half_rope)
    rope_mask = cols < rope_dim
    cos = tl.load(
        cos_sin_cache_ptr + pos * rope_dim + rope_pair,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + pos * rope_dim + half_rope + rope_pair,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)

    if lane < n_heads:
        vals = tl.load(
            index_q_ptr
            + token * index_q_stride0
            + lane * index_q_stride1
            + cols * index_q_stride2,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        x_vals = tl.load(
            index_q_ptr
            + token * index_q_stride0
            + lane * index_q_stride1
            + rope_pair * index_q_stride2,
            mask=rope_mask,
            other=0.0,
        ).to(tl.float32)
        y_vals = tl.load(
            index_q_ptr
            + token * index_q_stride0
            + lane * index_q_stride1
            + (rope_pair + half_rope) * index_q_stride2,
            mask=rope_mask,
            other=0.0,
        ).to(tl.float32)
        x_rot = x_vals * cos - y_vals * sin
        y_rot = y_vals * cos + x_vals * sin
        vals = tl.where(cols < half_rope, x_rot, vals)
        vals = tl.where((cols >= half_rope) & (cols < rope_dim), y_rot, vals)

        absmax = tl.max(tl.abs(tl.where(mask, vals, 0.0)), axis=0)
        scale = absmax / 448.0
        scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(scale), 1.0e-10))))
        q_vals = vals / scale
        q_vals = tl.minimum(tl.maximum(q_vals, -448.0), 448.0)
        tl.store(
            q_fp8_ptr
            + token * q_fp8_stride0
            + lane * q_fp8_stride1
            + cols * q_fp8_stride2,
            q_vals,
            mask=mask,
        )
        q_weight_scalar = tl.load(
            index_weights_ptr + token * weights_stride0 + lane * weights_stride1
        ).to(tl.float32)
        tl.store(
            scaled_weights_ptr
            + token * scaled_weights_stride0
            + lane * scaled_weights_stride1,
            q_weight_scalar * scale * FACTOR,
        )
    else:
        valid_token = token < num_k_tokens
        x = tl.load(
            index_k_ptr + token * index_k_stride0 + cols * index_k_stride1,
            mask=mask & valid_token,
            other=0.0,
        ).to(tl.float32)
        mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / head_dim
        centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / head_dim
        inv_std = tl.rsqrt(var + EPS)
        norm_weight_vals = tl.load(
            norm_weight_ptr + cols, mask=mask, other=0.0
        ).to(tl.float32)
        bias = tl.load(norm_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        vals = centered * inv_std * norm_weight_vals + bias

        x_pair = tl.load(
            index_k_ptr + token * index_k_stride0 + rope_pair * index_k_stride1,
            mask=rope_mask & valid_token,
            other=0.0,
        ).to(tl.float32)
        y_pair = tl.load(
            index_k_ptr
            + token * index_k_stride0
            + (rope_pair + half_rope) * index_k_stride1,
            mask=rope_mask & valid_token,
            other=0.0,
        ).to(tl.float32)
        x_weight = tl.load(
            norm_weight_ptr + rope_pair, mask=rope_mask, other=0.0
        ).to(tl.float32)
        y_weight = tl.load(
            norm_weight_ptr + rope_pair + half_rope,
            mask=rope_mask,
            other=0.0,
        ).to(tl.float32)
        x_bias = tl.load(norm_bias_ptr + rope_pair, mask=rope_mask, other=0.0).to(
            tl.float32
        )
        y_bias = tl.load(
            norm_bias_ptr + rope_pair + half_rope,
            mask=rope_mask,
            other=0.0,
        ).to(tl.float32)
        x_norm = (x_pair - mean) * inv_std * x_weight + x_bias
        y_norm = (y_pair - mean) * inv_std * y_weight + y_bias
        x_rot = x_norm * cos - y_norm * sin
        y_rot = y_norm * cos + x_norm * sin
        vals = tl.where(cols < half_rope, x_rot, vals)
        vals = tl.where((cols >= half_rope) & (cols < rope_dim), y_rot, vals)

        absmax = tl.max(tl.abs(tl.where(mask, vals, 0.0)), axis=0)
        scale = tl.maximum(absmax, 1.0e-4) / 448.0
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
        q_vals = vals / scale
        q_vals = tl.minimum(tl.maximum(q_vals, -448.0), 448.0)

        slot = tl.load(slot_mapping_ptr + token, mask=valid_token, other=-1)
        valid = slot >= 0
        block_idx = slot // cache_block_size
        block_offset = slot - block_idx * cache_block_size
        base = block_idx * cache_block_size * cache_stride
        data_offset = base + block_offset * head_dim + cols
        tl.store(kv_cache_fp8_ptr + data_offset, q_vals, mask=mask & valid)
        scale_byte_offset = base + cache_block_size * head_dim + block_offset * 4
        tl.store(kv_cache_f32_ptr + scale_byte_offset // 4, scale, mask=valid)


@triton.jit
def _mla_qkv_a_rmsnorm_kernel(
    qkv_ptr,  # [num_tokens, q_rank + kv_rank + rope_dim]
    q_weight_ptr,  # [q_rank]
    kv_weight_ptr,  # [kv_rank]
    q_out_ptr,  # [num_tokens, q_rank]
    kv_out_ptr,  # [num_tokens, kv_rank]
    q_rank: tl.constexpr,
    kv_rank: tl.constexpr,
    qkv_stride0,
    qkv_stride1,
    q_out_stride0,
    q_out_stride1,
    kv_out_stride0,
    kv_out_stride1,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    segment = tl.program_id(1)
    cols = tl.arange(0, BLOCK_N)
    dim = tl.where(segment == 0, q_rank, kv_rank)
    mask = cols < dim
    input_offset = tl.where(segment == 0, 0, q_rank)

    x = tl.load(
        qkv_ptr + token * qkv_stride0 + (input_offset + cols) * qkv_stride1,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sum_sq = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
    inv_rms = tl.rsqrt(sum_sq / dim + EPS)
    q_weight = tl.load(q_weight_ptr + cols, mask=mask & (segment == 0), other=0.0)
    kv_weight = tl.load(kv_weight_ptr + cols, mask=mask & (segment == 1), other=0.0)
    weight = tl.where(segment == 0, q_weight, kv_weight).to(tl.float32)
    vals = x * inv_rms * weight
    tl.store(
        q_out_ptr + token * q_out_stride0 + cols * q_out_stride1,
        vals,
        mask=mask & (segment == 0),
    )
    tl.store(
        kv_out_ptr + token * kv_out_stride0 + cols * kv_out_stride1,
        vals,
        mask=mask & (segment == 1),
    )


@triton.jit
def _mla_qkv_a_rmsnorm_k_rope_cache_fp8_kernel(
    qkv_ptr,  # [num_tokens, q_rank + kv_rank + rope_dim]
    q_weight_ptr,  # [q_rank]
    kv_weight_ptr,  # [kv_rank]
    q_out_ptr,  # [num_tokens, q_rank]
    positions_ptr,  # [num_tokens]
    cos_sin_cache_ptr,  # [max_position, rope_dim]
    slot_mapping_ptr,  # [num_tokens]
    kv_cache_fp8_ptr,  # flat fp8 view of [num_blocks, block_size, cache_stride]
    scale_ptr,  # scalar
    num_tokens,
    q_rank: tl.constexpr,
    kv_rank: tl.constexpr,
    rope_dim: tl.constexpr,
    qkv_stride0,
    qkv_stride1,
    q_out_stride0,
    q_out_stride1,
    cache_block_size: tl.constexpr,
    cache_stride: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    segment = tl.program_id(1)
    cols = tl.arange(0, BLOCK_N)

    if segment == 0:
        mask = cols < q_rank
        x = tl.load(
            qkv_ptr + token * qkv_stride0 + cols * qkv_stride1,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        sum_sq = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
        inv_rms = tl.rsqrt(sum_sq / q_rank + EPS)
        weight = tl.load(q_weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        vals = x * inv_rms * weight
        tl.store(
            q_out_ptr + token * q_out_stride0 + cols * q_out_stride1,
            vals,
            mask=mask,
        )
    else:
        kv_mask = cols < kv_rank
        kv_raw = tl.load(
            qkv_ptr + token * qkv_stride0 + (q_rank + cols) * qkv_stride1,
            mask=kv_mask,
            other=0.0,
        ).to(tl.float32)
        sum_sq = tl.sum(tl.where(kv_mask, kv_raw * kv_raw, 0.0), axis=0)
        inv_rms = tl.rsqrt(sum_sq / kv_rank + EPS)
        kv_weight = tl.load(kv_weight_ptr + cols, mask=kv_mask, other=0.0).to(
            tl.float32
        )
        kv_vals = kv_raw * inv_rms * kv_weight

        total_dim = kv_rank + rope_dim
        mask = cols < total_dim
        rope_feature = cols - kv_rank
        rope_mask = mask & (cols >= kv_rank)
        pair = rope_feature // 2
        x_feature = pair * 2
        y_feature = x_feature + 1
        pos = tl.load(positions_ptr + token, mask=token < num_tokens, other=0)
        cos = tl.load(
            cos_sin_cache_ptr + pos * rope_dim + pair,
            mask=rope_mask,
            other=0.0,
        ).to(tl.float32)
        sin = tl.load(
            cos_sin_cache_ptr + pos * rope_dim + rope_dim // 2 + pair,
            mask=rope_mask,
            other=0.0,
        ).to(tl.float32)
        rope_base = q_rank + kv_rank
        x_vals = tl.load(
            qkv_ptr + token * qkv_stride0 + (rope_base + x_feature) * qkv_stride1,
            mask=rope_mask,
            other=0.0,
        ).to(tl.float32)
        y_vals = tl.load(
            qkv_ptr + token * qkv_stride0 + (rope_base + y_feature) * qkv_stride1,
            mask=rope_mask,
            other=0.0,
        ).to(tl.float32)
        rope_vals = tl.where(
            (rope_feature % 2) == 0,
            x_vals * cos - y_vals * sin,
            y_vals * cos + x_vals * sin,
        )
        vals = tl.where(cols < kv_rank, kv_vals, rope_vals)

        slot = tl.load(slot_mapping_ptr + token, mask=token < num_tokens, other=-1)
        valid = slot >= 0
        block_idx = slot // cache_block_size
        block_offset = slot - block_idx * cache_block_size
        cache_offset = (
            block_idx * cache_block_size * cache_stride + block_offset * cache_stride
        )
        scale = tl.load(scale_ptr).to(tl.float32)
        q_vals = vals / scale
        q_vals = tl.minimum(tl.maximum(q_vals, -448.0), 448.0)
        tl.store(kv_cache_fp8_ptr + cache_offset + cols, q_vals, mask=mask & valid)


@triton.jit
def _mla_decode_q_concat_kernel(
    nope_ptr,  # [batch, heads, lora_rank]
    rope_ptr,  # [batch, heads, rope_dim]
    out_ptr,  # [batch, heads, lora_rank + rope_dim]
    total_elems,
    heads: tl.constexpr,
    lora_rank: tl.constexpr,
    rope_dim: tl.constexpr,
    out_dim: tl.constexpr,
    nope_stride0,
    nope_stride1,
    nope_stride2,
    rope_stride0,
    rope_stride1,
    rope_stride2,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < total_elems
    feature = offsets % out_dim
    tmp = offsets // out_dim
    head = tmp % heads
    batch = tmp // heads
    from_nope = feature < lora_rank
    nope_vals = tl.load(
        nope_ptr
        + batch * nope_stride0
        + head * nope_stride1
        + feature * nope_stride2,
        mask=mask & from_nope,
        other=0.0,
    )
    rope_feature = feature - lora_rank
    rope_vals = tl.load(
        rope_ptr
        + batch * rope_stride0
        + head * rope_stride1
        + rope_feature * rope_stride2,
        mask=mask & ~from_nope,
        other=0.0,
    )
    tl.store(
        out_ptr + offsets,
        tl.where(from_nope, nope_vals, rope_vals),
        mask=mask,
    )


@triton.jit
def _mla_decode_q_concat_quant_fp8_kernel(
    nope_ptr,  # [batch, heads, lora_rank]
    rope_ptr,  # [batch, heads, rope_dim]
    scale_ptr,  # scalar
    out_ptr,  # [batch, heads, lora_rank + rope_dim]
    total_elems,
    heads: tl.constexpr,
    lora_rank: tl.constexpr,
    rope_dim: tl.constexpr,
    out_dim: tl.constexpr,
    nope_stride0,
    nope_stride1,
    nope_stride2,
    rope_stride0,
    rope_stride1,
    rope_stride2,
    BLOCK_N: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < total_elems
    feature = offsets % out_dim
    tmp = offsets // out_dim
    head = tmp % heads
    batch = tmp // heads
    from_nope = feature < lora_rank
    nope_vals = tl.load(
        nope_ptr
        + batch * nope_stride0
        + head * nope_stride1
        + feature * nope_stride2,
        mask=mask & from_nope,
        other=0.0,
    ).to(tl.float32)
    rope_feature = feature - lora_rank
    rope_vals = tl.load(
        rope_ptr
        + batch * rope_stride0
        + head * rope_stride1
        + rope_feature * rope_stride2,
        mask=mask & ~from_nope,
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)
    vals = tl.where(from_nope, nope_vals, rope_vals) / scale
    vals = tl.minimum(tl.maximum(vals, -448.0), 448.0)
    tl.store(out_ptr + offsets, vals, mask=mask)


@triton.jit
def _mla_decode_q_project_concat_quant_fp8_kernel(
    q_ptr,  # [batch, heads, nope_dim + rope_dim]
    w_ptr,  # [heads, nope_dim, lora_rank]
    scale_ptr,  # scalar
    out_ptr,  # [batch, heads, lora_rank + rope_dim]
    heads: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    lora_rank: tl.constexpr,
    q_dim: tl.constexpr,
    out_dim: tl.constexpr,
    q_stride0,
    q_stride1,
    q_stride2,
    w_stride0,
    w_stride1,
    w_stride2,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    block = tl.program_id(2)
    offs_n = block * BLOCK_N + tl.arange(0, BLOCK_N)
    scale = tl.load(scale_ptr).to(tl.float32)

    if block * BLOCK_N < lora_rank:
        offs_k = tl.arange(0, BLOCK_K)
        q_vals = tl.load(
            q_ptr + batch * q_stride0 + head * q_stride1 + offs_k * q_stride2,
            mask=offs_k < nope_dim,
            other=0.0,
        )
        w_vals = tl.load(
            w_ptr
            + head * w_stride0
            + offs_k[:, None] * w_stride1
            + offs_n[None, :] * w_stride2,
            mask=(offs_k[:, None] < nope_dim) & (offs_n[None, :] < lora_rank),
            other=0.0,
        )
        vals = tl.reshape(tl.dot(q_vals[None, :], w_vals), [BLOCK_N]).to(
            tl.float32
        )
        vals = vals / scale
        vals = tl.minimum(tl.maximum(vals, -448.0), 448.0)
        tl.store(
            out_ptr
            + batch * heads * out_dim
            + head * out_dim
            + offs_n,
            vals,
            mask=offs_n < lora_rank,
        )
    else:
        rope_feature = offs_n - lora_rank
        vals = tl.load(
            q_ptr
            + batch * q_stride0
            + head * q_stride1
            + (nope_dim + rope_feature) * q_stride2,
            mask=rope_feature < rope_dim,
            other=0.0,
        ).to(tl.float32)
        vals = vals / scale
        vals = tl.minimum(tl.maximum(vals, -448.0), 448.0)
        tl.store(
            out_ptr
            + batch * heads * out_dim
            + head * out_dim
            + offs_n,
            vals,
            mask=rope_feature < rope_dim,
        )


@triton.jit
def _mla_decode_q_project_rope_concat_quant_fp8_kernel(
    q_ptr,  # [batch, heads, nope_dim + rope_dim]
    w_ptr,  # [heads, nope_dim, lora_rank]
    positions_ptr,  # [batch]
    cos_sin_cache_ptr,  # [max_position, rope_dim]
    scale_ptr,  # scalar
    out_ptr,  # [batch, heads, lora_rank + rope_dim]
    heads: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    lora_rank: tl.constexpr,
    q_dim: tl.constexpr,
    out_dim: tl.constexpr,
    q_stride0,
    q_stride1,
    q_stride2,
    w_stride0,
    w_stride1,
    w_stride2,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    block = tl.program_id(2)
    offs_n = block * BLOCK_N + tl.arange(0, BLOCK_N)
    scale = tl.load(scale_ptr).to(tl.float32)

    if block * BLOCK_N < lora_rank:
        offs_k = tl.arange(0, BLOCK_K)
        q_vals = tl.load(
            q_ptr + batch * q_stride0 + head * q_stride1 + offs_k * q_stride2,
            mask=offs_k < nope_dim,
            other=0.0,
        )
        w_vals = tl.load(
            w_ptr
            + head * w_stride0
            + offs_k[:, None] * w_stride1
            + offs_n[None, :] * w_stride2,
            mask=(offs_k[:, None] < nope_dim) & (offs_n[None, :] < lora_rank),
            other=0.0,
        )
        vals = tl.reshape(tl.dot(q_vals[None, :], w_vals), [BLOCK_N]).to(
            tl.float32
        )
        vals = vals / scale
        vals = tl.minimum(tl.maximum(vals, -448.0), 448.0)
        tl.store(
            out_ptr
            + batch * heads * out_dim
            + head * out_dim
            + offs_n,
            vals,
            mask=offs_n < lora_rank,
        )
    else:
        rope_feature = offs_n - lora_rank
        pair = rope_feature // 2
        x_feature = pair * 2
        y_feature = x_feature + 1
        pos = tl.load(positions_ptr + batch)
        cos = tl.load(
            cos_sin_cache_ptr + pos * rope_dim + pair,
            mask=rope_feature < rope_dim,
            other=0.0,
        ).to(tl.float32)
        sin = tl.load(
            cos_sin_cache_ptr + pos * rope_dim + rope_dim // 2 + pair,
            mask=rope_feature < rope_dim,
            other=0.0,
        ).to(tl.float32)
        x_vals = tl.load(
            q_ptr
            + batch * q_stride0
            + head * q_stride1
            + (nope_dim + x_feature) * q_stride2,
            mask=rope_feature < rope_dim,
            other=0.0,
        ).to(tl.float32)
        y_vals = tl.load(
            q_ptr
            + batch * q_stride0
            + head * q_stride1
            + (nope_dim + y_feature) * q_stride2,
            mask=rope_feature < rope_dim,
            other=0.0,
        ).to(tl.float32)
        vals = tl.where(
            (rope_feature % 2) == 0,
            x_vals * cos - y_vals * sin,
            y_vals * cos + x_vals * sin,
        )
        vals = vals / scale
        vals = tl.minimum(tl.maximum(vals, -448.0), 448.0)
        tl.store(
            out_ptr
            + batch * heads * out_dim
            + head * out_dim
            + offs_n,
            vals,
            mask=rope_feature < rope_dim,
        )


@triton.jit
def _mla_k_rope_cache_fp8_kernel(
    k_pe_ptr,  # [num_tokens, rope_dim]
    kv_c_ptr,  # [num_tokens, kv_lora_rank]
    positions_ptr,  # [num_tokens]
    cos_sin_cache_ptr,  # [max_position, rope_dim]
    slot_mapping_ptr,  # [num_tokens]
    kv_cache_fp8_ptr,  # flat fp8 view of [num_blocks, block_size, cache_stride]
    scale_ptr,  # scalar
    num_tokens,
    rope_dim: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    k_pe_stride0,
    k_pe_stride1,
    kv_c_stride0,
    kv_c_stride1,
    cache_block_size: tl.constexpr,
    cache_stride: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    total_dim = kv_lora_rank + rope_dim
    mask = offsets < total_dim
    slot = tl.load(slot_mapping_ptr + token, mask=token < num_tokens, other=-1)
    valid = slot >= 0
    block_idx = slot // cache_block_size
    block_offset = slot - block_idx * cache_block_size
    cache_offset = block_idx * cache_block_size * cache_stride + block_offset * cache_stride
    scale = tl.load(scale_ptr).to(tl.float32)

    is_kv = offsets < kv_lora_rank
    kv_vals = tl.load(
        kv_c_ptr + token * kv_c_stride0 + offsets * kv_c_stride1,
        mask=mask & is_kv & valid,
        other=0.0,
    ).to(tl.float32)

    rope_feature = offsets - kv_lora_rank
    pair = rope_feature // 2
    x_feature = pair * 2
    y_feature = x_feature + 1
    pos = tl.load(positions_ptr + token)
    cos = tl.load(
        cos_sin_cache_ptr + pos * rope_dim + pair,
        mask=mask & ~is_kv,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + pos * rope_dim + rope_dim // 2 + pair,
        mask=mask & ~is_kv,
        other=0.0,
    ).to(tl.float32)
    x_vals = tl.load(
        k_pe_ptr + token * k_pe_stride0 + x_feature * k_pe_stride1,
        mask=mask & ~is_kv,
        other=0.0,
    ).to(tl.float32)
    y_vals = tl.load(
        k_pe_ptr + token * k_pe_stride0 + y_feature * k_pe_stride1,
        mask=mask & ~is_kv,
        other=0.0,
    ).to(tl.float32)
    rope_vals = tl.where(
        (rope_feature % 2) == 0,
        x_vals * cos - y_vals * sin,
        y_vals * cos + x_vals * sin,
    )
    vals = tl.where(is_kv, kv_vals, rope_vals) / scale
    vals = tl.minimum(tl.maximum(vals, -448.0), 448.0)
    tl.store(kv_cache_fp8_ptr + cache_offset + offsets, vals, mask=mask & valid)


@triton.jit
def _indexer_layer_norm_kernel(
    input_ptr,  # [num_tokens, head_dim]
    weight_ptr,  # [head_dim]
    bias_ptr,  # [head_dim]
    out_ptr,  # [num_tokens, head_dim]
    head_dim: tl.constexpr,
    input_stride0,
    input_stride1,
    out_stride0,
    out_stride1,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < head_dim
    x = tl.load(
        input_ptr + row * input_stride0 + cols * input_stride1,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / head_dim
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / head_dim
    inv_std = tl.rsqrt(var + EPS)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = centered * inv_std * weight + bias
    tl.store(
        out_ptr + row * out_stride0 + cols * out_stride1,
        y,
        mask=mask,
    )


@triton.jit
def _scale_index_weights_kernel(
    weights_ptr,  # [num_tokens, n_head]
    scales_ptr,  # [num_tokens, n_head]
    out_ptr,  # [num_tokens, n_head]
    n_heads: tl.constexpr,
    weights_stride0,
    weights_stride1,
    scales_stride0,
    scales_stride1,
    out_stride0,
    out_stride1,
    FACTOR: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_heads
    weights = tl.load(
        weights_ptr + row * weights_stride0 + cols * weights_stride1,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    scales = tl.load(
        scales_ptr + row * scales_stride0 + cols * scales_stride1,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        out_ptr + row * out_stride0 + cols * out_stride1,
        weights * scales * FACTOR,
        mask=mask,
    )


@triton.jit
def _convert_req_index_to_global_index_kernel(
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    valid_count_ptr,  # int32 [num_tokens] - output valid count per row
    prefill_request_id_ptr,  # int32 [num_tokens], -1 for decode, >=0 for prefill
    workspace_starts_ptr,  # int32 [num_prefill_reqs+1] or nullptr
    # shapes (compile-time where possible)
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    HAS_PREFILL: tl.constexpr,
    COUNT_VALID: tl.constexpr,  # whether to count valid indices
    SINGLE_TILE: tl.constexpr,  # whether the row has one column tile
    # strides (in elements)
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    # program_id(0) -> token_id (row)
    # program_id(1) -> tile index along columns
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load request id for this token (no mask: grid is exact)
    req = tl.load(req_id_ptr + token_id)

    # Load token indices for this tile
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)  # int32

    # Only token == -1 should propagate as -1
    is_invalid_tok = tok < 0
    is_prefill = False
    if HAS_PREFILL:
        prefill_req_id = tl.load(prefill_request_id_ptr + token_id)
        is_prefill = prefill_req_id >= 0
    # Compute block id and in-block offset
    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE

    # Guard block_table access
    valid_block = (block_id < max_num_blocks_per_req) & (block_id >= 0)
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    is_invalid_tok |= ~valid_block
    base = tl.load(bt_ptr, mask=valid_block & ~is_prefill, other=0)
    out_val = base * BLOCK_SIZE + inblock_off

    # Override with prefill output if prefill is enabled
    if HAS_PREFILL:
        workspace_start = tl.load(
            workspace_starts_ptr + prefill_req_id, mask=is_prefill, other=0
        )
        prefill_out = workspace_start + tok
        out_val = tl.where(is_prefill, prefill_out, out_val)
    out_val = tl.where(is_invalid_tok, -1, out_val)

    # Store results
    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)

    # Count valid indices in this tile and atomically add to row total
    if COUNT_VALID:
        tile_valid_count = tl.sum((~is_invalid_tok).to(tl.int32))
        if SINGLE_TILE:
            tl.store(valid_count_ptr + token_id, tile_valid_count)
        else:
            tl.atomic_add(valid_count_ptr + token_id, tile_valid_count)


def triton_convert_req_index_to_global_index(
    req_id: torch.Tensor,  # int32 [num_tokens]
    block_table: torch.Tensor,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,  # tile width along columns
    HAS_PREFILL_WORKSPACE: bool = False,
    prefill_workspace_request_ids: torch.Tensor | None = None,
    prefill_workspace_starts: torch.Tensor | None = None,
    return_valid_counts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    out[token_id, indice_id] =
        block_table[req_id[token_id],
            token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Only when token_indices[token_id, indice_id] == -1 do we output -1.
    For safety, we also output -1 if the derived block_id would be
        out-of-bounds.

    When HAS_PREFILL_WORKSPACE is True, prefill tokens are mapped to workspace offsets
    instead of global cache slots. prefill_workspace_request_ids and
    prefill_workspace_starts must be provided.

    prefill_workspace_request_ids: int32 [num_tokens], -1 for decode else
        prefill request index (maps to prefill_workspace_starts)
    prefill_workspace_starts: int32 [num_prefills], 0-indexed workspace
        starts for each prefill request

    When return_valid_counts is True, also returns the count of valid (non -1)
    indices per row, computed during the same kernel pass (no extra overhead).
    """
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, (
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by BLOCK_N ({BLOCK_N})"
    )

    if HAS_PREFILL_WORKSPACE:
        assert prefill_workspace_request_ids is not None
        assert prefill_workspace_starts is not None
        assert prefill_workspace_request_ids.dtype == torch.int32
        assert prefill_workspace_starts.dtype == torch.int32

    num_tokens = req_id.shape[0]
    max_num_blocks_per_req = block_table.shape[1]
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # Ensure contiguous tensors on the same device
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    # Allocate valid count buffer if needed. Multiple tiles need atomics and a
    # zeroed buffer; the BS=1 decode specialization uses one tile and stores.
    valid_counts: torch.Tensor | None = None
    if return_valid_counts:
        if tiles_per_row == 1:
            valid_counts = torch.empty(
                num_tokens, dtype=torch.int32, device=token_indices.device
            )
        else:
            valid_counts = torch.zeros(
                num_tokens, dtype=torch.int32, device=token_indices.device
            )

    # Strides in elements
    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    # Prepare prefill pointers
    if HAS_PREFILL_WORKSPACE:
        assert prefill_workspace_request_ids is not None  # for mypy
        assert prefill_workspace_starts is not None  # for mypy
        assert prefill_workspace_request_ids.is_contiguous()
        assert prefill_workspace_starts.is_contiguous()

    # Exact 2D grid: tokens × column tiles
    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        valid_counts,
        prefill_workspace_request_ids,
        prefill_workspace_starts,
        # shapes / constexprs
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        HAS_PREFILL_WORKSPACE,
        return_valid_counts,
        tiles_per_row == 1,
        # strides
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
        num_warps=8,
    )

    if return_valid_counts:
        assert valid_counts is not None
        return out, valid_counts
    return out
