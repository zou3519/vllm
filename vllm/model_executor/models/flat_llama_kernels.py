# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Flat, function-level forward pass for Llama NVFP4.

The entire model forward is two functions:
  - flat_forward(): embedding → loop of transformer_layer → final norm
  - transformer_layer(): one decoder block with all params passed explicitly

No nn.Module dispatch in the hot path. All weights, buffers, and op handles
are passed as plain tensors / callables.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from vllm import _custom_ops as ops
from vllm._custom_ops import (
    create_fp4_output_tensors,
    cutlass_scaled_fp4_mm,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    NvFp4LinearBackend,
    pad_nvfp4_activation_for_cutlass,
    slice_nvfp4_output,
)
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm


# ---------------------------------------------------------------------------
# Triton fused kernels: residual-add + RMSNorm + FP4 quant (2 launches vs 3)
# Uses inline PTX cvt.rn.satfinite.e2m1x2.f32 for native E2M1 conversion.
# ---------------------------------------------------------------------------


@triton.jit
def _add_variance_kernel(
    hidden_ptr, residual_ptr, residual_out_ptr, variance_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    off = row * N
    _sum_sq = 0.0
    for start in tl.static_range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        h = tl.load(hidden_ptr + off + cols).to(tl.float32)
        r = tl.load(residual_ptr + off + cols).to(tl.float32)
        s = h + r
        tl.store(residual_out_ptr + off + cols, s.to(tl.bfloat16))
        _sum_sq += tl.sum(s * s)
    tl.store(variance_ptr + row, _sum_sq / N)


@triton.jit
def _norm_fp4_quant_kernel(
    residual_ptr, weight_ptr, variance_ptr, sf_scale_ptr,
    fp4_out_ptr, scale_out_ptr,
    N: tl.constexpr, SCALE_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_groups_per_row = N // 16
    row = pid // num_groups_per_row
    g = pid % num_groups_per_row

    variance = tl.load(variance_ptr + row)
    rrms = tl.math.rsqrt(variance + 1e-5)
    sf_scale = tl.load(sf_scale_ptr).to(tl.float32)

    base = row * N + g * 16
    pair_idx = tl.arange(0, 8)
    even_offs = base + pair_idx * 2
    odd_offs = even_offs + 1
    w_even_offs = g * 16 + pair_idx * 2
    w_odd_offs = w_even_offs + 1

    even_res = tl.load(residual_ptr + even_offs).to(tl.float32)
    odd_res = tl.load(residual_ptr + odd_offs).to(tl.float32)
    even_w = tl.load(weight_ptr + w_even_offs).to(tl.float32)
    odd_w = tl.load(weight_ptr + w_odd_offs).to(tl.float32)

    even_normed = even_res * rrms * even_w
    odd_normed = odd_res * rrms * odd_w

    block_max = tl.maximum(
        tl.max(tl.abs(even_normed)), tl.max(tl.abs(odd_normed))
    )
    sf_val = sf_scale * (block_max / 6.0)
    sf_fp8 = sf_val.to(tl.float8e4nv)
    sf_f32 = sf_fp8.to(tl.float32)
    quant_scale = tl.where(sf_f32 > 0.0, sf_scale / sf_f32, 0.0)

    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1;"
        " cvt.u16.u8 $0, tmp; }",
        "=h, r, r",
        [even_normed * quant_scale, odd_normed * quant_scale],
        dtype=tl.int16,
        is_pure=True,
        pack=1,
    )

    fp4_base = row * (N // 2) + g * 8
    tl.store(fp4_out_ptr + fp4_base + pair_idx, packed.to(tl.uint8))

    kTileIdx = g // 4
    innerKIdx = g % 4
    byte_offset = row * SCALE_STRIDE + kTileIdx * SCALE_STRIDE + innerKIdx
    tl.store(scale_out_ptr + byte_offset, sf_fp8.to(tl.uint8, bitcast=True))


# ---------------------------------------------------------------------------
# Single fused kernel: residual-add + RMSNorm + FP4 quant in ONE launch.
# Uses cooperative atomic reduction across programs (≤ SM count to avoid
# deadlock on the spin-wait barrier).
# ---------------------------------------------------------------------------

# Number of programs for the fused kernel. Must be <= number of SMs (152 on
# GB300) to guarantee all programs can run concurrently during the spin-wait.
_FUSED_NUM_PROGRAMS: int = 128


@triton.jit
def _fused_add_rms_norm_fp4_quant_kernel(
    hidden_ptr, residual_ptr, residual_out_ptr,
    weight_ptr, sf_scale_ptr,
    fp4_out_ptr, scale_out_ptr,
    # Atomic reduction workspace
    global_sum_ptr, counter_ptr, ready_ptr,
    N: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
):
    """Fused residual-add + RMSNorm + FP4 quant for BS=1 decode.

    Each of NUM_PROGRAMS programs handles GROUPS_PER_PROGRAM groups of 16
    elements. Phase 1 does residual-add and accumulates a partial
    sum-of-squares. Phase 2 synchronizes via an atomic barrier. Phase 3
    applies RMSNorm + E2M1 FP4 quantization using inline PTX.
    """
    pid = tl.program_id(0)
    row = 0  # BS=1

    # Phase 1: residual add + partial sum-of-squares
    base_group = pid * GROUPS_PER_PROGRAM
    local_sum = tl.zeros([], dtype=tl.float32)

    for g_off in tl.static_range(0, GROUPS_PER_PROGRAM):
        g = base_group + g_off
        elem_base = row * N + g * 16
        idx = tl.arange(0, 16)

        h = tl.load(hidden_ptr + elem_base + idx).to(tl.float32)
        r = tl.load(residual_ptr + elem_base + idx).to(tl.float32)
        s = h + r
        tl.store(residual_out_ptr + elem_base + idx, s.to(tl.bfloat16))
        local_sum += tl.sum(s * s)

    # Phase 2: atomic reduction + spin-wait barrier
    tl.atomic_add(global_sum_ptr, local_sum)
    arrived = tl.atomic_add(counter_ptr, 1)

    # Last program to arrive signals completion
    if arrived == NUM_PROGRAMS - 1:
        tl.store(ready_ptr, 1)

    # All programs spin-wait until ready
    while tl.load(ready_ptr, volatile=True) == 0:
        pass

    # Phase 3: read final variance, apply RMSNorm + FP4 quant
    total_sum = tl.load(global_sum_ptr)
    rrms = tl.math.rsqrt(total_sum / N + 1e-5)
    sf_scale = tl.load(sf_scale_ptr).to(tl.float32)

    for g_off in tl.static_range(0, GROUPS_PER_PROGRAM):
        g = base_group + g_off
        base = row * N + g * 16
        pair_idx = tl.arange(0, 8)
        even_offs = base + pair_idx * 2
        odd_offs = even_offs + 1
        w_even_offs = g * 16 + pair_idx * 2
        w_odd_offs = w_even_offs + 1

        even_res = tl.load(residual_out_ptr + even_offs).to(tl.float32)
        odd_res = tl.load(residual_out_ptr + odd_offs).to(tl.float32)
        even_w = tl.load(weight_ptr + w_even_offs).to(tl.float32)
        odd_w = tl.load(weight_ptr + w_odd_offs).to(tl.float32)

        even_normed = even_res * rrms * even_w
        odd_normed = odd_res * rrms * odd_w

        # FP4 quantization: compute per-group FP8 scale + E2M1 via PTX
        block_max = tl.maximum(
            tl.max(tl.abs(even_normed)), tl.max(tl.abs(odd_normed))
        )
        sf_val = sf_scale * (block_max / 6.0)
        sf_fp8 = sf_val.to(tl.float8e4nv)
        sf_f32 = sf_fp8.to(tl.float32)
        quant_scale = tl.where(sf_f32 > 0.0, sf_scale / sf_f32, 0.0)

        packed = tl.inline_asm_elementwise(
            "{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1;"
            " cvt.u16.u8 $0, tmp; }",
            "=h, r, r",
            [even_normed * quant_scale, odd_normed * quant_scale],
            dtype=tl.int16,
            is_pure=True,
            pack=1,
        )

        fp4_base = row * (N // 2) + g * 8
        tl.store(fp4_out_ptr + fp4_base + pair_idx, packed.to(tl.uint8))

        kTileIdx = g // 4
        innerKIdx = g % 4
        byte_offset = (
            row * SCALE_STRIDE + kTileIdx * SCALE_STRIDE + innerKIdx
        )
        tl.store(
            scale_out_ptr + byte_offset, sf_fp8.to(tl.uint8, bitcast=True)
        )


def triton_fused_add_rms_norm_fp4_quant(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    residual_out: torch.Tensor,
    weight: torch.Tensor,
    sf_scale_inv: torch.Tensor,
    fp4_out: torch.Tensor,
    scale_int32: torch.Tensor,
    global_sum: torch.Tensor,
    counter: torch.Tensor,
    ready: torch.Tensor,
) -> None:
    """Single fused kernel: residual-add + RMSNorm + FP4 quant.

    Replaces the two-kernel sequence of ``_add_variance_kernel`` +
    ``_norm_fp4_quant_kernel``, saving one kernel launch per call site
    (160 launches total across 80 layers × 2 norm+quant points).
    """
    N = hidden_states.shape[-1]
    scale_bytes = scale_int32.view(torch.uint8)
    scale_stride = scale_int32.shape[1] * 4
    num_programs = _FUSED_NUM_PROGRAMS
    groups_per_program = (N // 16) // num_programs

    # Reset atomic workspace before launch
    global_sum.zero_()
    counter.zero_()
    ready.zero_()

    _fused_add_rms_norm_fp4_quant_kernel[(num_programs,)](
        hidden_states, residual, residual_out,
        weight, sf_scale_inv,
        fp4_out, scale_bytes,
        global_sum, counter, ready,
        N=N,
        SCALE_STRIDE=scale_stride,
        NUM_PROGRAMS=num_programs,
        GROUPS_PER_PROGRAM=groups_per_program,
    )


@triton.jit
def _fp4_quant_kernel(
    input_ptr, sf_scale_ptr, fp4_out_ptr, scale_out_ptr,
    N: tl.constexpr, SCALE_STRIDE: tl.constexpr,
):
    """Standalone FP4 quant (no norm). One group of 16 per program.
    Uses PTX E2M1 — faster than C++ scaled_fp4_quant at BS=1."""
    pid = tl.program_id(0)
    num_groups_per_row = N // 16
    row = pid // num_groups_per_row
    g = pid % num_groups_per_row
    sf_scale = tl.load(sf_scale_ptr).to(tl.float32)

    base = row * N + g * 16
    pair_idx = tl.arange(0, 8)
    even = tl.load(input_ptr + base + pair_idx * 2).to(tl.float32)
    odd = tl.load(input_ptr + base + pair_idx * 2 + 1).to(tl.float32)

    block_max = tl.maximum(tl.max(tl.abs(even)), tl.max(tl.abs(odd)))
    sf_val = sf_scale * (block_max / 6.0)
    sf_fp8 = sf_val.to(tl.float8e4nv)
    sf_f32 = sf_fp8.to(tl.float32)
    qs = tl.where(sf_f32 > 0.0, sf_scale / sf_f32, 0.0)

    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1;"
        " cvt.u16.u8 $0, tmp; }",
        "=h, r, r", [even * qs, odd * qs],
        dtype=tl.int16, is_pure=True, pack=1,
    )
    fp4_base = row * (N // 2) + g * 8
    tl.store(fp4_out_ptr + fp4_base + pair_idx, packed.to(tl.uint8))

    kTileIdx = g // 4
    innerKIdx = g % 4
    byte_offset = row * SCALE_STRIDE + kTileIdx * SCALE_STRIDE + innerKIdx
    tl.store(scale_out_ptr + byte_offset, sf_fp8.to(tl.uint8, bitcast=True))


def triton_fp4_quant(x, gs_inv, fp4_out, scale_int32):
    """Triton FP4 quant — drop-in replacement for scaled_fp4_quant.out."""
    M, N = x.shape
    scale_bytes = scale_int32.view(torch.uint8)
    scale_stride = scale_int32.shape[1] * 4
    _fp4_quant_kernel[(M * (N // 16),)](
        x, gs_inv, fp4_out, scale_bytes, N=N, SCALE_STRIDE=scale_stride,
    )


# ---------------------------------------------------------------------------
# Triton: fused RoPE + KV cache write (1 kernel instead of 2)
# ---------------------------------------------------------------------------


@triton.jit
def _fused_rope_kv_kernel(
    q_ptr, k_ptr, v_ptr,
    cos_sin_ptr, positions_ptr,
    kv_cache_ptr, slot_mapping_ptr,
    k_scale_ptr, v_scale_ptr,
    # Strides of the 5D kv_cache tensor (in elements, not bytes).
    # Shape: [num_blocks, 2, ?, ?, head_dim] — dims 2/3 are block_size and
    # num_kv_heads but their order depends on NHD vs HND layout.
    KV_STRIDE_BLOCK,   # stride for dim-0 (num_blocks)
    KV_STRIDE_KV,      # stride for dim-1 (K=0 / V=1)
    KV_STRIDE_BS,      # stride for the block_size dimension
    KV_STRIDE_HEAD,    # stride for the num_kv_heads dimension
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_ROT: tl.constexpr,
    CS_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    KV_CACHE_IS_FP8: tl.constexpr,
    GQA_RATIO: tl.constexpr,
):
    """Fused RoPE(Q,K) + KV cache write. Grid = (NUM_Q_HEADS,).
    Each program handles one Q head's RoPE. Programs where
    head_id < NUM_KV_HEADS also handle K rotation + KV cache write."""
    head_id = tl.program_id(0)
    pos = tl.load(positions_ptr)
    d = tl.arange(0, HALF_ROT)

    cos = tl.load(cos_sin_ptr + pos * CS_STRIDE + d).to(tl.float32)
    sin = tl.load(cos_sin_ptr + pos * CS_STRIDE + HALF_ROT + d).to(tl.float32)

    # RoPE on this Q head
    q_base = head_id * HEAD_DIM
    x1 = tl.load(q_ptr + q_base + d).to(tl.float32)
    x2 = tl.load(q_ptr + q_base + HALF_ROT + d).to(tl.float32)
    tl.store(q_ptr + q_base + d, (x1 * cos - x2 * sin).to(tl.bfloat16))
    tl.store(q_ptr + q_base + HALF_ROT + d, (x2 * cos + x1 * sin).to(tl.bfloat16))

    # First GQA_RATIO programs also handle K rotation + KV cache write
    # (one KV head per GQA_RATIO Q heads)
    if head_id % GQA_RATIO == 0 and head_id // GQA_RATIO < NUM_KV_HEADS:
        kv_h = head_id // GQA_RATIO
        hd = kv_h * HEAD_DIM
        k1 = tl.load(k_ptr + hd + d).to(tl.float32)
        k2 = tl.load(k_ptr + hd + HALF_ROT + d).to(tl.float32)
        k_rot1 = (k1 * cos - k2 * sin).to(tl.bfloat16)
        k_rot2 = (k2 * cos + k1 * sin).to(tl.bfloat16)
        tl.store(k_ptr + hd + d, k_rot1)
        tl.store(k_ptr + hd + HALF_ROT + d, k_rot2)

        # Write to KV cache using actual strides (supports NHD & HND layouts)
        slot = tl.load(slot_mapping_ptr)
        block_idx = slot // BLOCK_SIZE
        block_off = slot % BLOCK_SIZE
        k_base = (block_idx * KV_STRIDE_BLOCK
                  + block_off * KV_STRIDE_BS
                  + kv_h * KV_STRIDE_HEAD)
        v_base = (block_idx * KV_STRIDE_BLOCK
                  + KV_STRIDE_KV
                  + block_off * KV_STRIDE_BS
                  + kv_h * KV_STRIDE_HEAD)
        dd = tl.arange(0, HEAD_DIM)

        if KV_CACHE_IS_FP8:
            # FP8 convention: fp8_val = bf16_val / scale (matches C++ cache
            # kernels which call scaled_convert with value/scale).
            ks = tl.load(k_scale_ptr).to(tl.float32)
            vs = tl.load(v_scale_ptr).to(tl.float32)
            k_full = tl.load(k_ptr + hd + dd).to(tl.float32)
            tl.store(kv_cache_ptr + k_base + dd,
                     (k_full / ks).to(tl.float8e4nv))
            v_vals = tl.load(v_ptr + hd + dd).to(tl.float32)
            tl.store(kv_cache_ptr + v_base + dd,
                     (v_vals / vs).to(tl.float8e4nv))
        else:
            tl.store(kv_cache_ptr + k_base + d, k_rot1)
            tl.store(kv_cache_ptr + k_base + HALF_ROT + d, k_rot2)
            v_vals = tl.load(v_ptr + hd + dd)
            tl.store(kv_cache_ptr + v_base + dd, v_vals)


# ---------------------------------------------------------------------------
# Data structures — hold extracted params for one NVFP4 linear projection
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class NvFp4Proj:
    """All tensors needed to run one NVFP4 linear projection."""
    weight: torch.Tensor           # [out, in/2] uint8
    weight_scale: torch.Tensor     # fp8 block scales (swizzled)
    alpha: torch.Tensor            # input_global_scale * weight_global_scale
    input_scale_inv: torch.Tensor  # 1 / input_global_scale
    weights_padding: int           # K-dimension padding bytes
    output_size: int               # unpadded output dim


@dataclass(slots=True)
class SharedPrefillBuffers:
    """Pre-allocated FP4 buffers for prefill (M>1) to avoid per-call allocs.

    Unlike SharedDecodeBuffers (BS=1), these are sized for the prefill batch.
    They are lazily created on first prefill and re-created if M changes.
    """
    m: int  # current batch size these buffers are allocated for

    # FP4 quant outputs for each projection (reused across layers)
    qkv_fp4: torch.Tensor
    qkv_scale: torch.Tensor
    o_fp4: torch.Tensor
    o_scale: torch.Tensor
    gu_fp4: torch.Tensor
    gu_scale: torch.Tensor
    down_fp4: torch.Tensor
    down_scale: torch.Tensor

    @staticmethod
    def create(
        m: int,
        hidden_size: int,
        q_size: int,
        intermediate_size: int,
        device: torch.device,
    ) -> "SharedPrefillBuffers":
        qkv_fp4, qkv_sc = create_fp4_output_tensors(
            m, hidden_size, device, True)
        o_fp4, o_sc = create_fp4_output_tensors(m, q_size, device, True)
        gu_fp4, gu_sc = create_fp4_output_tensors(
            m, hidden_size, device, True)
        d_fp4, d_sc = create_fp4_output_tensors(
            m, intermediate_size, device, True)
        return SharedPrefillBuffers(
            m=m,
            qkv_fp4=qkv_fp4, qkv_scale=qkv_sc,
            o_fp4=o_fp4, o_scale=o_sc,
            gu_fp4=gu_fp4, gu_scale=gu_sc,
            down_fp4=d_fp4, down_scale=d_sc,
        )


@dataclass(slots=True)
class SharedDecodeBuffers:
    """One set of pre-allocated FP4 buffers shared across all layers."""
    qkv_fp4: torch.Tensor
    qkv_scale: torch.Tensor
    o_fp4: torch.Tensor
    o_scale: torch.Tensor
    gu_fp4: torch.Tensor
    gu_scale: torch.Tensor
    down_fp4: torch.Tensor
    down_scale: torch.Tensor

    # Intermediate buffers for fused norm+quant
    residual_buf: torch.Tensor
    variance: torch.Tensor

    # Atomic workspace for single-kernel fused add+rms_norm+fp4_quant
    global_sum: torch.Tensor   # f32 accumulator for sum-of-squares
    counter: torch.Tensor      # i32 arrival counter
    ready: torch.Tensor        # i32 ready flag

    # CuTe DSL kernel param
    scale_stride_tensor: torch.Tensor  # i32[1] for swizzled scale stride

    @staticmethod
    def create(
        hidden_size: int,
        q_size: int,
        intermediate_size: int,
        device: torch.device,
    ) -> "SharedDecodeBuffers":
        qkv_fp4, qkv_sc = create_fp4_output_tensors(1, hidden_size, device, True)
        o_fp4, o_sc = create_fp4_output_tensors(1, q_size, device, True)
        gu_fp4, gu_sc = create_fp4_output_tensors(1, hidden_size, device, True)
        d_fp4, d_sc = create_fp4_output_tensors(1, intermediate_size, device, True)
        scale_stride_val = qkv_sc.view(torch.uint8).shape[-1]
        return SharedDecodeBuffers(
            qkv_fp4, qkv_sc, o_fp4, o_sc, gu_fp4, gu_sc, d_fp4, d_sc,
            residual_buf=torch.empty(1, hidden_size, dtype=torch.bfloat16, device=device),
            variance=torch.empty(1, dtype=torch.float32, device=device),
            global_sum=torch.zeros(1, dtype=torch.float32, device=device),
            counter=torch.zeros(1, dtype=torch.int32, device=device),
            ready=torch.zeros(1, dtype=torch.int32, device=device),
            scale_stride_tensor=torch.tensor([scale_stride_val], dtype=torch.int32, device=device),
        )


# ---------------------------------------------------------------------------
# Core: nvfp4_gemm — GEMM with pre-quantized FP4 input
# ---------------------------------------------------------------------------


def nvfp4_gemm(
    x_fp4: torch.Tensor,
    x_scale: torch.Tensor,
    proj: NvFp4Proj,
    backend: NvFp4LinearBackend,
) -> torch.Tensor:
    xp = pad_nvfp4_activation_for_cutlass(x_fp4, proj.weights_padding)
    args = (xp, proj.weight, x_scale, proj.weight_scale, proj.alpha, torch.bfloat16)
    if backend.value.startswith("flashinfer-"):
        out = flashinfer_scaled_fp4_mm(*args, backend=backend.value[len("flashinfer-"):])
    elif backend == NvFp4LinearBackend.FBGEMM:
        out = torch.ops.fbgemm.f4f4bf16(
            xp, proj.weight, x_scale.view(-1).view(torch.uint8),
            proj.weight_scale, proj.alpha, use_mx=False,
        ).to(torch.bfloat16)
    else:
        out = cutlass_scaled_fp4_mm(*args)
    return slice_nvfp4_output(out, proj.output_size)


def nvfp4_linear(x: torch.Tensor, proj: NvFp4Proj, backend) -> torch.Tensor:
    """Full NVFP4 linear: quantize input + GEMM. Works for any batch size."""
    from vllm._custom_ops import scaled_fp4_quant
    x_fp4, x_scale = scaled_fp4_quant(
        x, proj.input_scale_inv, is_sf_swizzled_layout=True,
        backend=backend.value,
    )
    return nvfp4_gemm(x_fp4, x_scale, proj, backend)


def _nvfp4_quant_and_gemm(
    x: torch.Tensor,
    proj: NvFp4Proj,
    fp4_buf: torch.Tensor,
    scale_buf: torch.Tensor,
    backend: NvFp4LinearBackend,
) -> torch.Tensor:
    """FP4 quantize into pre-allocated buffers, then GEMM.

    Avoids the tensor allocation in scaled_fp4_quant by writing directly
    into pre-allocated fp4_buf / scale_buf via the .out variant.
    """
    torch.ops._C.scaled_fp4_quant.out(
        x, proj.input_scale_inv, True,
        output=fp4_buf, output_scale=scale_buf,
    )
    return nvfp4_gemm(
        fp4_buf, scale_buf.view(torch.float8_e4m3fn), proj, backend,
    )


# Try to import CuTe DSL kernel; fall back to Triton two-kernel approach
_use_cute_norm_quant = False
# CuTe DSL kernel is correct and CUDA-graph compatible (stream fix applied),
# but from_dlpack() costs 43μs per call (8 tensors × 5.4μs), which adds
# 6.9ms per decode step (160 calls). This overhead negates the kernel speedup.
# Need to either cache CuTe tensors or pass raw pointers to fix.


def _fused_norm_quant(
    hidden_states, residual, bufs, ln_weight,
    input_scale_inv, fp4_out, scale_out,
    N, M, scale_stride,
):
    """Dispatch fused add+RMSNorm+FP4 quant to CuTe DSL (1 kernel) or Triton (2 kernels)."""
    if _use_cute_norm_quant and N == 8192 and M == 1:
        cute_fused_add_rms_norm_fp4_quant(
            hidden_states, residual, bufs.residual_buf,
            ln_weight, input_scale_inv,
            fp4_out, scale_out,
            bufs.scale_stride_tensor,
        )
    else:
        _add_variance_kernel[(M,)](
            hidden_states, residual, bufs.residual_buf, bufs.variance,
            N=N, BLOCK=min(N, 4096),
        )
        _norm_fp4_quant_kernel[(M * (N // 16),)](
            bufs.residual_buf, ln_weight, bufs.variance,
            input_scale_inv,
            fp4_out, scale_out.view(torch.uint8),
            N=N, SCALE_STRIDE=scale_stride,
        )


def transformer_layer_general(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    input_ln_w: torch.Tensor,
    post_attn_ln_w: torch.Tensor,
    eps: float,
    qkv: NvFp4Proj, o: NvFp4Proj,
    gate_up: NvFp4Proj, down: NvFp4Proj,
    rotary_emb, attn,
    q_size: int, kv_size: int,
    backend,
    prefill_bufs: SharedPrefillBuffers | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """General-purpose transformer layer (any batch size). Uses flat params
    with C++ ops — no nn.Module dispatch overhead.

    When prefill_bufs is provided, uses pre-allocated FP4 output buffers
    and the fused silu_and_mul_nvfp4_quant kernel to reduce allocation
    overhead and kernel launch count.
    """
    if residual is None:
        residual = hidden_states
        from vllm.model_executor.layers.layernorm import ir
        hidden_states = ir.ops.rms_norm(hidden_states, input_ln_w, eps)
    else:
        ops.fused_add_rms_norm(hidden_states, residual, input_ln_w, eps)

    if prefill_bufs is not None:
        # --- Optimized prefill path with pre-allocated buffers ---

        # QKV: quant into pre-alloc buffers + GEMM
        qkv_out = _nvfp4_quant_and_gemm(
            hidden_states, qkv,
            prefill_bufs.qkv_fp4, prefill_bufs.qkv_scale, backend)
        q, k, v = qkv_out.split([q_size, kv_size, kv_size], dim=-1)
        q, k = rotary_emb(positions, q, k)
        attn_output = attn(q, k, v)

        # O proj: quant into pre-alloc buffers + GEMM
        hidden_states = _nvfp4_quant_and_gemm(
            attn_output, o,
            prefill_bufs.o_fp4, prefill_bufs.o_scale, backend)

        ops.fused_add_rms_norm(hidden_states, residual, post_attn_ln_w, eps)

        # Gate+Up: quant into pre-alloc buffers + GEMM
        gate_up_out = _nvfp4_quant_and_gemm(
            hidden_states, gate_up,
            prefill_bufs.gu_fp4, prefill_bufs.gu_scale, backend)

        # Fused SiLU+mul+FP4 quant (1 kernel instead of silu_and_mul + quant)
        torch.ops._C.silu_and_mul_nvfp4_quant(
            prefill_bufs.down_fp4, prefill_bufs.down_scale,
            gate_up_out, down.input_scale_inv,
        )
        hidden_states = nvfp4_gemm(
            prefill_bufs.down_fp4,
            prefill_bufs.down_scale.view(torch.float8_e4m3fn),
            down, backend,
        )
    else:
        # --- Original path (fallback) ---
        qkv_out = nvfp4_linear(hidden_states, qkv, backend)
        q, k, v = qkv_out.split([q_size, kv_size, kv_size], dim=-1)
        q, k = rotary_emb(positions, q, k)
        attn_output = attn(q, k, v)
        hidden_states = nvfp4_linear(attn_output, o, backend)

        ops.fused_add_rms_norm(hidden_states, residual, post_attn_ln_w, eps)

        gate_up_out = nvfp4_linear(hidden_states, gate_up, backend)
        d = gate_up_out.shape[-1] // 2
        silu_out = torch.empty(
            gate_up_out.shape[:-1] + (d,),
            dtype=gate_up_out.dtype, device=gate_up_out.device,
        )
        torch.ops._C.silu_and_mul(silu_out, gate_up_out)
        hidden_states = nvfp4_linear(silu_out, down, backend)

    return hidden_states, residual


# ---------------------------------------------------------------------------
# Core: transformer_layer — one decoder block, all params explicit
# ---------------------------------------------------------------------------


def transformer_layer(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    # --- LayerNorm weights ---
    input_ln_w: torch.Tensor,
    post_attn_ln_w: torch.Tensor,
    eps: float,
    # --- NVFP4 projections ---
    qkv: NvFp4Proj,
    o: NvFp4Proj,
    gate_up: NvFp4Proj,
    down: NvFp4Proj,
    # --- Attention + RoPE (opaque callables) ---
    rotary_emb,      # callable(positions, q, k) -> (q, k)
    attn,            # callable(q, k, v) -> output
    # --- Sizes ---
    q_size: int,
    kv_size: int,
    # --- Pre-allocated FP4 buffers ---
    bufs: SharedDecodeBuffers,
    # --- Backend ---
    backend: NvFp4LinearBackend,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One decoder layer. Every input is an explicit parameter — no hidden state.

    With single-kernel fused norm+quant: 9 kernel launches (down from 13):
      1. _fused_add_rms_norm_fp4_quant (residual add + norm + FP4 quant)
      2. GEMM (QKV)
      3. Fused RoPE + KV cache write
      4. Attention
      5. scaled_fp4_quant (O input — not fused, different input size)
      6. GEMM (O)
      7. _fused_add_rms_norm_fp4_quant (post-attention norm + quant)
      8. GEMM (gate_up)
      9. silu_and_mul_nvfp4_quant (fused)
      10. GEMM (down)
    = 9 unique kernel launches (steps 1 and 7 each replace 3 separate ops)
    """
    N = hidden_states.shape[-1]
    M = hidden_states.shape[0]
    scale_stride = bufs.qkv_scale.view(torch.uint8).shape[-1]

    # 1. Fused pre-attention: residual-add + RMSNorm + FP4 quant (1 kernel)
    if residual is None:
        residual = hidden_states
        from vllm.model_executor.layers.layernorm import ir
        hidden_states = ir.ops.rms_norm(hidden_states, input_ln_w, eps)
        torch.ops._C.scaled_fp4_quant.out(
            hidden_states, qkv.input_scale_inv, True,
            output=bufs.qkv_fp4, output_scale=bufs.qkv_scale,
        )
    else:
        _fused_norm_quant(
            hidden_states, residual, bufs, input_ln_w,
            qkv.input_scale_inv, bufs.qkv_fp4, bufs.qkv_scale,
            N, M, scale_stride,
        )
        residual = bufs.residual_buf

    # 3. QKV GEMM
    qkv_out = nvfp4_gemm(
        bufs.qkv_fp4, bufs.qkv_scale.view(torch.float8_e4m3fn), qkv, backend,
    )

    # 3. Split Q/K/V
    q, k, v = qkv_out.split([q_size, kv_size, kv_size], dim=-1)

    # 4. Fused RoPE + KV cache write (1 kernel instead of 2)
    from vllm.forward_context import get_forward_context, is_forward_context_available
    kv_cache = attn.kv_cache
    can_fuse = (
        is_forward_context_available()
        and kv_cache.numel() > 0
        and isinstance(getattr(get_forward_context(), "slot_mapping", None), dict)
        and attn.layer_name in get_forward_context().slot_mapping
    )

    if can_fuse:
        fwd_ctx = get_forward_context()
        slot_mapping = fwd_ctx.slot_mapping[attn.layer_name]
        cos_sin_cache = rotary_emb.cos_sin_cache
        head_dim = attn.head_size
        is_fp8 = kv_cache.dtype == torch.float8_e4m3fn
        gqa = attn.num_heads // attn.num_kv_heads

        # Map strides for NHD [B,2,BS,H,D] vs HND [B,2,H,BS,D]
        strides = kv_cache.stride()
        nkv = attn.num_kv_heads
        if kv_cache.shape[2] == nkv:
            # HND layout: dim2=heads, dim3=block_size
            head_stride, bs_stride = strides[2], strides[3]
            block_size = kv_cache.shape[3]
        else:
            # NHD layout: dim2=block_size, dim3=heads
            bs_stride, head_stride = strides[2], strides[3]
            block_size = kv_cache.shape[2]

        _fused_rope_kv_kernel[(attn.num_heads,)](
            q, k, v,
            cos_sin_cache, positions,
            kv_cache, slot_mapping,
            attn._k_scale, attn._v_scale,
            KV_STRIDE_BLOCK=strides[0],
            KV_STRIDE_KV=strides[1],
            KV_STRIDE_BS=bs_stride,
            KV_STRIDE_HEAD=head_stride,
            NUM_Q_HEADS=attn.num_heads, NUM_KV_HEADS=attn.num_kv_heads,
            HEAD_DIM=head_dim, HALF_ROT=head_dim // 2,
            CS_STRIDE=cos_sin_cache.stride(0),
            BLOCK_SIZE=block_size,
            KV_CACHE_IS_FP8=is_fp8,
            GQA_RATIO=gqa,
        )
        # Attention reads from cache (KV write already done by fused kernel;
        # kv_sharing_target_layer_name was set permanently in extract_all_layer_params)
        attn_output = attn(q, k, v)
    else:
        q, k = rotary_emb(positions, q, k)
        attn_output = attn(q, k, v)

    # 5. O projection (Triton FP4 quant — faster than C++ at BS=1)
    triton_fp4_quant(attn_output, o.input_scale_inv, bufs.o_fp4, bufs.o_scale)
    hidden_states = nvfp4_gemm(
        bufs.o_fp4, bufs.o_scale.view(torch.float8_e4m3fn), o, backend,
    )

    # 7. Post-attention norm + FP4 quant
    _fused_norm_quant(
        hidden_states, residual, bufs, post_attn_ln_w,
        gate_up.input_scale_inv, bufs.gu_fp4, bufs.gu_scale,
        N, M, scale_stride,
    )
    residual = bufs.residual_buf

    # 8. Gate+Up GEMM
    gate_up_out = nvfp4_gemm(
        bufs.gu_fp4, bufs.gu_scale.view(torch.float8_e4m3fn), gate_up, backend,
    )

    # 8+9. Fused SiLU+mul+FP4 quant → down projection
    torch.ops._C.silu_and_mul_nvfp4_quant(
        bufs.down_fp4, bufs.down_scale, gate_up_out, down.input_scale_inv,
    )
    hidden_states = nvfp4_gemm(
        bufs.down_fp4, bufs.down_scale.view(torch.float8_e4m3fn), down, backend,
    )

    return hidden_states, residual


# ---------------------------------------------------------------------------
# Core: flat_forward — the entire model as one function
# ---------------------------------------------------------------------------


def flat_forward(
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    # --- Embedding ---
    embed_fn,           # callable(input_ids) -> hidden_states, or None
    # --- Per-layer params (lists of length num_layers) ---
    input_ln_weights: list[torch.Tensor],
    post_attn_ln_weights: list[torch.Tensor],
    eps: float,
    qkv_projs: list[NvFp4Proj],
    o_projs: list[NvFp4Proj],
    gate_up_projs: list[NvFp4Proj],
    down_projs: list[NvFp4Proj],
    rotary_embs: list,   # list of callable
    attns: list,          # list of callable (Attention layers)
    q_size: int,
    kv_size: int,
    # --- Final norm ---
    final_norm_w: torch.Tensor,
    # --- Shared buffers ---
    bufs: SharedDecodeBuffers,
    # --- Backend ---
    backend: NvFp4LinearBackend,
    # --- Layer range ---
    start_layer: int,
    end_layer: int,
    # --- Optional: pre-computed hidden states (skip embedding) ---
    hidden_states_in: torch.Tensor | None = None,
    # --- Prefill buffer cache (mutable list with single element) ---
    prefill_bufs_cache: list | None = None,
    hidden_size: int = 0,
    intermediate_size: int = 0,
) -> torch.Tensor:
    """Full model forward: embedding → N × transformer_layer → final norm.

    Every parameter is passed explicitly. No nn.Module attribute access.
    """
    if hidden_states_in is not None:
        hidden_states = hidden_states_in
    else:
        hidden_states = embed_fn(input_ids)
    residual = None
    num_tokens = hidden_states.shape[0]

    if num_tokens == 1 and bufs is not None:
        # BS=1 decode: use fused Triton kernels + pre-allocated buffers.
        # Mark attention layers to skip KV cache write (fused kernel handles it).
        # Must be set BEFORE graph capture and stay set during replay.
        for i in range(start_layer, end_layer):
            attns[i].kv_sharing_target_layer_name = attns[i].layer_name
        for i in range(start_layer, end_layer):
            hidden_states, residual = transformer_layer(
                positions, hidden_states, residual,
                input_ln_weights[i], post_attn_ln_weights[i], eps,
                qkv_projs[i], o_projs[i], gate_up_projs[i], down_projs[i],
                rotary_embs[i], attns[i],
                q_size, kv_size,
                bufs, backend,
            )
    else:
        # General path: direct C++ ops, no nn.Module dispatch.
        # Ensure attention layers DO write to KV cache (clear the flag).
        for i in range(start_layer, end_layer):
            attns[i].kv_sharing_target_layer_name = None

        # Get or create pre-allocated prefill buffers
        prefill_bufs: SharedPrefillBuffers | None = None
        if prefill_bufs_cache is not None and hidden_size > 0:
            if (len(prefill_bufs_cache) == 0
                    or prefill_bufs_cache[0] is None
                    or prefill_bufs_cache[0].m != num_tokens):
                pb = SharedPrefillBuffers.create(
                    num_tokens, hidden_size, q_size,
                    intermediate_size, hidden_states.device,
                )
                if len(prefill_bufs_cache) == 0:
                    prefill_bufs_cache.append(pb)
                else:
                    prefill_bufs_cache[0] = pb
            prefill_bufs = prefill_bufs_cache[0]

        for i in range(start_layer, end_layer):
            hidden_states, residual = transformer_layer_general(
                positions, hidden_states, residual,
                input_ln_weights[i], post_attn_ln_weights[i], eps,
                qkv_projs[i], o_projs[i], gate_up_projs[i], down_projs[i],
                rotary_embs[i], attns[i],
                q_size, kv_size,
                backend,
                prefill_bufs=prefill_bufs,
            )

    # Final norm
    ops.fused_add_rms_norm(hidden_states, residual, final_norm_w, eps)
    return hidden_states


# ---------------------------------------------------------------------------
# Param extraction — pull all weights out of nn.Module into flat lists
# ---------------------------------------------------------------------------


def extract_nvfp4_proj(linear_module: torch.nn.Module) -> NvFp4Proj:
    """Extract NVFP4 params from a vLLM ColumnParallelLinear/RowParallelLinear."""
    return NvFp4Proj(
        weight=linear_module.weight,
        weight_scale=linear_module.weight_scale,
        alpha=linear_module.alpha,
        input_scale_inv=linear_module.input_global_scale_inv,
        weights_padding=getattr(linear_module, "weights_padding_cols", 0),
        output_size=linear_module.output_size_per_partition,
    )


def extract_all_layer_params(layers, start_layer, end_layer):
    """Extract flat param lists from nn.Module decoder layers."""
    input_ln_weights = []
    post_attn_ln_weights = []
    qkv_projs = []
    o_projs = []
    gate_up_projs = []
    down_projs = []
    rotary_embs = []
    attns = []

    for i in range(start_layer, end_layer):
        layer = layers[i]
        input_ln_weights.append(layer.input_layernorm.weight.data)
        post_attn_ln_weights.append(layer.post_attention_layernorm.weight.data)
        qkv_projs.append(extract_nvfp4_proj(layer.self_attn.qkv_proj))
        o_projs.append(extract_nvfp4_proj(layer.self_attn.o_proj))
        gate_up_projs.append(extract_nvfp4_proj(layer.mlp.gate_up_proj))
        down_projs.append(extract_nvfp4_proj(layer.mlp.down_proj))
        rotary_embs.append(layer.self_attn.rotary_emb)
        attn_layer = layer.self_attn.attn
        attns.append(attn_layer)

    return (
        input_ln_weights, post_attn_ln_weights,
        qkv_projs, o_projs, gate_up_projs, down_projs,
        rotary_embs, attns,
    )
