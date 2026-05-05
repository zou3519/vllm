# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CuTe DSL fused kernel: residual-add + RMSNorm + FP4 (E2M1) quantization.

Single CTA with 256 threads (8 warps) processes all 8192 elements for BS=1.
Cross-warp reduction via shared memory + bar.sync (no spin-wait, no atomics).
FP32->E2M1 conversion via inline PTX: cvt.rn.satfinite.e2m1x2.f32

Replaces the two-kernel Triton approach (_add_variance_kernel +
_norm_fp4_quant_kernel) with a single kernel launch, saving CUDA graph
replay overhead (~1.2μs per eliminated node × 160 nodes = ~192μs/step).
"""
import torch

import cuda.bindings.driver as cuda_driver
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import (
    Float32, Uint8, Uint16, Int32, Pointer,
)
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T

# ============================================================================
# Constants
# ============================================================================
HIDDEN_SIZE = 8192
NUM_THREADS = 256
NUM_WARPS = NUM_THREADS // 32
ELEMS_PER_THREAD = HIDDEN_SIZE // NUM_THREADS  # 32
GROUPS_PER_THREAD = ELEMS_PER_THREAD // 16     # 2
EPS = 1e-5


# ============================================================================
# Inline PTX helpers
# ============================================================================

@dsl_user_op
def ptx_cvt_e2m1x2(even_f32, odd_f32, *, loc=None, ip=None):
    even_ir = Float32(even_f32).ir_value(loc=loc, ip=ip)
    odd_ir = Float32(odd_f32).ir_value(loc=loc, ip=ip)
    result_i16 = llvm.inline_asm(
        T.i16(), [even_ir, odd_ir],
        "{ .reg .b8 tmp; "
        "cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1; "
        "cvt.u16.u8 $0, tmp; }",
        "=h,f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return Uint8(llvm.trunc(T.i8(), result_i16,
                            llvm.IntegerOverflowFlags.none, loc=loc, ip=ip))


@dsl_user_op
def ptx_cvt_f32_to_e4m3(val_f32, *, loc=None, ip=None):
    val_ir = Float32(val_f32).ir_value(loc=loc, ip=ip)
    result_i16 = llvm.inline_asm(
        T.i16(), [val_ir],
        "{ .reg .b16 pair; "
        "cvt.rn.satfinite.e4m3x2.f32 pair, 0f00000000, $1; "
        "mov.b16 $0, pair; }",
        "=h,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return Uint8(llvm.trunc(T.i8(), result_i16,
                            llvm.IntegerOverflowFlags.none, loc=loc, ip=ip))


@dsl_user_op
def ptx_cvt_e4m3_to_f32(val_u8, *, loc=None, ip=None):
    val_ir = Uint8(val_u8).ir_value(loc=loc, ip=ip)
    val_i16 = llvm.zext(T.i16(), val_ir, loc=loc, ip=ip)
    result_f32 = llvm.inline_asm(
        T.f32(), [val_i16],
        "{ .reg .b32 f16x2_packed; .reg .b16 lo_f16; "
        "cvt.rn.f16x2.e4m3x2 f16x2_packed, $1; "
        "mov.b32 {lo_f16, _}, f16x2_packed; "
        "cvt.f32.f16 $0, lo_f16; }",
        "=f,h",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return Float32(result_f32)


@dsl_user_op
def ptx_abs_f32(val, *, loc=None, ip=None):
    val_ir = Float32(val).ir_value(loc=loc, ip=ip)
    result = llvm.inline_asm(
        T.f32(), [val_ir], "abs.f32 $0, $1;", "=f,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return Float32(result)


@dsl_user_op
def load_bf16_as_f32(ptr, *, loc=None, ip=None):
    val_u16 = cute.arch.load(ptr, Uint16)
    result = llvm.inline_asm(
        T.f32(), [Uint16(val_u16).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 tmp; mov.b16 tmp, $1; cvt.f32.bf16 $0, tmp; }",
        "=r,h",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return Float32(result)


@dsl_user_op
def store_f32_as_bf16(ptr, val_f32, *, loc=None, ip=None):
    val_ir = Float32(val_f32).ir_value(loc=loc, ip=ip)
    result_u16 = llvm.inline_asm(
        T.i16(), [val_ir],
        "{ .reg .b16 tmp; cvt.rn.bf16.f32 tmp, $1; mov.b16 $0, tmp; }",
        "=h,r",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    cute.arch.store(ptr, Uint16(result_u16))


# ============================================================================
# Kernel
# ============================================================================

@cute.kernel
def fused_add_rms_norm_fp4_quant_kernel(
    hidden_ptr: Pointer, residual_ptr: Pointer,
    residual_out_ptr: Pointer, weight_ptr: Pointer,
    sf_scale_inv_ptr: Pointer,
    fp4_out_ptr: Pointer, scale_out_ptr: Pointer,
    scale_stride_ptr: Pointer,
):
    tid_x, _, _ = cute.arch.thread_idx()
    warp_id = tid_x // Int32(32)
    lane_id = tid_x % Int32(32)

    smem_ptr = cute.arch.alloc_smem(Float32, NUM_WARPS, alignment=16)
    scale_stride = cute.arch.load(scale_stride_ptr, Int32)
    sf_scale_inv = cute.arch.load(sf_scale_inv_ptr, Float32)

    # Phase 1: residual add + partial sum-of-squares
    thread_base = tid_x * Int32(ELEMS_PER_THREAD)
    partial_sum = Float32(0.0)
    for i in range(ELEMS_PER_THREAD):
        idx = thread_base + Int32(i)
        h_val = load_bf16_as_f32(hidden_ptr + idx)
        r_val = load_bf16_as_f32(residual_ptr + idx)
        s = h_val + r_val
        store_f32_as_bf16(residual_out_ptr + idx, s)
        partial_sum = partial_sum + s * s

    # Phase 2: warp reduction + cross-warp reduction
    warp_sum = cute.arch.warp_reduction_sum(partial_sum)
    if lane_id == Int32(0):
        cute.arch.store(smem_ptr + warp_id, warp_sum)
    cute.arch.sync_threads()

    if tid_x == Int32(0):
        total = Float32(0.0)
        for w in range(NUM_WARPS):
            total = total + cute.arch.load(smem_ptr + Int32(w), Float32)
        cute.arch.store(smem_ptr, total)
    cute.arch.sync_threads()

    # Phase 3: compute rrms
    total_sum = cute.arch.load(smem_ptr, Float32)
    rrms = cute.rsqrt(total_sum / Float32(float(HIDDEN_SIZE)) + Float32(EPS))

    # Phase 4: RMSNorm + FP4 quantization
    for g_off in range(GROUPS_PER_THREAD):
        g = tid_x * Int32(GROUPS_PER_THREAD) + Int32(g_off)
        group_base = g * Int32(16)

        # Find block max
        block_max = Float32(0.0)
        for p in range(8):
            even_idx = group_base + Int32(p * 2)
            odd_idx = even_idx + Int32(1)
            even_normed = load_bf16_as_f32(residual_out_ptr + even_idx) * rrms * load_bf16_as_f32(weight_ptr + even_idx)
            odd_normed = load_bf16_as_f32(residual_out_ptr + odd_idx) * rrms * load_bf16_as_f32(weight_ptr + odd_idx)
            block_max = cute.arch.fmax(block_max, cute.arch.fmax(ptx_abs_f32(even_normed), ptx_abs_f32(odd_normed)))

        # FP8 scale
        sf_fp8_u8 = ptx_cvt_f32_to_e4m3(sf_scale_inv * (block_max / Float32(6.0)))
        quant_scale = sf_scale_inv / cute.arch.fmax(ptx_cvt_e4m3_to_f32(sf_fp8_u8), Float32(1e-30))

        # Quantize pairs to E2M1 and pack
        for p in range(8):
            even_idx = group_base + Int32(p * 2)
            odd_idx = even_idx + Int32(1)
            even_val = load_bf16_as_f32(residual_out_ptr + even_idx) * rrms * load_bf16_as_f32(weight_ptr + even_idx) * quant_scale
            odd_val = load_bf16_as_f32(residual_out_ptr + odd_idx) * rrms * load_bf16_as_f32(weight_ptr + odd_idx) * quant_scale
            cute.arch.store(fp4_out_ptr + g * Int32(8) + Int32(p), ptx_cvt_e2m1x2(even_val, odd_val))

        # Swizzled scale
        cute.arch.store(scale_out_ptr + (g // Int32(4)) * scale_stride + g % Int32(4), sf_fp8_u8)


@cute.jit
def _launch_fused_kernel(
    hidden_t: cute.Tensor, residual_t: cute.Tensor,
    residual_out_t: cute.Tensor, weight_t: cute.Tensor,
    sf_scale_inv_t: cute.Tensor,
    fp4_out_t: cute.Tensor, scale_out_t: cute.Tensor,
    scale_stride_t: cute.Tensor,
    stream: cuda_driver.CUstream = cuda_driver.CUstream(0),
):
    fused_add_rms_norm_fp4_quant_kernel(
        hidden_t.iterator, residual_t.iterator,
        residual_out_t.iterator, weight_t.iterator,
        sf_scale_inv_t.iterator,
        fp4_out_t.iterator, scale_out_t.iterator,
        scale_stride_t.iterator,
    ).launch(grid=[1], block=[NUM_THREADS], stream=stream)


# ============================================================================
# Python wrapper (lazy-compiled, cached)
# ============================================================================

_compiled_fn = None


def cute_fused_add_rms_norm_fp4_quant(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    residual_out: torch.Tensor,
    weight: torch.Tensor,
    sf_scale_inv: torch.Tensor,
    fp4_out: torch.Tensor,
    scale_int32: torch.Tensor,
    scale_stride_tensor: torch.Tensor,
):
    """Drop-in replacement for _add_variance_kernel + _norm_fp4_quant_kernel.

    All tensors must be on the same CUDA device. Compiles on first call.
    """
    global _compiled_fn

    scale_bytes = scale_int32.view(torch.uint8)
    cu_stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    args = (
        from_dlpack(hidden_states.view(-1)),
        from_dlpack(residual.view(-1)),
        from_dlpack(residual_out.view(-1)),
        from_dlpack(weight),
        from_dlpack(sf_scale_inv),
        from_dlpack(fp4_out.view(-1)),
        from_dlpack(scale_bytes.view(-1)),
        from_dlpack(scale_stride_tensor),
        cu_stream,
    )

    if _compiled_fn is None:
        _compiled_fn = cute.compile(_launch_fused_kernel, *args)

    _compiled_fn(*args)
