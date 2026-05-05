"""Triton NVFP4 GEMV for down projection — benchmark vs CUDA GEMV."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fp4_group_dot(w0, x0, w1, x1):
    """16-element FP4 dot product via PTX.

    Takes 2 int32 weight + 2 int32 input (each int32 = 4 bytes = 8 FP4).
    Returns float32 sum of all 16 element-wise products.
    """
    return tl.inline_asm_elementwise(
        "{ .reg .b8 a0,a1,a2,a3,b0,b1,b2,b3;"
        " .reg .b32 ha,hb,acc;"
        " .reg .f16 lo,hi; .reg .f32 flo,fhi;"
        " mov.b32 {a0,a1,a2,a3},$1; mov.b32 {b0,b1,b2,b3},$2;"
        " cvt.rn.f16x2.e2m1x2 ha,a0; cvt.rn.f16x2.e2m1x2 hb,b0; mul.rn.f16x2 acc,ha,hb;"
        " cvt.rn.f16x2.e2m1x2 ha,a1; cvt.rn.f16x2.e2m1x2 hb,b1; fma.rn.f16x2 acc,ha,hb,acc;"
        " cvt.rn.f16x2.e2m1x2 ha,a2; cvt.rn.f16x2.e2m1x2 hb,b2; fma.rn.f16x2 acc,ha,hb,acc;"
        " cvt.rn.f16x2.e2m1x2 ha,a3; cvt.rn.f16x2.e2m1x2 hb,b3; fma.rn.f16x2 acc,ha,hb,acc;"
        " mov.b32 {a0,a1,a2,a3},$3; mov.b32 {b0,b1,b2,b3},$4;"
        " cvt.rn.f16x2.e2m1x2 ha,a0; cvt.rn.f16x2.e2m1x2 hb,b0; fma.rn.f16x2 acc,ha,hb,acc;"
        " cvt.rn.f16x2.e2m1x2 ha,a1; cvt.rn.f16x2.e2m1x2 hb,b1; fma.rn.f16x2 acc,ha,hb,acc;"
        " cvt.rn.f16x2.e2m1x2 ha,a2; cvt.rn.f16x2.e2m1x2 hb,b2; fma.rn.f16x2 acc,ha,hb,acc;"
        " cvt.rn.f16x2.e2m1x2 ha,a3; cvt.rn.f16x2.e2m1x2 hb,b3; fma.rn.f16x2 acc,ha,hb,acc;"
        " mov.b32 {lo,hi},acc;"
        " cvt.f32.f16 flo,lo; cvt.f32.f16 fhi,hi;"
        " add.f32 $0,flo,fhi; }",
        "=f, r, r, r, r",
        [w0, x0, w1, x1],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def triton_nvfp4_gemv_kernel(
    W_ptr,   # [M, K_I32] int32 (weight FP4, viewed as int32)
    X_ptr,   # [K_I32] int32 (input FP4, viewed as int32)
    WS_ptr,  # [M, K_GROUPS] float8 (weight scales, row-major)
    XS_ptr,  # [K_GROUPS] float8 (input scales, row-major)
    Y_ptr,   # [M] bfloat16 output
    alpha,
    K_I32: tl.constexpr,
    K_GROUPS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    row = tl.program_id(0)
    acc = 0.0

    for g_start in range(0, K_GROUPS, BLOCK_GROUPS):
        g = g_start + tl.arange(0, BLOCK_GROUPS)

        w0 = tl.load(W_ptr + row * K_I32 + g * 2,
                      eviction_policy='evict_first')
        w1 = tl.load(W_ptr + row * K_I32 + g * 2 + 1,
                      eviction_policy='evict_first')
        x0 = tl.load(X_ptr + g * 2, eviction_policy='evict_last')
        x1 = tl.load(X_ptr + g * 2 + 1, eviction_policy='evict_last')

        dots = _fp4_group_dot(w0, x0, w1, x1)

        ws = tl.load(WS_ptr + row * K_GROUPS + g,
                     eviction_policy='evict_first').to(tl.float32)
        xs = tl.load(XS_ptr + g,
                     eviction_policy='evict_last').to(tl.float32)

        acc += tl.sum(dots * ws * xs)

    tl.store(Y_ptr + row, (acc * alpha).to(tl.bfloat16))


def triton_nvfp4_gemv(
    weight: torch.Tensor,       # [M, K_fp4x2] uint8
    input_fp4: torch.Tensor,    # [K_fp4x2] uint8
    weight_scale: torch.Tensor, # [M, K_fp4x2/8] uint8 (row-major FP8)
    input_scale: torch.Tensor,  # [K_fp4x2/8] uint8 (row-major FP8)
    output: torch.Tensor,       # [M] bfloat16
    alpha: float,
):
    M = weight.shape[0]
    K_bytes = weight.shape[1]
    K_i32 = K_bytes // 4
    K_groups = K_bytes // 8

    W_i32 = weight.view(torch.int32)
    X_i32 = input_fp4.view(torch.int32)
    WS_f8 = weight_scale.view(torch.float8_e4m3fn)
    XS_f8 = input_scale.view(torch.float8_e4m3fn)

    BLOCK_GROUPS = min(K_groups, 256)

    triton_nvfp4_gemv_kernel[(M,)](
        W_i32, X_i32,
        WS_f8, XS_f8,
        output, alpha,
        K_I32=K_i32, K_GROUPS=K_groups,
        BLOCK_GROUPS=BLOCK_GROUPS,
        num_warps=4,
    )


if __name__ == "__main__":
    M, K = 8192, 14336
    K_sf = K // 8
    K_i32 = K // 4
    K_groups = K // 8

    A   = torch.randint(0, 256, (M, K), dtype=torch.uint8, device="cuda")
    B   = torch.randint(0, 256, (K,),   dtype=torch.uint8, device="cuda")
    SFA = torch.randint(1, 4,   (M, K_sf), dtype=torch.uint8, device="cuda")
    SFB = torch.randint(1, 4,   (K_sf,),   dtype=torch.uint8, device="cuda")
    C   = torch.zeros(M, dtype=torch.bfloat16, device="cuda")

    total_bytes = M * K + M * K_sf
    W_i32 = A.view(torch.int32)
    X_i32 = B.view(torch.int32)
    WS_f8 = SFA.view(torch.float8_e4m3fn)
    XS_f8 = SFB.view(torch.float8_e4m3fn)

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)

    print(f"Down proj GEMV: M={M}, K_fp4x2={K}, groups={K_groups}")
    print(f"{'Config':<40} {'μs':>7} {'TB/s':>7}")
    print("─" * 60)

    for num_warps in [1, 2, 4, 8]:
        for block_groups in [64, 128, 256, 448]:
            if block_groups > K_groups:
                continue
            if K_groups % block_groups != 0:
                continue
            label = f"warps={num_warps} block_groups={block_groups}"
            try:
                triton_nvfp4_gemv_kernel[(M,)](
                    W_i32, X_i32, WS_f8, XS_f8, C, 1.0,
                    K_I32=K_i32, K_GROUPS=K_groups,
                    BLOCK_GROUPS=block_groups, num_warps=num_warps)
                for _ in range(100):
                    triton_nvfp4_gemv_kernel[(M,)](
                        W_i32, X_i32, WS_f8, XS_f8, C, 1.0,
                        K_I32=K_i32, K_GROUPS=K_groups,
                        BLOCK_GROUPS=block_groups, num_warps=num_warps)
                torch.cuda.synchronize()
                s.record()
                for _ in range(500):
                    triton_nvfp4_gemv_kernel[(M,)](
                        W_i32, X_i32, WS_f8, XS_f8, C, 1.0,
                        K_I32=K_i32, K_GROUPS=K_groups,
                        BLOCK_GROUPS=block_groups, num_warps=num_warps)
                e.record(); torch.cuda.synchronize()
                us = s.elapsed_time(e) / 500 * 1000
                bw = total_bytes / (us * 1e-6) / 1e12
                print(f"  {label:<38} {us:>6.1f} {bw:>6.2f}")
            except Exception as ex:
                print(f"  {label:<38} FAIL: {str(ex)[:40]}")

    # CUDA reference
    from vllm.model_executor.models.flat_llama_gemv import nvfp4_gemv, _ensure_compiled
    _ensure_compiled()
    for _ in range(200):
        nvfp4_gemv(A, B, SFA, SFB, C, 1.0)
    torch.cuda.synchronize()
    s.record()
    for _ in range(500):
        nvfp4_gemv(A, B, SFA, SFB, C, 1.0)
    e.record(); torch.cuda.synchronize()
    cuda_us = s.elapsed_time(e) / 500 * 1000
    bw2 = total_bytes / (cuda_us * 1e-6) / 1e12
    print(f"  {'CUDA GEMV (reference)':<38} {cuda_us:>6.1f} {bw2:>6.2f}")
