"""Tune NVFP4 GEMV kernel for down projection (M=8192, K=28672).

Sweeps BLOCK_K, NUM_WARPS, CP_SIZE, BLOCK_M configurations.
14336 = 2^11 * 7, so valid BLOCK_K divisors include:
  512, 1024, 2048, 3584, 7168, 14336
"""
import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int WARP_SIZE = 32;

__device__
void fp4x8_to_fp16x2x4(int *out, int in) {
  asm volatile(
    "{\n\t"
    ".reg .b8 tmp0, tmp1, tmp2, tmp3;\n\t"
    "mov.b32 {tmp0, tmp1, tmp2, tmp3}, %4;\n\t"
    "cvt.rn.f16x2.e2m1x2 %0, tmp0;\n\t"
    "cvt.rn.f16x2.e2m1x2 %1, tmp1;\n\t"
    "cvt.rn.f16x2.e2m1x2 %2, tmp2;\n\t"
    "cvt.rn.f16x2.e2m1x2 %3, tmp3;\n\t"
    "}"
    : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
    : "r"(in)
  );
}

__device__ void ldcs_i16(int16_t *dst, const void *src) {
  asm volatile("ld.global.L1::no_allocate.b16 %0, [%1];" : "=h"(dst[0]) : "l"(src));
}
__device__ void ldca_i16(int16_t *dst, const void *src) {
  asm volatile("ld.global.L1::evict_last.b16 %0, [%1];" : "=h"(dst[0]) : "l"(src));
}
__device__ void ldcs_i16x2(int16_t *dst, const void *src) {
  asm volatile("ld.global.L1::no_allocate.v2.b16 {%0, %1}, [%2];\n"
              : "=h"(dst[0]), "=h"(dst[1]) : "l"(src));
}
__device__ void ldca_i16x2(int16_t *dst, const void *src) {
  asm volatile("ld.global.L1::evict_last.v2.b16 {%0, %1}, [%2];\n"
              : "=h"(dst[0]), "=h"(dst[1]) : "l"(src));
}
__device__ void ldcs_i32x4(int *dst, const void *src) {
  asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
              : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
              : "l"(src));
}
__device__ void ldca_i32x4(int *dst, const void *src) {
  asm volatile("ld.global.L1::evict_last.v4.b32 {%0, %1, %2, %3}, [%4];"
              : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
              : "l"(src));
}
__device__ void ldcs_i32x8(int *dst, const void *src) {
  asm volatile("ld.global.L1::no_allocate.L2::evict_first.v8.b32 "
              "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
              : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]),
                "=r"(dst[4]), "=r"(dst[5]), "=r"(dst[6]), "=r"(dst[7])
              : "l"(src));
}
__device__ void ldca_i32x8(int *dst, const void *src) {
  asm volatile("ld.global.L1::evict_last.L2::evict_last.v8.b32 "
              "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
              : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]),
                "=r"(dst[4]), "=r"(dst[5]), "=r"(dst[6]), "=r"(dst[7])
              : "l"(src));
}

template <int BLOCK_M, int BLOCK_K, int K, int NUM_WARPS, int CP_SIZE>
__global__
__launch_bounds__(NUM_WARPS * WARP_SIZE)
void nvfp4_gemv_kernel(
  const char   *A_ptr,
  const char   *B_ptr,
  const char *SFA_ptr,
  const char *SFB_ptr,
        half   *C_ptr,
  int M
) {
  static_assert(BLOCK_K % CP_SIZE == 0);
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
  constexpr int SF_BLOCK_K = BLOCK_K / 8;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  constexpr int num_cols = BLOCK_K / CP_SIZE;
  static_assert(num_cols <= TB_SIZE);
  constexpr int num_rows = TB_SIZE / num_cols;

  const int t_col = tid % num_cols;
  const int t_row = tid / num_cols;

  {
    const int off_m = bid * BLOCK_M;
    const int off_k = t_col * CP_SIZE;
    A_ptr   += off_m * K + off_k;
    B_ptr   += off_k;
    C_ptr   += off_m;
    SFA_ptr += off_m * (K / 8) + off_k / 8;
    SFB_ptr += off_k / 8;
  }

  int A_rmem[BLOCK_M / num_rows][CP_SIZE / 4];
  int B_rmem[CP_SIZE / 4];
  int16_t SFA_rmem[BLOCK_M / num_rows][CP_SIZE / 16];
  int16_t SFB_rmem[CP_SIZE / 16];

  half2 A_fp16x2[BLOCK_M / num_rows][CP_SIZE / 16][16];
  half2 B_fp16x2[CP_SIZE / 16][16];
  half2 SFA_fp16x2[BLOCK_M / num_rows][CP_SIZE / 16];
  half2 SFB_fp16x2[CP_SIZE / 16];

  half2 acc[BLOCK_M / num_rows][CP_SIZE / 16][2];
  float master_acc[BLOCK_M / num_rows] = {};

  constexpr int num_iters = K / BLOCK_K;
  for (int iter_k = 0; iter_k < num_iters; iter_k++) {
    if constexpr (CP_SIZE == 16) {
      ldca_i16(SFB_rmem, SFB_ptr);
      ldca_i32x4(B_rmem, B_ptr);
      for (int m = 0; m < BLOCK_M / num_rows; m++) {
        const int row = m * num_rows + t_row;
        ldcs_i16(SFA_rmem[m], SFA_ptr + row * (K / 8));
        ldcs_i32x4(A_rmem[m], A_ptr + row * K);
      }
    }
    else if constexpr (CP_SIZE == 32) {
      ldca_i16x2(SFB_rmem, SFB_ptr);
      ldca_i32x8(B_rmem, B_ptr);
      for (int m = 0; m < BLOCK_M / num_rows; m++) {
        const int row = m * num_rows + t_row;
        ldcs_i16x2(SFA_rmem[m], SFA_ptr + row * (K / 8));
        ldcs_i32x8(A_rmem[m], A_ptr + row * K);
      }
    }

    A_ptr += BLOCK_K;
    B_ptr += BLOCK_K;
    SFA_ptr += SF_BLOCK_K;
    SFB_ptr += SF_BLOCK_K;

    for (int i = 0; i < CP_SIZE / 16; i++) {
      SFB_fp16x2[i] = static_cast<half2>(reinterpret_cast<__nv_fp8x2_e4m3 *>(&SFB_rmem)[i]);
      for (int j = 0; j < 4; j++)
        fp4x8_to_fp16x2x4(reinterpret_cast<int *>(&B_fp16x2[i][j * 4]), B_rmem[i * 4 + j]);
    }

    for (int m = 0; m < BLOCK_M / num_rows; m++)
      for (int i = 0; i < CP_SIZE / 16; i++) {
        SFA_fp16x2[m][i] = static_cast<half2>(reinterpret_cast<__nv_fp8x2_e4m3 *>(&SFA_rmem[m])[i]);
        for (int j = 0; j < 4; j++)
          fp4x8_to_fp16x2x4(reinterpret_cast<int *>(&A_fp16x2[m][i][j * 4]), A_rmem[m][i * 4 + j]);
        SFA_fp16x2[m][i] = __hmul2(SFA_fp16x2[m][i], SFB_fp16x2[i]);
      }

    for (int m = 0; m < BLOCK_M / num_rows; m++)
      for (int i = 0; i < CP_SIZE / 16; i++) {
        acc[m][i][0] = __hmul2(A_fp16x2[m][i][0], B_fp16x2[i][0]);
        acc[m][i][1] = __hmul2(A_fp16x2[m][i][8], B_fp16x2[i][8]);
        for (int j = 1; j < 8; j++) {
          acc[m][i][0] = __hfma2(A_fp16x2[m][i][0 + j], B_fp16x2[i][0 + j], acc[m][i][0]);
          acc[m][i][1] = __hfma2(A_fp16x2[m][i][8 + j], B_fp16x2[i][8 + j], acc[m][i][1]);
        }
      }

    for (int m = 0; m < BLOCK_M / num_rows; m++)
      for (int i = 0; i < CP_SIZE / 16; i++) {
        __half2_raw scales = SFA_fp16x2[m][i];
        __half_raw group0 = __hadd(acc[m][i][0].x, acc[m][i][0].y);
        __half_raw group1 = __hadd(acc[m][i][1].x, acc[m][i][1].y);
        asm volatile("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(master_acc[m]) : "h"(group0.x), "h"(scales.x));
        asm volatile("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(master_acc[m]) : "h"(group1.x), "h"(scales.y));
      }
  }

  // reduction
  if constexpr (NUM_WARPS % 2 == 0) {
    if constexpr (num_cols > WARP_SIZE) {
      __shared__ float smem[BLOCK_M / num_rows][TB_SIZE];

      for (int m = 0; m < BLOCK_M / num_rows; m++)
        smem[m][tid] = master_acc[m];
      __syncthreads();

      for (int stride = num_cols / 2; stride >= WARP_SIZE * 2; stride /= 2) {
        if (t_col < stride)
          for (int m = 0; m < BLOCK_M / num_rows; m++) {
            master_acc[m] += smem[m][tid + stride];
            smem[m][tid] = master_acc[m];
          }
        __syncthreads();
      }

      if (t_col < WARP_SIZE)
        for (int m = 0; m < BLOCK_M / num_rows; m++)
          master_acc[m] += smem[m][tid + WARP_SIZE];
    }

    constexpr int start_stride = std::min(num_cols, WARP_SIZE) / 2;
    for (int stride = start_stride; stride > 0; stride /= 2)
      for (int m = 0; m < BLOCK_M / num_rows; m++)
        master_acc[m] += __shfl_down_sync(0xFFFF'FFFF, master_acc[m], stride);

    if (t_col == 0)
      for (int m = 0; m < BLOCK_M / num_rows; m++)
        C_ptr[m * num_rows + t_row] = __float2half(master_acc[m]);
  }
  else if constexpr (NUM_WARPS > 1) {
    __shared__ float smem[BLOCK_M / num_rows][(NUM_WARPS - 1) * WARP_SIZE];

    const int warp_id = tid / WARP_SIZE;
    if (warp_id > 0)
      for (int m = 0; m < BLOCK_M / num_rows; m++)
        smem[m][tid - WARP_SIZE] = master_acc[m];
    __syncthreads();

    if (warp_id == 0) {
      for (int w = 0; w < NUM_WARPS - 1; w++)
        for (int m = 0; m < BLOCK_M / num_rows; m++)
          master_acc[m] += smem[m][tid + w * WARP_SIZE];

      constexpr int start_stride = std::min(num_cols, WARP_SIZE) / 2;
      for (int stride = start_stride; stride > 0; stride /= 2)
        for (int m = 0; m < BLOCK_M / num_rows; m++)
          master_acc[m] += __shfl_down_sync(0xFFFF'FFFF, master_acc[m], stride);

      if (t_col == 0)
        for (int m = 0; m < BLOCK_M / num_rows; m++)
          C_ptr[m * num_rows + t_row] = __float2half(master_acc[m]);
    }
  }
  else {
    // NUM_WARPS == 1: single warp, just shuffle reduce
    constexpr int start_stride = std::min(num_cols, WARP_SIZE) / 2;
    for (int stride = start_stride; stride > 0; stride /= 2)
      for (int m = 0; m < BLOCK_M / num_rows; m++)
        master_acc[m] += __shfl_down_sync(0xFFFF'FFFF, master_acc[m], stride);

    if (t_col == 0)
      for (int m = 0; m < BLOCK_M / num_rows; m++)
        C_ptr[m * num_rows + t_row] = __float2half(master_acc[m]);
  }
}

void gemv_tune(
  const at::Tensor& A,
  const at::Tensor& B,
  const at::Tensor& SFA,
  const at::Tensor& SFB,
        at::Tensor& C,
  int64_t config
) {
  const int M = A.size(0);
  const int K = A.size(1);

  auto a = reinterpret_cast<const char *>(A.data_ptr());
  auto b = reinterpret_cast<const char *>(B.data_ptr());
  auto sa = reinterpret_cast<const char *>(SFA.data_ptr());
  auto sb = reinterpret_cast<const char *>(SFB.data_ptr());
  auto c = reinterpret_cast<half *>(C.data_ptr());

  TORCH_CHECK(K == 14336, "tuning for K=14336 only");

  // Each config: <BLOCK_M, BLOCK_K, K, NUM_WARPS, CP_SIZE>
  //                                                        iters  threads  rows/blk  blocks
  switch (config) {
    // --- CP_SIZE=32 (32-byte loads, ~111 regs/thread) ---
    case 0:  // baseline
      nvfp4_gemv_kernel<1, 2048, 14336, 2, 32><<<M, 2*32>>>(a,b,sa,sb,c,M);       //  7   64  1  8192
      break;
    case 1:
      nvfp4_gemv_kernel<1, 1024, 14336, 1, 32><<<M, 1*32>>>(a,b,sa,sb,c,M);       // 14   32  1  8192
      break;
    case 2:
      nvfp4_gemv_kernel<1, 7168, 14336, 7, 32><<<M, 7*32>>>(a,b,sa,sb,c,M);       //  2  224  1  8192
      break;
    case 3:
      nvfp4_gemv_kernel<1, 14336, 14336, 14, 32><<<M, 14*32>>>(a,b,sa,sb,c,M);    //  1  448  1  8192
      break;
    case 4:
      nvfp4_gemv_kernel<2, 2048, 14336, 4, 32><<<M/2, 4*32>>>(a,b,sa,sb,c,M);     //  7  128  2  4096
      break;
    case 5:
      nvfp4_gemv_kernel<4, 2048, 14336, 8, 32><<<M/4, 8*32>>>(a,b,sa,sb,c,M);     //  7  256  4  2048
      break;

    // --- CP_SIZE=16 (16-byte loads, ~47 regs/thread → higher occupancy) ---
    case 6:
      nvfp4_gemv_kernel<1, 2048, 14336, 4, 16><<<M, 4*32>>>(a,b,sa,sb,c,M);       //  7  128  1  8192
      break;
    case 7:
      nvfp4_gemv_kernel<1, 1024, 14336, 2, 16><<<M, 2*32>>>(a,b,sa,sb,c,M);       // 14   64  1  8192
      break;
    case 8:
      nvfp4_gemv_kernel<1, 512, 14336, 1, 16><<<M, 1*32>>>(a,b,sa,sb,c,M);        // 28   32  1  8192
      break;
    case 9:
      nvfp4_gemv_kernel<1, 3584, 14336, 7, 16><<<M, 7*32>>>(a,b,sa,sb,c,M);       //  4  224  1  8192
      break;
    case 10:
      nvfp4_gemv_kernel<2, 1792, 14336, 7, 16><<<M/2, 7*32>>>(a,b,sa,sb,c,M);     //  8  224  2  4096
      break;
    default:
      TORCH_CHECK(false, "invalid config");
  }
}

TORCH_LIBRARY(nvfp4_gemv_mod, m) {
  m.def("gemv_tune(Tensor A, Tensor B, Tensor SFA, Tensor SFB, Tensor(a!) C, int config) -> ()");
  m.impl("gemv_tune", &gemv_tune);
}
"""

print("Compiling NVFP4 GEMV kernel (11 configs)...")
load_inline(
    "nvfp4_gemv_mod",
    cpp_sources="",
    cuda_sources=CUDA_SRC,
    verbose=False,
    is_python_module=False,
    no_implicit_headers=True,
    extra_cuda_cflags=[
        "-O3",
        "-gencode=arch=compute_103a,code=sm_103a",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--relocatable-device-code=false",
    ],
)

gemv_tune = torch.ops.nvfp4_gemv_mod.gemv_tune

CONFIGS = [
    # (id, description)
    (0,  "BK=2048  W=2  CP=32  BM=1  (baseline)"),
    (1,  "BK=1024  W=1  CP=32  BM=1"),
    (2,  "BK=7168  W=7  CP=32  BM=1"),
    (3,  "BK=14336 W=14 CP=32  BM=1  (single-pass)"),
    (4,  "BK=2048  W=4  CP=32  BM=2"),
    (5,  "BK=2048  W=8  CP=32  BM=4"),
    (6,  "BK=2048  W=4  CP=16  BM=1"),
    (7,  "BK=1024  W=2  CP=16  BM=1"),
    (8,  "BK=512   W=1  CP=16  BM=1"),
    (9,  "BK=3584  W=7  CP=16  BM=1"),
    (10, "BK=1792  W=7  CP=16  BM=2"),
]

M = 8192
K = 14336
K_scales = K // 8

A   = torch.randint(0, 256, (M, K), dtype=torch.uint8, device="cuda")
B   = torch.randint(0, 256, (K,),   dtype=torch.uint8, device="cuda")
SFA = torch.randint(1, 4,   (M, K_scales), dtype=torch.uint8, device="cuda")
SFB = torch.randint(1, 4,   (K_scales,),   dtype=torch.uint8, device="cuda")
C   = torch.zeros(M, dtype=torch.float16, device="cuda")

total_bytes = M * K + M * K_scales

print(f"\nDown projection: M={M}, K_actual=28672, K_fp4x2={K}")
print(f"Weight data: {total_bytes/1e6:.1f} MB  (theoretical min @ 8 TB/s: {total_bytes/8e12*1e6:.1f} μs)")
print(f"{'Config':<42} {'μs':>7} {'TB/s':>7} {'vs base':>8}")
print("─" * 70)

baseline = None
for cfg_id, desc in CONFIGS:
    # warmup
    for _ in range(100):
        gemv_tune(A, B, SFA, SFB, C, cfg_id)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(1000):
        gemv_tune(A, B, SFA, SFB, C, cfg_id)
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) / 1000 * 1000
    bw = total_bytes / (us * 1e-6) / 1e12

    if baseline is None:
        baseline = us
    ratio = f"{baseline/us:.2f}x"

    print(f"  [{cfg_id:>2}] {desc:<36} {us:>6.1f} {bw:>6.2f}  {ratio:>7}")
