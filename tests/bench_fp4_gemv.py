"""Tune NVFP4 GEMV kernels for FlatLlama decode projection shapes.

This is a benchmark script, not a pytest test. Example:

    CUDA_VISIBLE_DEVICES=3 python tests/bench_fp4_gemv.py --case all

Sweeps BLOCK_K, NUM_WARPS, CP_SIZE, BLOCK_M configurations for the four
BS=1 FlatLlama GEMV shapes:

  qkv:     M=10240, K_bytes=4096
  o:       M=8192,  K_bytes=4096
  gate_up: M=57344, K_bytes=4096
  down:    M=8192,  K_bytes=14336
"""
import argparse

import torch
import triton
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
        __nv_bfloat16 *C_ptr,
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
        C_ptr[m * num_rows + t_row] = __float2bfloat16(master_acc[m]);
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
          C_ptr[m * num_rows + t_row] = __float2bfloat16(master_acc[m]);
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
        C_ptr[m * num_rows + t_row] = __float2bfloat16(master_acc[m]);
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
  auto c = reinterpret_cast<__nv_bfloat16 *>(C.data_ptr());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Each config: <BLOCK_M, BLOCK_K, K, NUM_WARPS, CP_SIZE>
  //                                                        iters  threads  rows/blk  blocks
  if (K == 14336) {
    switch (config) {
      // --- CP_SIZE=32 (32-byte loads, ~111 regs/thread) ---
      case 0:  // baseline
        nvfp4_gemv_kernel<1, 2048, 14336, 2, 32><<<M, 2*32, 0, stream>>>(a,b,sa,sb,c,M);       //  7   64  1
        break;
      case 1:
        nvfp4_gemv_kernel<1, 1024, 14336, 1, 32><<<M, 1*32, 0, stream>>>(a,b,sa,sb,c,M);       // 14   32  1
        break;
      case 2:
        nvfp4_gemv_kernel<1, 7168, 14336, 7, 32><<<M, 7*32, 0, stream>>>(a,b,sa,sb,c,M);       //  2  224  1
        break;
      case 3:
        nvfp4_gemv_kernel<1, 14336, 14336, 14, 32><<<M, 14*32, 0, stream>>>(a,b,sa,sb,c,M);    //  1  448  1
        break;
      case 4:
        nvfp4_gemv_kernel<2, 2048, 14336, 4, 32><<<M/2, 4*32, 0, stream>>>(a,b,sa,sb,c,M);     //  7  128  2
        break;
      case 5:
        nvfp4_gemv_kernel<4, 2048, 14336, 8, 32><<<M/4, 8*32, 0, stream>>>(a,b,sa,sb,c,M);     //  7  256  4
        break;

      // --- CP_SIZE=16 (16-byte loads, ~47 regs/thread -> higher occupancy) ---
      case 6:
        nvfp4_gemv_kernel<1, 2048, 14336, 4, 16><<<M, 4*32, 0, stream>>>(a,b,sa,sb,c,M);       //  7  128  1
        break;
      case 7:
        nvfp4_gemv_kernel<1, 1024, 14336, 2, 16><<<M, 2*32, 0, stream>>>(a,b,sa,sb,c,M);       // 14   64  1
        break;
      case 8:
        nvfp4_gemv_kernel<1, 512, 14336, 1, 16><<<M, 1*32, 0, stream>>>(a,b,sa,sb,c,M);        // 28   32  1
        break;
      case 9:
        nvfp4_gemv_kernel<1, 3584, 14336, 7, 16><<<M, 7*32, 0, stream>>>(a,b,sa,sb,c,M);       //  4  224  1
        break;
      case 10:
        nvfp4_gemv_kernel<2, 1792, 14336, 7, 16><<<M/2, 7*32, 0, stream>>>(a,b,sa,sb,c,M);     //  8  224  2
        break;
      case 11:
        nvfp4_gemv_kernel<2, 512, 14336, 1, 32><<<M/2, 1*32, 0, stream>>>(a,b,sa,sb,c,M);       // 28   32  2
        break;
      case 12:
        nvfp4_gemv_kernel<4, 512, 14336, 2, 32><<<M/4, 2*32, 0, stream>>>(a,b,sa,sb,c,M);       // 28   64  4
        break;
      case 13:
        nvfp4_gemv_kernel<8, 512, 14336, 4, 32><<<M/8, 4*32, 0, stream>>>(a,b,sa,sb,c,M);       // 28  128  8
        break;
      case 14:
        nvfp4_gemv_kernel<2, 1024, 14336, 2, 32><<<M/2, 2*32, 0, stream>>>(a,b,sa,sb,c,M);      // 14   64  2
        break;
      case 15:
        nvfp4_gemv_kernel<4, 1024, 14336, 4, 32><<<M/4, 4*32, 0, stream>>>(a,b,sa,sb,c,M);      // 14  128  4
        break;
      case 16:
        nvfp4_gemv_kernel<8, 1024, 14336, 8, 32><<<M/8, 8*32, 0, stream>>>(a,b,sa,sb,c,M);      // 14  256  8
        break;
      case 17:
        nvfp4_gemv_kernel<4, 256, 14336, 1, 32><<<M/4, 1*32, 0, stream>>>(a,b,sa,sb,c,M);       // 56   32  4
        break;
      case 18:
        nvfp4_gemv_kernel<8, 256, 14336, 2, 32><<<M/8, 2*32, 0, stream>>>(a,b,sa,sb,c,M);       // 56   64  8
        break;
      case 19:
        nvfp4_gemv_kernel<16, 256, 14336, 4, 32><<<M/16, 4*32, 0, stream>>>(a,b,sa,sb,c,M);     // 56  128 16
        break;
      case 20:
        nvfp4_gemv_kernel<8, 128, 14336, 1, 32><<<M/8, 1*32, 0, stream>>>(a,b,sa,sb,c,M);       //112   32  8
        break;
      case 21:
        nvfp4_gemv_kernel<16, 128, 14336, 2, 32><<<M/16, 2*32, 0, stream>>>(a,b,sa,sb,c,M);     //112   64 16
        break;
      case 22:
        nvfp4_gemv_kernel<32, 128, 14336, 4, 32><<<M/32, 4*32, 0, stream>>>(a,b,sa,sb,c,M);     //112  128 32
        break;
      case 23:
        nvfp4_gemv_kernel<16, 64, 14336, 1, 32><<<M/16, 1*32, 0, stream>>>(a,b,sa,sb,c,M);      //224   32 16
        break;
      case 24:
        nvfp4_gemv_kernel<32, 64, 14336, 2, 32><<<M/32, 2*32, 0, stream>>>(a,b,sa,sb,c,M);      //224   64 32
        break;
      case 25:
        nvfp4_gemv_kernel<64, 32, 14336, 1, 32><<<M/64, 1*32, 0, stream>>>(a,b,sa,sb,c,M);      //448   32 64
        break;
      case 26:
        nvfp4_gemv_kernel<2, 512, 14336, 1, 16><<<M/2, 1*32, 0, stream>>>(a,b,sa,sb,c,M);       // 28   32  2
        break;
      case 27:
        nvfp4_gemv_kernel<4, 512, 14336, 2, 16><<<M/4, 2*32, 0, stream>>>(a,b,sa,sb,c,M);       // 28   64  4
        break;
      case 28:
        nvfp4_gemv_kernel<2, 1024, 14336, 4, 16><<<M/2, 4*32, 0, stream>>>(a,b,sa,sb,c,M);      // 14  128  2
        break;
      case 29:
        nvfp4_gemv_kernel<4, 1024, 14336, 8, 16><<<M/4, 8*32, 0, stream>>>(a,b,sa,sb,c,M);      // 14  256  4
        break;
      default:
        TORCH_CHECK(false, "invalid K=14336 config");
    }
  } else if (K == 4096) {
    switch (config) {
      case 0:  // production baseline
        nvfp4_gemv_kernel<1, 2048, 4096, 2, 32><<<M, 2*32, 0, stream>>>(a,b,sa,sb,c,M);        //  2   64  1
        break;
      case 1:
        nvfp4_gemv_kernel<1, 1024, 4096, 1, 32><<<M, 1*32, 0, stream>>>(a,b,sa,sb,c,M);        //  4   32  1
        break;
      case 2:
        nvfp4_gemv_kernel<1, 4096, 4096, 4, 32><<<M, 4*32, 0, stream>>>(a,b,sa,sb,c,M);        //  1  128  1
        break;
      case 3:
        nvfp4_gemv_kernel<2, 2048, 4096, 4, 32><<<M/2, 4*32, 0, stream>>>(a,b,sa,sb,c,M);      //  2  128  2
        break;
      case 4:
        nvfp4_gemv_kernel<4, 2048, 4096, 8, 32><<<M/4, 8*32, 0, stream>>>(a,b,sa,sb,c,M);      //  2  256  4
        break;
      case 5:
        nvfp4_gemv_kernel<2, 4096, 4096, 8, 32><<<M/2, 8*32, 0, stream>>>(a,b,sa,sb,c,M);      //  1  256  2
        break;
      case 6:
        nvfp4_gemv_kernel<2, 512, 4096, 1, 32><<<M/2, 1*32, 0, stream>>>(a,b,sa,sb,c,M);       //  8   32  2
        break;
      case 7:
        nvfp4_gemv_kernel<1, 2048, 4096, 4, 16><<<M, 4*32, 0, stream>>>(a,b,sa,sb,c,M);        //  2  128  1
        break;
      case 8:
        nvfp4_gemv_kernel<1, 1024, 4096, 2, 16><<<M, 2*32, 0, stream>>>(a,b,sa,sb,c,M);        //  4   64  1
        break;
      case 9:
        nvfp4_gemv_kernel<2, 1024, 4096, 4, 16><<<M/2, 4*32, 0, stream>>>(a,b,sa,sb,c,M);      //  4  128  2
        break;
      case 10:
        nvfp4_gemv_kernel<4, 1024, 4096, 8, 16><<<M/4, 8*32, 0, stream>>>(a,b,sa,sb,c,M);      //  4  256  4
        break;
      case 11:
        nvfp4_gemv_kernel<4, 256, 4096, 1, 32><<<M/4, 1*32, 0, stream>>>(a,b,sa,sb,c,M);        // 16   32  4
        break;
      case 12:
        nvfp4_gemv_kernel<8, 128, 4096, 1, 32><<<M/8, 1*32, 0, stream>>>(a,b,sa,sb,c,M);        // 32   32  8
        break;
      case 13:
        nvfp4_gemv_kernel<16, 64, 4096, 1, 32><<<M/16, 1*32, 0, stream>>>(a,b,sa,sb,c,M);       // 64   32 16
        break;
      case 14:
        nvfp4_gemv_kernel<32, 32, 4096, 1, 32><<<M/32, 1*32, 0, stream>>>(a,b,sa,sb,c,M);       //128   32 32
        break;
      case 15:
        nvfp4_gemv_kernel<1, 512, 4096, 1, 16><<<M, 1*32, 0, stream>>>(a,b,sa,sb,c,M);          //  8   32  1
        break;
      case 16:
        nvfp4_gemv_kernel<2, 256, 4096, 1, 16><<<M/2, 1*32, 0, stream>>>(a,b,sa,sb,c,M);        // 16   32  2
        break;
      case 17:
        nvfp4_gemv_kernel<4, 128, 4096, 1, 16><<<M/4, 1*32, 0, stream>>>(a,b,sa,sb,c,M);        // 32   32  4
        break;
      case 18:
        nvfp4_gemv_kernel<8, 64, 4096, 1, 16><<<M/8, 1*32, 0, stream>>>(a,b,sa,sb,c,M);         // 64   32  8
        break;
      case 19:
        nvfp4_gemv_kernel<16, 32, 4096, 1, 16><<<M/16, 1*32, 0, stream>>>(a,b,sa,sb,c,M);       //128   32 16
        break;
      case 20:
        nvfp4_gemv_kernel<2, 1024, 4096, 2, 32><<<M/2, 2*32, 0, stream>>>(a,b,sa,sb,c,M);       //  4   64  2
        break;
      case 21:
        nvfp4_gemv_kernel<4, 1024, 4096, 4, 32><<<M/4, 4*32, 0, stream>>>(a,b,sa,sb,c,M);       //  4  128  4
        break;
      case 22:
        nvfp4_gemv_kernel<8, 1024, 4096, 8, 32><<<M/8, 8*32, 0, stream>>>(a,b,sa,sb,c,M);       //  4  256  8
        break;
      default:
        TORCH_CHECK(false, "invalid K=4096 config");
    }
  } else {
    TORCH_CHECK(false, "unsupported K=", K);
  }
}

TORCH_LIBRARY(nvfp4_gemv_mod, m) {
  m.def("gemv_tune(Tensor A, Tensor B, Tensor SFA, Tensor SFB, Tensor(a!) C, int config) -> ()");
  m.impl("gemv_tune", &gemv_tune);
}
"""

CONFIGS_BY_K = {
    14336: [
        (0, "BK=2048  W=2  CP=32  BM=1  (baseline)"),
        (1, "BK=1024  W=1  CP=32  BM=1"),
        (2, "BK=7168  W=7  CP=32  BM=1"),
        (3, "BK=14336 W=14 CP=32  BM=1  (single-pass)"),
        (4, "BK=2048  W=4  CP=32  BM=2"),
        (5, "BK=2048  W=8  CP=32  BM=4"),
        (6, "BK=2048  W=4  CP=16  BM=1"),
        (7, "BK=1024  W=2  CP=16  BM=1"),
        (8, "BK=512   W=1  CP=16  BM=1"),
        (9, "BK=3584  W=7  CP=16  BM=1"),
        (10, "BK=1792  W=7  CP=16  BM=2"),
        (11, "BK=512   W=1  CP=32  BM=2"),
        (12, "BK=512   W=2  CP=32  BM=4"),
        (13, "BK=512   W=4  CP=32  BM=8"),
        (14, "BK=1024  W=2  CP=32  BM=2"),
        (15, "BK=1024  W=4  CP=32  BM=4"),
        (16, "BK=1024  W=8  CP=32  BM=8"),
        (17, "BK=256   W=1  CP=32  BM=4"),
        (18, "BK=256   W=2  CP=32  BM=8"),
        (19, "BK=256   W=4  CP=32  BM=16"),
        (20, "BK=128   W=1  CP=32  BM=8"),
        (21, "BK=128   W=2  CP=32  BM=16"),
        (22, "BK=128   W=4  CP=32  BM=32"),
        (23, "BK=64    W=1  CP=32  BM=16"),
        (24, "BK=64    W=2  CP=32  BM=32"),
        (25, "BK=32    W=1  CP=32  BM=64"),
        (26, "BK=512   W=1  CP=16  BM=2"),
        (27, "BK=512   W=2  CP=16  BM=4"),
        (28, "BK=1024  W=4  CP=16  BM=2"),
        (29, "BK=1024  W=8  CP=16  BM=4"),
    ],
    4096: [
        (0, "BK=2048 W=2 CP=32 BM=1  (baseline)"),
        (1, "BK=1024 W=1 CP=32 BM=1"),
        (2, "BK=4096 W=4 CP=32 BM=1  (single-pass)"),
        (3, "BK=2048 W=4 CP=32 BM=2"),
        (4, "BK=2048 W=8 CP=32 BM=4"),
        (5, "BK=4096 W=8 CP=32 BM=2"),
        (6, "BK=512  W=1 CP=32 BM=2"),
        (7, "BK=2048 W=4 CP=16 BM=1"),
        (8, "BK=1024 W=2 CP=16 BM=1"),
        (9, "BK=1024 W=4 CP=16 BM=2"),
        (10, "BK=1024 W=8 CP=16 BM=4"),
        (11, "BK=256  W=1 CP=32 BM=4"),
        (12, "BK=128  W=1 CP=32 BM=8"),
        (13, "BK=64   W=1 CP=32 BM=16"),
        (14, "BK=32   W=1 CP=32 BM=32"),
        (15, "BK=512  W=1 CP=16 BM=1"),
        (16, "BK=256  W=1 CP=16 BM=2"),
        (17, "BK=128  W=1 CP=16 BM=4"),
        (18, "BK=64   W=1 CP=16 BM=8"),
        (19, "BK=32   W=1 CP=16 BM=16"),
        (20, "BK=1024 W=2 CP=32 BM=2"),
        (21, "BK=1024 W=4 CP=32 BM=4"),
        (22, "BK=1024 W=8 CP=32 BM=8"),
    ],
}

CASES = {
    "qkv": (10240, 4096),
    "o": (8192, 4096),
    "gate_up": (57344, 4096),
    "down": (8192, 14336),
}


def compile_extension() -> None:
    print("Compiling NVFP4 GEMV tuner configs...")
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


def bench_case(
    name: str,
    m: int,
    k: int,
    device: torch.device,
    warmup: int,
    iters: int,
    cudagraph: bool,
) -> None:
    gemv_tune = torch.ops.nvfp4_gemv_mod.gemv_tune
    k_scales = k // 8

    weight = torch.randint(0, 256, (m, k), dtype=torch.uint8, device=device)
    x = torch.randint(0, 256, (k,), dtype=torch.uint8, device=device)
    weight_scale = torch.randint(1, 4, (m, k_scales),
                                 dtype=torch.uint8, device=device)
    x_scale = torch.randint(1, 4, (k_scales,),
                            dtype=torch.uint8, device=device)
    out = torch.zeros(m, dtype=torch.bfloat16, device=device)

    total_bytes = m * k + m * k_scales
    print(f"\n{name}: M={m}, K_bytes={k}, K_actual={k * 2}")
    print(f"Weight data: {total_bytes / 1e6:.1f} MB"
          f"  (theoretical min @ 8 TB/s: {total_bytes / 8e12 * 1e6:.1f} us)")
    print(f"{'Config':<45} {'us':>8} {'TB/s':>8} {'vs base':>8}")
    print("-" * 76)

    baseline = None
    best = (float("inf"), -1, "")
    for cfg_id, desc in CONFIGS_BY_K[k]:
        for _ in range(warmup):
            gemv_tune(weight, x, weight_scale, x_scale, out, cfg_id)
        torch.cuda.synchronize(device)

        fn = lambda: gemv_tune(weight, x, weight_scale, x_scale, out, cfg_id)
        if cudagraph:
            us = triton.testing.do_bench_cudagraph(
                fn, rep=iters, return_mode="median") * 1000.0
        else:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize(device)
            us = start.elapsed_time(end) * 1000.0 / iters
        bw = total_bytes / (us * 1e-6) / 1e12

        if baseline is None:
            baseline = us
        if us < best[0]:
            best = (us, cfg_id, desc)
        ratio = f"{baseline / us:.2f}x"

        print(f"  [{cfg_id:>2}] {desc:<38} {us:>7.2f} {bw:>7.2f}  {ratio:>7}")

    print(f"Best {name}: [{best[1]}] {best[2]} -> {best[0]:.2f} us")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=[*CASES.keys(), "all"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--cudagraph",
        action="store_true",
        help="Benchmark each config with CUDA graph replay.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is not None:
            torch.cuda.set_device(device)
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    print(f"Device: {torch.cuda.get_device_name(device)} ({device})")
    print(f"Mode: {'CUDA graph replay' if args.cudagraph else 'event loop'}")
    compile_extension()

    case_names = CASES.keys() if args.case == "all" else [args.case]
    for name in case_names:
        m, k = CASES[name]
        bench_case(name, m, k, device, args.warmup, args.iters, args.cudagraph)


if __name__ == "__main__":
    main()
