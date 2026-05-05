# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NVFP4 GEMV kernel for BS=1 decode — replaces CUTLASS GEMM for the down
projection where K=28672 and CUTLASS achieves only 33% bandwidth.

Adapted from Popcorn leaderboard nvfp4_gemv v2h submission.
Uses PTX FP4→FP16 conversion + FP8 block scales with cache hints
(weight streamed via L2::evict_first, input cached via L1::evict_last).

22.8 μs vs 48 μs CUTLASS on GB300 for down projection (M=8192, K=28672).
"""
import torch
from torch.utils.cpp_extension import load_inline

from vllm.logger import init_logger

logger = init_logger(__name__)

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

__device__ half2 bf16x2_to_half2(int packed) {
  half2 result;
  asm("{\n\t"
      ".reg .b16 lo_bf, hi_bf, lo_f16, hi_f16;\n\t"
      "mov.b32 {lo_bf, hi_bf}, %1;\n\t"
      "cvt.rn.f16.bf16 lo_f16, lo_bf;\n\t"
      "cvt.rn.f16.bf16 hi_f16, hi_bf;\n\t"
      "mov.b32 %0, {lo_f16, hi_f16};\n\t"
      "}"
      : "=r"(reinterpret_cast<int&>(result))
      : "r"(packed));
  return result;
}

__device__ void ldcs_i16x2(int16_t *dst, const void *src) {
  asm volatile("ld.global.L1::no_allocate.v2.b16 {%0, %1}, [%2];\n"
              : "=h"(dst[0]), "=h"(dst[1]) : "l"(src));
}
__device__ void ldca_i16x2(int16_t *dst, const void *src) {
  asm volatile("ld.global.L1::evict_last.v2.b16 {%0, %1}, [%2];\n"
              : "=h"(dst[0]), "=h"(dst[1]) : "l"(src));
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

// ---------------------------------------------------------------------------
// BF16-input variant: weight is FP4+FP8 scales, input is BF16 (no quant).
// Eliminates the activation FP4 quantization kernel entirely.
// K is in fp4x2 bytes (= actual_elements / 2).
// ---------------------------------------------------------------------------
template <int BLOCK_M, int BLOCK_K, int K, int NUM_WARPS>
__global__
__launch_bounds__(NUM_WARPS * WARP_SIZE)
void nvfp4_gemv_bf16in_kernel(
  const char         *A_ptr,    // [M, K]   weight (FP4 packed)
  const __nv_bfloat16 *B_ptr,   // [2*K]    input (BF16)
  const char        *SFA_ptr,   // [M, K/8] weight scales (FP8, row-major)
  __nv_bfloat16      *C_ptr,    // [M]      output (BF16)
  int M, float alpha
) {
  constexpr int CP_SIZE = 32;
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
  constexpr int SF_BLOCK_K = BLOCK_K / 8;
  constexpr int num_cols = BLOCK_K / CP_SIZE;
  constexpr int num_rows = TB_SIZE / num_cols;
  constexpr int GROUPS = CP_SIZE / 16;  // 2 scale-group pairs per thread

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int t_col = tid % num_cols;
  const int t_row = tid / num_cols;

  {
    const int off_m = bid * BLOCK_M;
    const int off_k = t_col * CP_SIZE;
    A_ptr   += off_m * K + off_k;
    B_ptr   += off_k * 2;                  // 2 BF16 elements per fp4x2 byte
    C_ptr   += off_m;
    SFA_ptr += off_m * (K / 8) + off_k / 8;
  }

  int A_rmem[BLOCK_M / num_rows][CP_SIZE / 4];
  int16_t SFA_rmem[BLOCK_M / num_rows][GROUPS];

  half2 A_fp16x2[BLOCK_M / num_rows][GROUPS][16];
  half2 B_fp16x2[GROUPS][16];
  half2 SFA_fp16x2[BLOCK_M / num_rows][GROUPS];
  half2 acc[BLOCK_M / num_rows][GROUPS][2];
  float master_acc[BLOCK_M / num_rows] = {};

  constexpr int num_iters = K / BLOCK_K;
  for (int iter_k = 0; iter_k < num_iters; iter_k++) {
    // Load BF16 input (L1/L2 cached — tiny vs weight matrix)
    {
      const char *B_bytes = reinterpret_cast<const char*>(B_ptr);
      for (int i = 0; i < GROUPS; i++) {
        int B_raw_lo[8], B_raw_hi[8];
        ldca_i32x8(B_raw_lo, B_bytes + i * 64);
        ldca_i32x8(B_raw_hi, B_bytes + i * 64 + 32);
        for (int j = 0; j < 8; j++)
          B_fp16x2[i][j] = bf16x2_to_half2(B_raw_lo[j]);
        for (int j = 0; j < 8; j++)
          B_fp16x2[i][j + 8] = bf16x2_to_half2(B_raw_hi[j]);
      }
    }

    // Load weight (streamed from HBM)
    for (int m = 0; m < BLOCK_M / num_rows; m++) {
      const int row = m * num_rows + t_row;
      ldcs_i16x2(SFA_rmem[m], SFA_ptr + row * (K / 8));
      ldcs_i32x8(A_rmem[m], A_ptr + row * K);
    }

    A_ptr += BLOCK_K;
    B_ptr += BLOCK_K * 2;
    SFA_ptr += SF_BLOCK_K;

    // Unpack weight FP4→FP16, weight scale FP8→FP16 (no SFB pre-multiply)
    for (int m = 0; m < BLOCK_M / num_rows; m++)
      for (int i = 0; i < GROUPS; i++) {
        SFA_fp16x2[m][i] = static_cast<half2>(reinterpret_cast<__nv_fp8x2_e4m3 *>(&SFA_rmem[m])[i]);
        for (int j = 0; j < 4; j++)
          fp4x8_to_fp16x2x4(reinterpret_cast<int *>(&A_fp16x2[m][i][j * 4]), A_rmem[m][i * 4 + j]);
      }

    // Dot product (same structure as FP4 path)
    for (int m = 0; m < BLOCK_M / num_rows; m++)
      for (int i = 0; i < GROUPS; i++) {
        acc[m][i][0] = __hmul2(A_fp16x2[m][i][0], B_fp16x2[i][0]);
        acc[m][i][1] = __hmul2(A_fp16x2[m][i][8], B_fp16x2[i][8]);
        for (int j = 1; j < 8; j++) {
          acc[m][i][0] = __hfma2(A_fp16x2[m][i][0 + j], B_fp16x2[i][0 + j], acc[m][i][0]);
          acc[m][i][1] = __hfma2(A_fp16x2[m][i][8 + j], B_fp16x2[i][8 + j], acc[m][i][1]);
        }
      }

    // Scale by weight scale only (no SFB) and accumulate in FP32
    for (int m = 0; m < BLOCK_M / num_rows; m++)
      for (int i = 0; i < GROUPS; i++) {
        __half2_raw scales = SFA_fp16x2[m][i];
        __half_raw group0 = __hadd(acc[m][i][0].x, acc[m][i][0].y);
        __half_raw group1 = __hadd(acc[m][i][1].x, acc[m][i][1].y);
        asm volatile("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(master_acc[m]) : "h"(group0.x), "h"(scales.x));
        asm volatile("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(master_acc[m]) : "h"(group1.x), "h"(scales.y));
      }
  }

  // Cross-thread reduction (identical to FP4 path)
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
      C_ptr[m * num_rows + t_row] = __float2bfloat16(master_acc[m] * alpha);
}

// ---------------------------------------------------------------------------
// FP4-input variant (original): both weight and input are FP4 + FP8 scales.
// ---------------------------------------------------------------------------

// A[M, K] = weight (FP4 packed), B[K] = input (FP4 packed)
// SFA[M, K/8] = weight scales, SFB[K/8] = input scales (FP8, row-major)
// C[M] = output (BF16), alpha = global scale factor
template <int BLOCK_M, int BLOCK_K, int K, int NUM_WARPS>
__global__
__launch_bounds__(NUM_WARPS * WARP_SIZE)
void nvfp4_gemv_kernel(
  const char     *A_ptr,
  const char     *B_ptr,
  const char   *SFA_ptr,
  const char   *SFB_ptr,
  __nv_bfloat16  *C_ptr,
  int M, float alpha
) {
  constexpr int CP_SIZE = 32;
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
  constexpr int SF_BLOCK_K = BLOCK_K / 8;
  constexpr int num_cols = BLOCK_K / CP_SIZE;
  constexpr int num_rows = TB_SIZE / num_cols;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
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
    ldca_i16x2(SFB_rmem, SFB_ptr);
    ldca_i32x8(B_rmem, B_ptr);
    for (int m = 0; m < BLOCK_M / num_rows; m++) {
      const int row = m * num_rows + t_row;
      ldcs_i16x2(SFA_rmem[m], SFA_ptr + row * (K / 8));
      ldcs_i32x8(A_rmem[m], A_ptr + row * K);
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

  // Cross-thread reduction
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
      C_ptr[m * num_rows + t_row] = __float2bfloat16(master_acc[m] * alpha);
}

void nvfp4_gemv(
  const at::Tensor& A,
  const at::Tensor& B,
  const at::Tensor& SFA,
  const at::Tensor& SFB,
        at::Tensor& C,
  double alpha
) {
  const int M = A.size(0);
  const int K = A.size(1);
  float alpha_f = static_cast<float>(alpha);

  auto a = reinterpret_cast<const char *>(A.data_ptr());
  auto b = reinterpret_cast<const char *>(B.data_ptr());
  auto sa = reinterpret_cast<const char *>(SFA.data_ptr());
  auto sb = reinterpret_cast<const char *>(SFB.data_ptr());
  auto c = reinterpret_cast<__nv_bfloat16 *>(C.data_ptr());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Shape-tuned FlatLlama decode GEMVs.
  // K is FP4x2 bytes. M distinguishes projections when K=4096:
  //   QKV:     M=10240, K=4096
  //   O:       M=8192,  K=4096
  //   Gate+Up: M=57344, K=4096
  //   Down:    M=8192,  K=14336
  //
  // GB300 CUDA-graph replay tuning, tests/bench_fp4_gemv.py --cudagraph:
  //   K=4096 projections prefer BK=1024/W=1 over BK=2048/W=2.
  //   QKV: ~7.70us vs ~7.90us; O: ~6.52us vs ~6.57us.
  //   Gate+Up also prefers BK=1024/W=1 in the event-loop tuner.
  //   Down remains fastest on the existing BK=2048/W=2 baseline.

  // 14336 = 2048 * 7  (down projection: 28672 FP4 / 2)
  if (K == 14336) {
    nvfp4_gemv_kernel<1, 2048, 14336, 2><<<M, 64, 0, stream>>>(a,b,sa,sb,c,M,alpha_f);
  }
  // 4096 = 2048 * 2  (QKV/O/gate_up: 8192 FP4 / 2)
  else if (K == 4096) {
    nvfp4_gemv_kernel<1, 1024, 4096, 1><<<M, 32, 0, stream>>>(a,b,sa,sb,c,M,alpha_f);
  }
  else {
    TORCH_CHECK(false, "nvfp4_gemv: unsupported K=", K);
  }
}

// BF16-input GEMV: weight is FP4+FP8, input is BF16, no activation quant.
void nvfp4_gemv_bf16in(
  const at::Tensor& A,
  const at::Tensor& B,
  const at::Tensor& SFA,
        at::Tensor& C,
  double alpha
) {
  const int M = A.size(0);
  const int K = A.size(1);
  float alpha_f = static_cast<float>(alpha);

  auto a = reinterpret_cast<const char *>(A.data_ptr());
  auto b = reinterpret_cast<const __nv_bfloat16 *>(B.data_ptr());
  auto sa = reinterpret_cast<const char *>(SFA.data_ptr());
  auto c = reinterpret_cast<__nv_bfloat16 *>(C.data_ptr());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (K == 14336) {
    nvfp4_gemv_bf16in_kernel<1, 2048, 14336, 2><<<M, 64, 0, stream>>>(a,b,sa,c,M,alpha_f);
  }
  else if (K == 4096) {
    nvfp4_gemv_bf16in_kernel<1, 2048, 4096, 2><<<M, 64, 0, stream>>>(a,b,sa,c,M,alpha_f);
  }
  else {
    TORCH_CHECK(false, "nvfp4_gemv_bf16in: unsupported K=", K);
  }
}

TORCH_LIBRARY(flat_llama_gemv, m) {
  m.def("nvfp4_gemv(Tensor A, Tensor B, Tensor SFA, Tensor SFB, Tensor(a!) C, float alpha) -> ()");
  m.impl("nvfp4_gemv", &nvfp4_gemv);
  m.def("nvfp4_gemv_bf16in(Tensor A, Tensor B, Tensor SFA, Tensor(a!) C, float alpha) -> ()");
  m.impl("nvfp4_gemv_bf16in", &nvfp4_gemv_bf16in);
}
"""

_compiled = False


def _ensure_compiled():
    global _compiled
    if _compiled:
        return
    import torch
    cap = torch.cuda.get_device_capability()
    arch = f"compute_{cap[0]}{cap[1]}a"
    code = f"sm_{cap[0]}{cap[1]}a"
    logger.info("Compiling NVFP4 GEMV kernel for %s...", code)
    load_inline(
        "flat_llama_gemv",
        cpp_sources="",
        cuda_sources=CUDA_SRC,
        verbose=False,
        is_python_module=False,
        no_implicit_headers=True,
        extra_cuda_cflags=[
            "-O3",
            f"-gencode=arch={arch},code={code}",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--relocatable-device-code=false",
        ],
    )
    _compiled = True


def unswizzle_blockscale(
    swizzled: torch.Tensor, orig_M: int, orig_K: int,
) -> torch.Tensor:
    """Convert CUTLASS swizzled block scales to row-major [M, K] FP8."""
    from vllm.utils.math_utils import round_up
    M_pad = round_up(orig_M, 128)
    K_pad = round_up(orig_K, 4)
    data = swizzled.view(torch.uint8).reshape(
        M_pad // 128, K_pad // 4, 32, 4, 4)
    data = data.permute(0, 3, 2, 1, 4).contiguous()
    return data.reshape(M_pad, K_pad)[:orig_M, :orig_K].contiguous()


def nvfp4_gemv(
    weight: torch.Tensor,
    input_fp4: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
    output: torch.Tensor,
    alpha: float,
) -> None:
    """NVFP4 GEMV: output = (weight @ input) * alpha.

    All scale tensors must be in row-major (non-swizzled) format.
    """
    _ensure_compiled()
    torch.ops.flat_llama_gemv.nvfp4_gemv(
        weight, input_fp4,
        weight_scale, input_scale,
        output, alpha,
    )


def nvfp4_gemv_bf16in(
    weight: torch.Tensor,
    input_bf16: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    alpha: float,
) -> None:
    """NVFP4 GEMV with BF16 input: output = (weight @ input) * alpha.

    Weight is FP4 packed with row-major FP8 scales. Input is BF16 (no quant).
    alpha should be weight_global_scale (= proj.alpha * proj.input_scale_inv).
    """
    _ensure_compiled()
    torch.ops.flat_llama_gemv.nvfp4_gemv_bf16in(
        weight, input_bf16,
        weight_scale,
        output, alpha,
    )
