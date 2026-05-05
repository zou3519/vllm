# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Specialized FlatLlama BS=1 RMSNorm+NVFP4 quant kernel."""

from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline

from vllm.logger import init_logger

logger = init_logger(__name__)

CUDA_SRC = r"""
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <torch/library.h>

constexpr int N = 8192;
constexpr int THREADS = 256;
constexpr int ELEMS_PER_THREAD = N / THREADS;
constexpr int PAIRS_PER_THREAD = ELEMS_PER_THREAD / 2;
constexpr float FP4_MAX = 6.0f;

__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__device__ __forceinline__ unsigned char fp8_e4m3_byte(float x) {
  __nv_fp8_e4m3 s(x);
  return *reinterpret_cast<unsigned char*>(&s);
}

__device__ __forceinline__ unsigned char fp4x2_byte(float lo, float hi) {
  unsigned short out;
  asm volatile(
    "{ .reg .b8 tmp;"
    "  cvt.rn.satfinite.e2m1x2.f32 tmp, %2, %1;"
    "  cvt.u16.u8 %0, tmp; }"
    : "=h"(out)
    : "f"(lo), "f"(hi));
  return static_cast<unsigned char>(out);
}

__global__ __launch_bounds__(THREADS)
void add_rms_norm_fp4_quant_8192_serial_kernel(
  const __nv_bfloat16* __restrict__ hidden,
  const __nv_bfloat16* __restrict__ residual,
  __nv_bfloat16* __restrict__ residual_out,
  const __nv_bfloat16* __restrict__ weight,
  const float* __restrict__ sf_scale_inv,
  unsigned char* __restrict__ fp4_out,
  unsigned char* __restrict__ scale_out,
  float eps
) {
  __shared__ __nv_bfloat162 r_smem2[N / 2];
  __shared__ float warp_partials[8];
  __shared__ float rsigma_smem;
  __shared__ float sf_scale_smem;

  const int tid = threadIdx.x;
  float local_sum = 0.0f;

  const auto* hidden2 = reinterpret_cast<const __nv_bfloat162*>(hidden);
  const auto* residual2 = reinterpret_cast<const __nv_bfloat162*>(residual);
  auto* residual_out2 = reinterpret_cast<__nv_bfloat162*>(residual_out);

  #pragma unroll
  for (int j = 0; j < PAIRS_PER_THREAD; ++j) {
    const int pair_idx = tid * PAIRS_PER_THREAD + j;
    const float2 h = __bfloat1622float2(hidden2[pair_idx]);
    const float2 r = __bfloat1622float2(residual2[pair_idx]);
    const float sum0 = h.x + r.x;
    const float sum1 = h.y + r.y;
    const __nv_bfloat162 out2 = __floats2bfloat162_rn(sum0, sum1);
    r_smem2[pair_idx] = out2;
    residual_out2[pair_idx] = out2;
    local_sum += sum0 * sum0 + sum1 * sum1;
  }

  local_sum = warp_sum(local_sum);
  if ((tid & 31) == 0) {
    warp_partials[tid >> 5] = local_sum;
  }
  __syncthreads();

  if (tid < 32) {
    float total = tid < 8 ? warp_partials[tid] : 0.0f;
    total = warp_sum(total);
    if (tid == 0) {
      rsigma_smem = rsqrtf(total / static_cast<float>(N) + eps);
      sf_scale_smem = sf_scale_inv[0];
    }
  }
  __syncthreads();

  const float rsigma = rsigma_smem;
  const float sf_scale = sf_scale_smem;

  // 512 groups total; each thread handles two contiguous FP4 groups. This is
  // faster than an 8-lane subgroup version on GB300 for this small row.
  const __nv_bfloat16* r_smem =
      reinterpret_cast<const __nv_bfloat16*>(r_smem2);
  #pragma unroll
  for (int group_iter = 0; group_iter < 2; ++group_iter) {
    const int group = tid + group_iter * THREADS;
    const int base = group * 16;

    float vals[16];
    float amax = 0.0f;
    bool finite = true;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int idx = base + i;
      const float r = __bfloat162float(r_smem[idx]);
      const float w = __bfloat162float(weight[idx]);
      const float v = r * rsigma * w * sf_scale;
      vals[i] = v;
      finite = finite && isfinite(v);
      amax = fmaxf(amax, fabsf(v));
    }

    if (!finite || !(amax > 0.0f)) {
      scale_out[group] = fp8_e4m3_byte(1.0f);
      #pragma unroll
      for (int p = 0; p < 8; ++p) {
        fp4_out[group * 8 + p] = 0;
      }
      continue;
    }

    const float raw_scale = amax / FP4_MAX;
    const float scale = __bfloat162float(
        static_cast<__nv_bfloat16>(__nv_fp8_e4m3(raw_scale)));
    const float inv_scale = 1.0f / scale;
    scale_out[group] = fp8_e4m3_byte(raw_scale);

    #pragma unroll
    for (int p = 0; p < 8; ++p) {
      fp4_out[group * 8 + p] = fp4x2_byte(
          vals[p * 2] * inv_scale,
          vals[p * 2 + 1] * inv_scale);
    }
  }
}

void check_add_rms_norm_fp4_quant_8192_args(
  const at::Tensor& hidden,
  const at::Tensor& residual,
        at::Tensor& residual_out,
  const at::Tensor& weight,
  const at::Tensor& sf_scale_inv,
        at::Tensor& fp4_out,
        at::Tensor& scale_out
) {
  TORCH_CHECK(hidden.numel() == N, "hidden must have 8192 elements");
  TORCH_CHECK(residual.numel() == N, "residual must have 8192 elements");
  TORCH_CHECK(residual_out.numel() == N, "residual_out must have 8192 elements");
  TORCH_CHECK(weight.numel() == N, "weight must have 8192 elements");
  TORCH_CHECK(sf_scale_inv.numel() == 1, "sf_scale_inv must have 1 element");
  TORCH_CHECK(fp4_out.numel() == N / 2, "fp4_out must have 4096 elements");
  TORCH_CHECK(scale_out.numel() == N / 16, "scale_out must have 512 elements");
}

void add_rms_norm_fp4_quant_8192_serial(
  const at::Tensor& hidden,
  const at::Tensor& residual,
        at::Tensor& residual_out,
  const at::Tensor& weight,
  const at::Tensor& sf_scale_inv,
        at::Tensor& fp4_out,
        at::Tensor& scale_out,
  double eps
) {
  check_add_rms_norm_fp4_quant_8192_args(
      hidden, residual, residual_out, weight, sf_scale_inv, fp4_out, scale_out);

  auto h = reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr());
  auto r = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr());
  auto ro = reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr());
  auto w = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr());
  auto s = reinterpret_cast<const float*>(sf_scale_inv.data_ptr());
  auto f4 = reinterpret_cast<unsigned char*>(fp4_out.data_ptr());
  auto sc = reinterpret_cast<unsigned char*>(scale_out.data_ptr());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  add_rms_norm_fp4_quant_8192_serial_kernel<<<1, THREADS, 0, stream>>>(
      h, r, ro, w, s, f4, sc, static_cast<float>(eps));
}

TORCH_LIBRARY(flat_llama_norm_quant, m) {
  m.def("add_rms_norm_fp4_quant_8192(Tensor hidden, Tensor residual, Tensor(a!) residual_out, Tensor weight, Tensor sf_scale_inv, Tensor(a!) fp4_out, Tensor(a!) scale_out, float eps) -> ()");
  m.def("add_rms_norm_fp4_quant_8192_serial(Tensor hidden, Tensor residual, Tensor(a!) residual_out, Tensor weight, Tensor sf_scale_inv, Tensor(a!) fp4_out, Tensor(a!) scale_out, float eps) -> ()");
  m.impl("add_rms_norm_fp4_quant_8192", &add_rms_norm_fp4_quant_8192_serial);
  m.impl("add_rms_norm_fp4_quant_8192_serial", &add_rms_norm_fp4_quant_8192_serial);
}
"""

_compiled = False


def _ensure_compiled() -> None:
    global _compiled
    if _compiled:
        return
    cap = torch.cuda.get_device_capability()
    arch = f"compute_{cap[0]}{cap[1]}a"
    code = f"sm_{cap[0]}{cap[1]}a"
    logger.info("Compiling FlatLlama norm+FP4 quant kernel for %s...", code)
    load_inline(
        "flat_llama_norm_quant",
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


def add_rms_norm_fp4_quant_8192(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    residual_out: torch.Tensor,
    weight: torch.Tensor,
    sf_scale_inv: torch.Tensor,
    fp4_out: torch.Tensor,
    scale_out: torch.Tensor,
    eps: float = 1e-5,
) -> None:
    _ensure_compiled()
    torch.ops.flat_llama_norm_quant.add_rms_norm_fp4_quant_8192(
        hidden_states,
        residual,
        residual_out,
        weight,
        sf_scale_inv,
        fp4_out,
        scale_out,
        eps,
    )


def add_rms_norm_fp4_quant_8192_serial(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    residual_out: torch.Tensor,
    weight: torch.Tensor,
    sf_scale_inv: torch.Tensor,
    fp4_out: torch.Tensor,
    scale_out: torch.Tensor,
    eps: float = 1e-5,
) -> None:
    _ensure_compiled()
    torch.ops.flat_llama_norm_quant.add_rms_norm_fp4_quant_8192_serial(
        hidden_states,
        residual,
        residual_out,
        weight,
        sf_scale_inv,
        fp4_out,
        scale_out,
        eps,
    )
