# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.utils.cpp_extension import load_inline


CUDA_SRC = r"""
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

namespace {

constexpr int kTopK = 4;
constexpr int kThreads = 256;

__device__ __forceinline__ float fp4_to_float(unsigned int x) {
  switch (x & 0xF) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    case 7: return 6.0f;
    case 8: return -0.0f;
    case 9: return -0.5f;
    case 10: return -1.0f;
    case 11: return -1.5f;
    case 12: return -2.0f;
    case 13: return -3.0f;
    case 14: return -4.0f;
    default: return -6.0f;
  }
}

__device__ __forceinline__ float scale_byte_to_float(uint8_t x) {
  return __uint_as_float(static_cast<uint32_t>(x) << 23);
}

__device__ __forceinline__ float load_mxfp4(
    const uint8_t* weight,
    const uint8_t* scale,
    int row,
    int k,
    int packed_k,
    int scale_k) {
  uint8_t packed = weight[row * packed_k + (k >> 1)];
  unsigned int nibble = (k & 1) ? (packed >> 4) : (packed & 0xF);
  return fp4_to_float(nibble) * scale_byte_to_float(scale[row * scale_k + (k >> 5)]);
}

__global__ void route_top4_kernel(
    const __nv_bfloat16* logits,
    int32_t* topk_ids,
    float* topk_weights,
    int num_experts) {
  if (threadIdx.x != 0) {
    return;
  }
  float vals[kTopK] = {-3.402823466e38f, -3.402823466e38f,
                       -3.402823466e38f, -3.402823466e38f};
  int ids[kTopK] = {-1, -1, -1, -1};
  for (int e = 0; e < num_experts; ++e) {
    float v = __bfloat162float(logits[e]);
    if (v > vals[3]) {
      vals[3] = v;
      ids[3] = e;
      for (int i = 3; i > 0 && vals[i] > vals[i - 1]; --i) {
        float tv = vals[i - 1];
        vals[i - 1] = vals[i];
        vals[i] = tv;
        int ti = ids[i - 1];
        ids[i - 1] = ids[i];
        ids[i] = ti;
      }
    }
  }
  float max_v = vals[0];
  float sum = 0.0f;
  float weights[kTopK];
  for (int i = 0; i < kTopK; ++i) {
    weights[i] = __expf(vals[i] - max_v);
    sum += weights[i];
  }
  float inv_sum = 1.0f / sum;
  for (int i = 0; i < kTopK; ++i) {
    topk_ids[i] = ids[i];
    topk_weights[i] = weights[i] * inv_sum;
  }
}

__global__ void gemm1_kernel(
    const __nv_bfloat16* x,
    const uint8_t* w13,
    const uint8_t* w13_scale,
    const float* w13_bias,
    const int32_t* topk_ids,
    float* gemm1_out,
    int hidden_size,
    int padded_hidden_size,
    int intermediate_size,
    int rows13,
    int packed_k,
    int scale_k) {
  __shared__ float smem[kThreads];
  int out_idx = blockIdx.x;
  int slot = out_idx / rows13;
  int row = out_idx - slot * rows13;
  int expert = topk_ids[slot];
  int expert_row = expert * rows13 + row;

  float acc = 0.0f;
  for (int k = threadIdx.x; k < padded_hidden_size; k += blockDim.x) {
    float xv = k < hidden_size ? __bfloat162float(x[k]) : 0.0f;
    float wv = load_mxfp4(w13, w13_scale, expert_row, k, packed_k, scale_k);
    acc += xv * wv;
  }
  smem[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] += smem[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    gemm1_out[out_idx] = smem[0] + w13_bias[expert_row];
  }
}

__global__ void swiglu_kernel(
    const float* gemm1_out,
    float* act_out,
    int intermediate_size,
    int rows13) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = kTopK * intermediate_size;
  if (idx >= total) {
    return;
  }
  int slot = idx / intermediate_size;
  int col = idx - slot * intermediate_size;
  const float* base = gemm1_out + slot * rows13;
  float gate = fminf(base[col], 7.0f);
  float up = fminf(fmaxf(base[intermediate_size + col], -7.0f), 7.0f);
  float sigmoid = 1.0f / (1.0f + __expf(-1.702f * gate));
  act_out[idx] = gate * sigmoid * (up + 1.0f);
}

__global__ void gemm2_kernel(
    const float* act,
    const uint8_t* w2,
    const uint8_t* w2_scale,
    const float* w2_bias,
    const int32_t* topk_ids,
    const float* topk_weights,
    __nv_bfloat16* out,
    int hidden_size,
    int padded_hidden_size,
    int intermediate_size,
    int packed_k,
    int scale_k) {
  __shared__ float smem[kThreads];
  int row = blockIdx.x;
  float total = 0.0f;

  for (int slot = 0; slot < kTopK; ++slot) {
    int expert = topk_ids[slot];
    int expert_row = expert * padded_hidden_size + row;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < intermediate_size; k += blockDim.x) {
      float av = act[slot * intermediate_size + k];
      float wv = load_mxfp4(w2, w2_scale, expert_row, k, packed_k, scale_k);
      acc += av * wv;
    }
    smem[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        smem[threadIdx.x] += smem[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      total += (smem[0] + w2_bias[expert_row]) * topk_weights[slot];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && row < hidden_size) {
    out[row] = __float2bfloat16(total);
  }
}

}  // namespace

at::Tensor bs1_moe(
    const at::Tensor& hidden_states,
    const at::Tensor& routing_logits,
    const at::Tensor& w13,
    const at::Tensor& w13_scale,
    const at::Tensor& w13_bias,
    const at::Tensor& w2,
    const at::Tensor& w2_scale,
    const at::Tensor& w2_bias,
    int64_t hidden_size) {
  TORCH_CHECK(hidden_states.dim() == 2 && hidden_states.size(0) == 1);
  TORCH_CHECK(routing_logits.dim() == 2 && routing_logits.size(0) == 1);
  TORCH_CHECK(w13.dim() == 3 && w2.dim() == 3);

  const int num_experts = static_cast<int>(w13.size(0));
  const int rows13 = static_cast<int>(w13.size(1));
  const int intermediate_size = rows13 / 2;
  const int padded_hidden_size = static_cast<int>(w13.size(2)) * 2;
  const int out_hidden_size = static_cast<int>(hidden_size);
  const int w13_packed_k = static_cast<int>(w13.size(2));
  const int w13_scale_k = static_cast<int>(w13_scale.size(2));
  const int w2_packed_k = static_cast<int>(w2.size(2));
  const int w2_scale_k = static_cast<int>(w2_scale.size(2));

  auto int_opts = hidden_states.options().dtype(at::kInt);
  auto float_opts = hidden_states.options().dtype(at::kFloat);
  auto out_opts = hidden_states.options().dtype(at::kBFloat16);
  at::Tensor topk_ids = at::empty({kTopK}, int_opts);
  at::Tensor topk_weights = at::empty({kTopK}, float_opts);
  at::Tensor gemm1_out = at::empty({kTopK, rows13}, float_opts);
  at::Tensor act_out = at::empty({kTopK, intermediate_size}, float_opts);
  at::Tensor out = at::empty({1, out_hidden_size}, out_opts);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  route_top4_kernel<<<1, 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(routing_logits.data_ptr()),
      topk_ids.data_ptr<int32_t>(),
      topk_weights.data_ptr<float>(),
      num_experts);
  gemm1_kernel<<<kTopK * rows13, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr()),
      reinterpret_cast<const uint8_t*>(w13.data_ptr()),
      reinterpret_cast<const uint8_t*>(w13_scale.data_ptr()),
      w13_bias.data_ptr<float>(),
      topk_ids.data_ptr<int32_t>(),
      gemm1_out.data_ptr<float>(),
      out_hidden_size,
      padded_hidden_size,
      intermediate_size,
      rows13,
      w13_packed_k,
      w13_scale_k);
  int act_blocks = (kTopK * intermediate_size + kThreads - 1) / kThreads;
  swiglu_kernel<<<act_blocks, kThreads, 0, stream>>>(
      gemm1_out.data_ptr<float>(),
      act_out.data_ptr<float>(),
      intermediate_size,
      rows13);
  gemm2_kernel<<<out_hidden_size, kThreads, 0, stream>>>(
      act_out.data_ptr<float>(),
      reinterpret_cast<const uint8_t*>(w2.data_ptr()),
      reinterpret_cast<const uint8_t*>(w2_scale.data_ptr()),
      w2_bias.data_ptr<float>(),
      topk_ids.data_ptr<int32_t>(),
      topk_weights.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      out_hidden_size,
      padded_hidden_size,
      intermediate_size,
      w2_packed_k,
      w2_scale_k);
  return out;
}

TORCH_LIBRARY(vllm_flat_gpt_oss, m) {
  m.def("bs1_moe(Tensor hidden_states, Tensor routing_logits, Tensor w13, "
        "Tensor w13_scale, Tensor w13_bias, Tensor w2, Tensor w2_scale, "
        "Tensor w2_bias, int hidden_size) -> Tensor");
  m.impl("bs1_moe", &bs1_moe);
}
"""


load_inline(
    name="vllm_flat_gpt_oss_moe_cuda",
    cpp_sources="",
    cuda_sources=CUDA_SRC,
    functions=[],
    verbose=False,
    is_python_module=False,
    extra_cuda_cflags=[
        "-O3",
        "-gencode=arch=compute_103a,code=sm_103a",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
    ],
)


def flat_bs1_moe(
    hidden_states: torch.Tensor,
    routing_logits: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_bias: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_bias: torch.Tensor,
    hidden_size: int,
) -> torch.Tensor:
    return torch.ops.vllm_flat_gpt_oss.bs1_moe(
        hidden_states,
        routing_logits,
        w13,
        w13_scale,
        w13_bias,
        w2,
        w2_scale,
        w2_bias,
        hidden_size,
    )
