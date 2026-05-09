#include "ops.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <torch/all.h>

namespace vllm {

constexpr int kMlaPrepThreads = 1024;

__device__ __forceinline__ float load_bf16(const __nv_bfloat16* ptr,
                                           int64_t offset) {
  return __bfloat162float(ptr[offset]);
}

__device__ __forceinline__ __nv_fp8_e4m3 float_to_fp8(float x) {
  x = fminf(fmaxf(x, -448.0f), 448.0f);
  return __nv_fp8_e4m3(x);
}

__global__ __launch_bounds__(kMlaPrepThreads) void mla_prep_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ kv_weight,
    __nv_bfloat16* __restrict__ q_out,
    const int64_t* __restrict__ positions,
    const __nv_bfloat16* __restrict__ cos_sin_cache,
    const int64_t* __restrict__ slot_mapping,
    __nv_fp8_e4m3* __restrict__ kv_cache, const float* __restrict__ scale,
    int64_t qkv_stride0, int64_t qkv_stride1, int64_t q_out_stride0,
    int64_t q_out_stride1, int64_t cache_block_size, int64_t cache_stride,
    int32_t q_rank, int32_t kv_rank, int32_t rope_dim, float eps) {
  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float smem[kMlaPrepThreads];

  float sum = 0.0f;
  for (int col = tid; col < q_rank; col += blockDim.x) {
    const float x = load_bf16(qkv, token * qkv_stride0 + col * qkv_stride1);
    sum += x * x;
  }
  smem[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
  const float q_inv_rms = rsqrtf(smem[0] / static_cast<float>(q_rank) + eps);
  for (int col = tid; col < q_rank; col += blockDim.x) {
    const float x = load_bf16(qkv, token * qkv_stride0 + col * qkv_stride1);
    const float w = __bfloat162float(q_weight[col]);
    q_out[token * q_out_stride0 + col * q_out_stride1] =
        __float2bfloat16(x * q_inv_rms * w);
  }

  sum = 0.0f;
  for (int col = tid; col < kv_rank; col += blockDim.x) {
    const float x =
        load_bf16(qkv, token * qkv_stride0 + (q_rank + col) * qkv_stride1);
    sum += x * x;
  }
  smem[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
  const float kv_inv_rms = rsqrtf(smem[0] / static_cast<float>(kv_rank) + eps);
  const int64_t slot = slot_mapping[token];
  if (slot < 0) {
    return;
  }
  const int64_t block_idx = slot / cache_block_size;
  const int64_t block_offset = slot - block_idx * cache_block_size;
  const int64_t cache_offset =
      block_idx * cache_block_size * cache_stride + block_offset * cache_stride;
  const float inv_scale = 1.0f / scale[0];

  for (int col = tid; col < kv_rank + rope_dim; col += blockDim.x) {
    float val;
    if (col < kv_rank) {
      const float x =
          load_bf16(qkv, token * qkv_stride0 + (q_rank + col) * qkv_stride1);
      const float w = __bfloat162float(kv_weight[col]);
      val = x * kv_inv_rms * w;
    } else {
      const int rope_feature = col - kv_rank;
      const int pair = rope_feature / 2;
      const int x_feature = pair * 2;
      const int y_feature = x_feature + 1;
      const int64_t pos = positions[token];
      const int64_t rope_base = q_rank + kv_rank;
      const float cos = load_bf16(cos_sin_cache, pos * rope_dim + pair);
      const float sin =
          load_bf16(cos_sin_cache, pos * rope_dim + rope_dim / 2 + pair);
      const float x = load_bf16(
          qkv, token * qkv_stride0 + (rope_base + x_feature) * qkv_stride1);
      const float y = load_bf16(
          qkv, token * qkv_stride0 + (rope_base + y_feature) * qkv_stride1);
      val = (rope_feature & 1) == 0 ? (x * cos - y * sin) : (y * cos + x * sin);
    }
    kv_cache[cache_offset + col] = float_to_fp8(val * inv_scale);
  }
}

}  // namespace vllm

void mla_qkv_a_rmsnorm_k_rope_cache_fp8(
    const torch::Tensor& qkv, const torch::Tensor& q_weight,
    const torch::Tensor& kv_weight, torch::Tensor& q_out,
    const torch::Tensor& positions, const torch::Tensor& cos_sin_cache,
    const torch::Tensor& slot_mapping, torch::Tensor& kv_cache,
    const torch::Tensor& scale, int64_t q_rank, int64_t kv_rank,
    int64_t rope_dim, int64_t cache_block_size, int64_t cache_stride,
    double eps) {
  TORCH_CHECK(qkv.is_cuda(), "qkv must be CUDA");
  TORCH_CHECK(q_out.is_cuda(), "q_out must be CUDA");
  TORCH_CHECK(kv_cache.is_cuda(), "kv_cache must be CUDA");
  TORCH_CHECK(qkv.scalar_type() == at::ScalarType::BFloat16,
              "qkv must be bfloat16");
  TORCH_CHECK(q_out.scalar_type() == at::ScalarType::BFloat16,
              "q_out must be bfloat16");
  TORCH_CHECK(cos_sin_cache.scalar_type() == at::ScalarType::BFloat16,
              "cos_sin_cache must be bfloat16");
  TORCH_CHECK(kv_cache.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "kv_cache must be float8_e4m3fn");
  TORCH_CHECK(positions.scalar_type() == at::ScalarType::Long,
              "positions must be int64");
  TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Long,
              "slot_mapping must be int64");
  TORCH_CHECK(scale.scalar_type() == at::ScalarType::Float,
              "scale must be float32");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qkv));
  const int64_t num_tokens = qkv.size(0);
  const dim3 grid(static_cast<uint32_t>(num_tokens));
  const dim3 block(vllm::kMlaPrepThreads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  vllm::mla_prep_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(kv_weight.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr()),
      positions.data_ptr<int64_t>(),
      reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache.data_ptr()),
      slot_mapping.data_ptr<int64_t>(),
      reinterpret_cast<__nv_fp8_e4m3*>(kv_cache.data_ptr()),
      scale.data_ptr<float>(), qkv.stride(0), qkv.stride(1), q_out.stride(0),
      q_out.stride(1), cache_block_size, cache_stride,
      static_cast<int32_t>(q_rank), static_cast<int32_t>(kv_rank),
      static_cast<int32_t>(rope_dim), static_cast<float>(eps));
  const cudaError_t result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess,
              "mla_qkv_a_rmsnorm_k_rope_cache_fp8 kernel failed: ",
              cudaGetErrorString(result));
}
