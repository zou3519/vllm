#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace vllm {

constexpr int kIndexerPrepThreads = 128;

__device__ __forceinline__ float load_bf16(const __nv_bfloat16* ptr,
                                           int64_t offset) {
  return __bfloat162float(ptr[offset]);
}

__device__ __forceinline__ __nv_fp8_e4m3 float_to_fp8(float x) {
  x = fminf(fmaxf(x, -448.0f), 448.0f);
  return __nv_fp8_e4m3(x);
}

__device__ __forceinline__ float block_reduce_sum(float sum, float* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  if (lane == 0) {
    smem[warp] = sum;
  }
  __syncthreads();

  const int num_warps = (blockDim.x + 31) >> 5;
  sum = threadIdx.x < num_warps ? smem[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
  }
  if (threadIdx.x == 0) {
    smem[0] = sum;
  }
  __syncthreads();
  return smem[0];
}

__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  if (lane == 0) {
    smem[warp] = val;
  }
  __syncthreads();

  const int num_warps = (blockDim.x + 31) >> 5;
  val = threadIdx.x < num_warps ? smem[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
  }
  if (threadIdx.x == 0) {
    smem[0] = val;
  }
  __syncthreads();
  return smem[0];
}

__device__ __forceinline__ float pow2_ceil_scale(float x) {
  return exp2f(ceilf(log2f(x)));
}

__global__ __launch_bounds__(kIndexerPrepThreads) void indexer_prep_kernel(
    const __nv_bfloat16* __restrict__ index_q,
    const __nv_bfloat16* __restrict__ index_k,
    const int64_t* __restrict__ positions,
    const __nv_bfloat16* __restrict__ cos_sin_cache,
    const __nv_bfloat16* __restrict__ index_weights,
    __nv_fp8_e4m3* __restrict__ q_fp8,
    float* __restrict__ scaled_weights,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ norm_bias,
    const int64_t* __restrict__ slot_mapping,
    __nv_fp8_e4m3* __restrict__ kv_cache_fp8,
    float* __restrict__ kv_cache_f32,
    int64_t num_k_tokens, int64_t index_q_stride0, int64_t index_q_stride1,
    int64_t index_q_stride2, int64_t index_k_stride0,
    int64_t index_k_stride1, int64_t weights_stride0, int64_t weights_stride1,
    int64_t q_fp8_stride0, int64_t q_fp8_stride1, int64_t q_fp8_stride2,
    int64_t scaled_weights_stride0, int64_t scaled_weights_stride1,
    int64_t cache_block_size, int64_t cache_stride, int32_t n_heads,
    int32_t head_dim, int32_t rope_dim, float eps, float factor) {
  const int token = blockIdx.x;
  const int lane_or_k = blockIdx.y;
  const int tid = threadIdx.x;
  __shared__ float smem[32];
  const int half_rope = rope_dim / 2;
  const int64_t pos = positions[token];

  if (lane_or_k < n_heads) {
    const int head = lane_or_k;
    float val = 0.0f;
    if (tid < head_dim) {
      val = load_bf16(index_q, token * index_q_stride0 +
                                  head * index_q_stride1 +
                                  tid * index_q_stride2);
      if (tid < rope_dim) {
        const int pair = tid < half_rope ? tid : tid - half_rope;
        const float cos = load_bf16(cos_sin_cache, pos * rope_dim + pair);
        const float sin =
            load_bf16(cos_sin_cache, pos * rope_dim + half_rope + pair);
        const float x =
            load_bf16(index_q, token * index_q_stride0 +
                                   head * index_q_stride1 +
                                   pair * index_q_stride2);
        const float y =
            load_bf16(index_q, token * index_q_stride0 +
                                   head * index_q_stride1 +
                                   (pair + half_rope) * index_q_stride2);
        val = tid < half_rope ? x * cos - y * sin : y * cos + x * sin;
      }
    }
    const float absmax =
        block_reduce_max(tid < head_dim ? fabsf(val) : 0.0f, smem);
    const float scale = pow2_ceil_scale(fmaxf(absmax / 448.0f, 1.0e-10f));
    if (tid < head_dim) {
      q_fp8[token * q_fp8_stride0 + head * q_fp8_stride1 +
            tid * q_fp8_stride2] = float_to_fp8(val / scale);
    }
    if (tid == 0) {
      const float w =
          load_bf16(index_weights, token * weights_stride0 + head * weights_stride1);
      scaled_weights[token * scaled_weights_stride0 +
                     head * scaled_weights_stride1] = w * scale * factor;
    }
    return;
  }

  const bool valid_token = token < num_k_tokens;
  float x = 0.0f;
  if (tid < head_dim && valid_token) {
    x = load_bf16(index_k, token * index_k_stride0 + tid * index_k_stride1);
  }
  const float mean =
      block_reduce_sum(tid < head_dim ? x : 0.0f, smem) / static_cast<float>(head_dim);
  const float centered = tid < head_dim ? x - mean : 0.0f;
  const float var = block_reduce_sum(centered * centered, smem) /
                    static_cast<float>(head_dim);
  const float inv_std = rsqrtf(var + eps);

  float val = 0.0f;
  if (tid < head_dim && valid_token) {
    const float weight = load_bf16(norm_weight, tid);
    const float bias = load_bf16(norm_bias, tid);
    val = centered * inv_std * weight + bias;
    if (tid < rope_dim) {
      const int pair = tid < half_rope ? tid : tid - half_rope;
      const float cos = load_bf16(cos_sin_cache, pos * rope_dim + pair);
      const float sin =
          load_bf16(cos_sin_cache, pos * rope_dim + half_rope + pair);
      const float raw_x =
          load_bf16(index_k, token * index_k_stride0 + pair * index_k_stride1);
      const float raw_y = load_bf16(index_k, token * index_k_stride0 +
                                                 (pair + half_rope) *
                                                     index_k_stride1);
      const float wx = load_bf16(norm_weight, pair);
      const float wy = load_bf16(norm_weight, pair + half_rope);
      const float bx = load_bf16(norm_bias, pair);
      const float by = load_bf16(norm_bias, pair + half_rope);
      const float nx = (raw_x - mean) * inv_std * wx + bx;
      const float ny = (raw_y - mean) * inv_std * wy + by;
      val = tid < half_rope ? nx * cos - ny * sin : ny * cos + nx * sin;
    }
  }

  const float absmax =
      block_reduce_max(tid < head_dim ? fabsf(val) : 0.0f, smem);
  const float scale = pow2_ceil_scale(fmaxf(absmax, 1.0e-4f) / 448.0f);
  if (!valid_token) {
    return;
  }
  const int64_t slot = slot_mapping[token];
  if (slot < 0) {
    return;
  }
  const int64_t block_idx = slot / cache_block_size;
  const int64_t block_offset = slot - block_idx * cache_block_size;
  const int64_t base = block_idx * cache_block_size * cache_stride;
  if (tid < head_dim) {
    kv_cache_fp8[base + block_offset * head_dim + tid] =
        float_to_fp8(val / scale);
  }
  if (tid == 0) {
    const int64_t scale_byte_offset =
        base + cache_block_size * head_dim + block_offset * 4;
    kv_cache_f32[scale_byte_offset / 4] = scale;
  }
}

}  // namespace vllm

void indexer_qk_rope_quant_cache(
    const torch::Tensor& index_q, const torch::Tensor& index_k,
    const torch::Tensor& positions, const torch::Tensor& cos_sin_cache,
    const torch::Tensor& index_weights, torch::Tensor& q_fp8,
    torch::Tensor& scaled_weights, const torch::Tensor& norm_weight,
    const torch::Tensor& norm_bias, const torch::Tensor& slot_mapping,
    torch::Tensor& kv_cache_fp8, torch::Tensor& kv_cache_f32,
    int64_t num_k_tokens, int64_t n_heads, int64_t head_dim, int64_t rope_dim,
    int64_t cache_block_size, int64_t cache_stride, double eps, double factor) {
  TORCH_CHECK(index_q.is_cuda(), "index_q must be CUDA");
  TORCH_CHECK(index_k.is_cuda(), "index_k must be CUDA");
  TORCH_CHECK(q_fp8.is_cuda(), "q_fp8 must be CUDA");
  TORCH_CHECK(kv_cache_fp8.is_cuda(), "kv_cache_fp8 must be CUDA");
  TORCH_CHECK(index_q.scalar_type() == at::ScalarType::BFloat16,
              "index_q must be bfloat16");
  TORCH_CHECK(index_k.scalar_type() == at::ScalarType::BFloat16,
              "index_k must be bfloat16");
  TORCH_CHECK(index_weights.scalar_type() == at::ScalarType::BFloat16,
              "index_weights must be bfloat16");
  TORCH_CHECK(norm_weight.scalar_type() == at::ScalarType::BFloat16,
              "norm_weight must be bfloat16");
  TORCH_CHECK(norm_bias.scalar_type() == at::ScalarType::BFloat16,
              "norm_bias must be bfloat16");
  TORCH_CHECK(q_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "q_fp8 must be float8_e4m3fn");
  TORCH_CHECK(kv_cache_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "kv_cache_fp8 must be float8_e4m3fn");
  TORCH_CHECK(scaled_weights.scalar_type() == at::ScalarType::Float,
              "scaled_weights must be float32");
  TORCH_CHECK(kv_cache_f32.scalar_type() == at::ScalarType::Float,
              "kv_cache_f32 must be float32");
  TORCH_CHECK(positions.scalar_type() == at::ScalarType::Long,
              "positions must be int64");
  TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Long,
              "slot_mapping must be int64");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(index_q));
  const dim3 grid(static_cast<uint32_t>(index_q.size(0)),
                  static_cast<uint32_t>(n_heads + 1));
  const dim3 block(vllm::kIndexerPrepThreads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vllm::indexer_prep_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(index_q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(index_k.data_ptr()),
      positions.data_ptr<int64_t>(),
      reinterpret_cast<const __nv_bfloat16*>(cos_sin_cache.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(index_weights.data_ptr()),
      reinterpret_cast<__nv_fp8_e4m3*>(q_fp8.data_ptr()),
      scaled_weights.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(norm_bias.data_ptr()),
      slot_mapping.data_ptr<int64_t>(),
      reinterpret_cast<__nv_fp8_e4m3*>(kv_cache_fp8.data_ptr()),
      kv_cache_f32.data_ptr<float>(), num_k_tokens, index_q.stride(0),
      index_q.stride(1), index_q.stride(2), index_k.stride(0),
      index_k.stride(1), index_weights.stride(0), index_weights.stride(1),
      q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2),
      scaled_weights.stride(0), scaled_weights.stride(1), cache_block_size,
      cache_stride, static_cast<int32_t>(n_heads),
      static_cast<int32_t>(head_dim), static_cast<int32_t>(rope_dim),
      static_cast<float>(eps), static_cast<float>(factor));
  const cudaError_t result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess,
              "indexer_qk_rope_quant_cache kernel failed: ",
              cudaGetErrorString(result));
}
