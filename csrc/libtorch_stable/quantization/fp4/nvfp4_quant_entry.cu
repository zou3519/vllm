/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/csrc/stable/tensor.h>

#include "libtorch_stable/torch_utils.h"

#include "cutlass_extensions/common.hpp"
#include "nvfp4_utils.cuh"

#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
void scaled_fp4_quant_sm1xxa(torch::stable::Tensor const& output,
                             torch::stable::Tensor const& input,
                             torch::stable::Tensor const& output_sf,
                             torch::stable::Tensor const& input_sf,
                             bool is_sf_swizzled_layout);
void scaled_fp4_quant_dual_8x4_128x4_sm1xxa(
    torch::stable::Tensor const& output8x4,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& output_sf8x4,
    torch::stable::Tensor const& input_sf8x4,
    torch::stable::Tensor const& output128x4,
    torch::stable::Tensor const& output_sf128x4,
    torch::stable::Tensor const& input_sf128x4);
#endif

#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
void scaled_fp4_experts_quant_sm1xxa(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_global_scale,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts);
#endif

#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
void silu_and_mul_nvfp4_quant_sm1xxa(torch::stable::Tensor& output,
                                     torch::stable::Tensor& output_sf,
                                     torch::stable::Tensor& input,
                                     torch::stable::Tensor& input_sf);
#endif

#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
void silu_and_mul_scaled_fp4_experts_quant_sm1xxa(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_global_scale,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts);
#endif

#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
static bool nvfp4_quant_sm_supported() {
  const int32_t sm = get_sm_version_num();
  #if defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100
  if (sm >= 100 && sm < 120) return true;
  #endif
  #if defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120
  if (sm >= 120 && sm < 130) return true;
  #endif
  return false;
}
#endif

void scaled_fp4_quant_out(torch::stable::Tensor const& input,
                          torch::stable::Tensor const& input_sf,
                          bool is_sf_swizzled_layout,
                          torch::stable::Tensor& output,
                          torch::stable::Tensor& output_sf) {
#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
  STD_TORCH_CHECK(nvfp4_quant_sm_supported(),
                  "No compiled nvfp4 quantization kernel for SM ",
                  get_sm_version_num(),
                  ". Recompile with the appropriate CUDA arch.");
  return scaled_fp4_quant_sm1xxa(output, input, output_sf, input_sf,
                                 is_sf_swizzled_layout);
#endif
  STD_TORCH_CHECK_NOT_IMPLEMENTED(false,
                                  "No compiled nvfp4 quantization kernel");
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor> scaled_fp4_quant_func(
    torch::stable::Tensor const& input, torch::stable::Tensor const& input_sf,
    bool is_sf_swizzled_layout) {
  int64_t n = input.size(-1);
  int64_t m = input.numel() / n;
  auto device = input.device();

  // Two fp4 values packed into a uint8
  auto output = torch::stable::empty(
      {m, n / 2}, torch::headeronly::ScalarType::Byte, std::nullopt, device);

  torch::stable::Tensor output_sf;
  if (is_sf_swizzled_layout) {
    auto [sf_m, sf_n] = vllm::computeSwizzledSFShape(m, n);
    output_sf = torch::stable::empty(
        {sf_m, sf_n}, torch::headeronly::ScalarType::Int, std::nullopt, device);
  } else {
    output_sf = torch::stable::empty({m, n / CVT_FP4_SF_VEC_SIZE},
                                     torch::headeronly::ScalarType::Byte,
                                     std::nullopt, device);
  }

  scaled_fp4_quant_out(input, input_sf, is_sf_swizzled_layout, output,
                       output_sf);
  return {output, output_sf};
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor,
           torch::stable::Tensor>
scaled_fp4_quant_dual_8x4_128x4_func(
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_sf8x4,
    torch::stable::Tensor const& input_sf128x4) {
  int64_t n = input.size(-1);
  int64_t m = input.numel() / n;
  auto device = input.device();

  auto output8x4 = torch::stable::empty(
      {m, n / 2}, torch::headeronly::ScalarType::Byte, std::nullopt, device);
  auto [sf8x4_m, sf8x4_n] = vllm::computeSwizzledSFShape8x4(m, n);
  auto output_sf8x4 =
      torch::stable::empty({sf8x4_m, sf8x4_n},
                           torch::headeronly::ScalarType::Byte, std::nullopt,
                           device);

  auto output128x4 = torch::stable::empty(
      {m, n / 2}, torch::headeronly::ScalarType::Byte, std::nullopt, device);
  auto [sf128x4_m, sf128x4_n] = vllm::computeSwizzledSFShape(m, n);
  auto output_sf128x4 =
      torch::stable::empty({sf128x4_m, sf128x4_n},
                           torch::headeronly::ScalarType::Int, std::nullopt,
                           device);

  scaled_fp4_quant_dual_8x4_128x4_sm1xxa(
      output8x4, input, output_sf8x4, input_sf8x4, output128x4,
      output_sf128x4, input_sf128x4);
  return {output8x4, output_sf8x4, output128x4, output_sf128x4};
}

void scaled_fp4_experts_quant(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_global_scale,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts) {
#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
  STD_TORCH_CHECK(nvfp4_quant_sm_supported(),
                  "No compiled nvfp4 experts quantization kernel for SM ",
                  get_sm_version_num(),
                  ". Recompile with the appropriate CUDA arch.");
  return scaled_fp4_experts_quant_sm1xxa(
      output, output_scale, input, input_global_scale, input_offset_by_experts,
      output_scale_offset_by_experts);
#endif
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false, "No compiled nvfp4 experts quantization kernel");
}

void silu_and_mul_nvfp4_quant(torch::stable::Tensor& output,
                              torch::stable::Tensor& output_sf,
                              torch::stable::Tensor& input,
                              torch::stable::Tensor& input_sf) {
#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
  STD_TORCH_CHECK(nvfp4_quant_sm_supported(),
                  "No compiled silu_and_mul nvfp4 quantization kernel for SM ",
                  get_sm_version_num(),
                  ". Recompile with the appropriate CUDA arch.");
  return silu_and_mul_nvfp4_quant_sm1xxa(output, output_sf, input, input_sf);
#endif
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false, "No compiled silu_and_mul nvfp4 quantization kernel");
}

void silu_and_mul_scaled_fp4_experts_quant(
    torch::stable::Tensor& output, torch::stable::Tensor& output_scale,
    torch::stable::Tensor const& input,
    torch::stable::Tensor const& input_global_scale,
    torch::stable::Tensor const& input_offset_by_experts,
    torch::stable::Tensor const& output_scale_offset_by_experts) {
#if (defined(ENABLE_NVFP4_SM100) && ENABLE_NVFP4_SM100) || \
    (defined(ENABLE_NVFP4_SM120) && ENABLE_NVFP4_SM120)
  STD_TORCH_CHECK(nvfp4_quant_sm_supported(),
                  "No compiled silu_and_mul nvfp4 experts quantization kernel "
                  "for SM ",
                  get_sm_version_num(),
                  ". Recompile with the appropriate CUDA arch.");
  return silu_and_mul_scaled_fp4_experts_quant_sm1xxa(
      output, output_scale, input, input_global_scale, input_offset_by_experts,
      output_scale_offset_by_experts);
#endif
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false, "No compiled silu_and_mul nvfp4 experts quantization kernel");
}
