# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Optimized ops for the flat Llama model at batch-size 1.

Strategy: call vLLM's existing optimized C++ ops directly with
pre-allocated buffers, and use fused CUDA kernels where available
(silu_and_mul_nvfp4_quant). This matches or exceeds the standard
model's performance without torch.compile.
"""

import torch

from vllm import _custom_ops as ops
from vllm._custom_ops import (
    create_fp4_output_tensors,
    cutlass_scaled_fp4_mm,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    NvFp4LinearBackend,
    pad_nvfp4_activation_for_cutlass,
    slice_nvfp4_output,
)
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm


# ---------------------------------------------------------------------------
# Pre-allocated decode buffers for one layer
# ---------------------------------------------------------------------------


class LayerDecodeBuffers:
    """Pre-allocated intermediate tensors for BS=1 decode in one layer."""

    def __init__(
        self,
        hidden_size: int,
        q_size: int,
        kv_size: int,
        intermediate_size_per_tp: int,
        device: torch.device,
    ):
        self.device = device
        self.hidden_size = hidden_size

        # FP4 quant pre-allocated outputs (avoids torch.empty per call)
        self.qkv_fp4, self.qkv_scale = create_fp4_output_tensors(
            1, hidden_size, device, is_sf_swizzled_layout=True
        )
        self.o_fp4, self.o_scale = create_fp4_output_tensors(
            1, q_size, device, is_sf_swizzled_layout=True
        )
        self.gate_up_fp4, self.gate_up_scale = create_fp4_output_tensors(
            1, hidden_size, device, is_sf_swizzled_layout=True
        )
        self.down_fp4, self.down_scale = create_fp4_output_tensors(
            1, intermediate_size_per_tp, device, is_sf_swizzled_layout=True
        )


# ---------------------------------------------------------------------------
# Optimized decode-step for one layer (BS=1)
# ---------------------------------------------------------------------------


def flat_decode_layer(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    # Layer sub-modules
    self_attn,
    mlp,
    input_layernorm,
    post_attention_layernorm,
    # Sizes
    q_size: int,
    kv_size: int,
    nvfp4_backend,
    tp_size: int,
    # Pre-allocated buffers
    bufs: LayerDecodeBuffers,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimized single-layer decode for BS=1.

    Calls vLLM's C++ ops directly with pre-allocated FP4 buffers.
    Uses the fused silu_and_mul_nvfp4_quant CUDA kernel.

    Per-layer ops (12 kernel launches at TP=1):
      1. fused_add_rms_norm          (C++ in-place)
      2. scaled_fp4_quant            (C++ pre-alloc output)
      3. cutlass_scaled_fp4_mm       (QKV GEMM)
      4. rotary_embedding            (C++ in-place)
      5. unified_kv_cache_update     (flash-attn)
      6. unified_attention            (flash-attn)
      7. scaled_fp4_quant            (C++ pre-alloc output)
      8. cutlass_scaled_fp4_mm       (O GEMM)
      9. fused_add_rms_norm          (C++ in-place)
     10. scaled_fp4_quant            (C++ pre-alloc output)
     11. cutlass_scaled_fp4_mm       (gate_up GEMM)
     12. silu_and_mul_nvfp4_quant    (FUSED: silu+mul+fp4_quant)
     13. cutlass_scaled_fp4_mm       (down GEMM)
    """
    eps = input_layernorm.variance_epsilon

    # --- 1. Pre-attention LayerNorm (C++ fused_add_rms_norm, in-place) ---
    if residual is None:
        residual = hidden_states
        hidden_states = input_layernorm(hidden_states)
    else:
        hidden_states, residual = input_layernorm(hidden_states, residual)

    # --- 2+3. QKV projection (FP4 quant + GEMM) ---
    qkv_proj = self_attn.qkv_proj
    torch.ops._C.scaled_fp4_quant.out(
        hidden_states,
        qkv_proj.input_global_scale_inv,
        True,
        output=bufs.qkv_fp4,
        output_scale=bufs.qkv_scale,
    )
    qkv = _nvfp4_gemm(
        bufs.qkv_fp4,
        bufs.qkv_scale.view(torch.float8_e4m3fn),
        qkv_proj,
        nvfp4_backend,
    )

    # --- 4. Split Q/K/V + RoPE ---
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q, k = self_attn.rotary_emb(positions, q, k)

    # --- 5+6. Attention ---
    attn_output = self_attn.attn(q, k, v)

    # --- 7+8. O projection ---
    o_proj = self_attn.o_proj
    torch.ops._C.scaled_fp4_quant.out(
        attn_output,
        o_proj.input_global_scale_inv,
        True,
        output=bufs.o_fp4,
        output_scale=bufs.o_scale,
    )
    hidden_states = _nvfp4_gemm(
        bufs.o_fp4,
        bufs.o_scale.view(torch.float8_e4m3fn),
        o_proj,
        nvfp4_backend,
    )
    if tp_size > 1:
        from vllm.distributed import tensor_model_parallel_all_reduce

        hidden_states = tensor_model_parallel_all_reduce(hidden_states)

    # --- 9. Post-attention LayerNorm ---
    hidden_states, residual = post_attention_layernorm(hidden_states, residual)

    # --- 10+11. Gate+Up projection ---
    gate_up_proj = mlp.gate_up_proj
    torch.ops._C.scaled_fp4_quant.out(
        hidden_states,
        gate_up_proj.input_global_scale_inv,
        True,
        output=bufs.gate_up_fp4,
        output_scale=bufs.gate_up_scale,
    )
    gate_up = _nvfp4_gemm(
        bufs.gate_up_fp4,
        bufs.gate_up_scale.view(torch.float8_e4m3fn),
        gate_up_proj,
        nvfp4_backend,
    )

    # --- 12+13. Fused SiLU+mul+FP4 quant + down projection ---
    down_proj = mlp.down_proj
    torch.ops._C.silu_and_mul_nvfp4_quant(
        bufs.down_fp4,
        bufs.down_scale,
        gate_up,
        down_proj.input_global_scale_inv,
    )
    hidden_states = _nvfp4_gemm(
        bufs.down_fp4,
        bufs.down_scale.view(torch.float8_e4m3fn),
        down_proj,
        nvfp4_backend,
    )
    if tp_size > 1:
        from vllm.distributed import tensor_model_parallel_all_reduce

        hidden_states = tensor_model_parallel_all_reduce(hidden_states)

    return hidden_states, residual


def _nvfp4_gemm(
    x_fp4: torch.Tensor,
    x_scale: torch.Tensor,
    layer: torch.nn.Module,
    backend: NvFp4LinearBackend,
) -> torch.Tensor:
    """NVFP4 GEMM with pre-quantized input. Minimal Python overhead."""
    weight = layer.weight
    weight_scale = layer.weight_scale
    alpha = layer.alpha
    output_size = layer.output_size_per_partition
    weights_padding_cols = getattr(layer, "weights_padding_cols", 0)

    x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_cols)

    mm_args = (x_fp4, weight, x_scale, weight_scale, alpha, torch.bfloat16)

    if backend.value.startswith("flashinfer-"):
        backend_name = backend.value[len("flashinfer-"):]
        out = flashinfer_scaled_fp4_mm(*mm_args, backend=backend_name)
    elif backend == NvFp4LinearBackend.FBGEMM:
        out = torch.ops.fbgemm.f4f4bf16(
            x_fp4,
            weight,
            x_scale.view(-1).view(torch.uint8),
            weight_scale,
            alpha,
            use_mx=False,
        ).to(torch.bfloat16)
    else:
        out = cutlass_scaled_fp4_mm(*mm_args)

    return slice_nvfp4_output(out, output_size)
