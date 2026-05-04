# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Optimized ops for the flat Llama model at batch-size 1.

Strategy: call vLLM's existing optimized C++ ops directly with
pre-allocated buffers, use fused CUDA kernels where available
(silu_and_mul_nvfp4_quant), and use inductor-style Triton kernels
for fused_add_rms_norm that are autotuned for the target hardware.
"""

import torch
import triton
import triton.language as tl

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
# Inductor-style Triton kernels for fused_add_rms_norm
# (Adapted from torch.compile output — autotuned two-pass reduction)
# ---------------------------------------------------------------------------


@triton.jit
def _fused_add_rms_norm_triton(
    in_ptr0,   # hidden_states (bf16)
    in_ptr1,   # residual (bf16)
    in_ptr2,   # weight (bf16)
    out_ptr0,  # normed output (bf16)
    out_ptr1,  # updated residual (bf16)
    xnumel,
    r0_numel: tl.constexpr,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    """Fused residual-add + RMSNorm.

    new_residual = hidden_states + residual
    normed = rms_norm(new_residual) * weight
    """
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex

    # Pass 1: residual add + sum-of-squares
    _sum_sq = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        h = tl.load(
            in_ptr0 + (r0_1 + r0_numel * x0),
            r0_mask & xmask, other=0.0,
        ).to(tl.float32)
        r = tl.load(
            in_ptr1 + (r0_1 + r0_numel * x0),
            r0_mask & xmask, other=0.0,
        ).to(tl.float32)
        s = h + r
        sq = s * s
        _sum_sq = _sum_sq + tl.where(r0_mask & xmask, sq, 0.0)
    sum_sq = tl.sum(_sum_sq, 1)[:, None]

    # Pass 2: normalize + store residual and normed output
    rrms = tl.math.rsqrt(sum_sq / r0_numel + 1e-5)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        h = tl.load(
            in_ptr0 + (r0_1 + r0_numel * x0),
            r0_mask & xmask, other=0.0,
        ).to(tl.float32)
        r = tl.load(
            in_ptr1 + (r0_1 + r0_numel * x0),
            r0_mask & xmask, other=0.0,
        ).to(tl.float32)
        w = tl.load(
            in_ptr2 + r0_1, r0_mask, other=0.0,
        ).to(tl.float32)
        s = h + r
        normed = s * rrms * w
        tl.store(
            out_ptr0 + (r0_1 + r0_numel * x0), normed.to(tl.bfloat16),
            r0_mask & xmask,
        )
        tl.store(
            out_ptr1 + (r0_1 + r0_numel * x0), s.to(tl.bfloat16),
            r0_mask & xmask,
        )


def fused_add_rms_norm_triton(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    normed_out: torch.Tensor,
    residual_out: torch.Tensor,
):
    """Triton fused_add_rms_norm. Writes normed and residual to separate
    pre-allocated buffers (avoiding in-place mutation for CUDA graph compat)."""
    M = hidden_states.shape[0]
    N = hidden_states.shape[1]
    grid = (M,)
    _fused_add_rms_norm_triton[grid](
        hidden_states, residual, weight,
        normed_out, residual_out,
        M, r0_numel=N, XBLOCK=1, R0_BLOCK=min(N, 4096),
    )


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

        # Normed output (written by Triton norm kernel)
        self.normed = torch.empty(
            1, hidden_size, dtype=torch.bfloat16, device=device
        )
        # Residual buffer (written by Triton norm kernel)
        self.residual_buf = torch.empty(
            1, hidden_size, dtype=torch.bfloat16, device=device
        )

        # FP4 quant pre-allocated outputs
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

    Uses inductor-style Triton norm kernels, pre-allocated FP4 buffers,
    and the fused silu_and_mul_nvfp4_quant CUDA kernel.
    """
    # --- 1. Pre-attention LayerNorm (C++ fused_add_rms_norm, in-place) ---
    if residual is None:
        residual = hidden_states
        hidden_states = input_layernorm(hidden_states)
    else:
        hidden_states, residual = input_layernorm(hidden_states, residual)

    # --- 2+3. QKV projection ---
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

    # --- 9. Post-attention LayerNorm (C++ fused_add_rms_norm, in-place) ---
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
