# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Optimized ops for the flat Llama model at batch-size 1.

Strategy: pre-allocate all intermediate tensors and call existing C++ ops
directly, eliminating per-call tensor allocation and Python wrapper overhead.
"""

import torch
import triton
import triton.language as tl

from vllm import _custom_ops as ops
from vllm._custom_ops import (
    create_fp4_output_tensors,
    cutlass_scaled_fp4_mm,
    scaled_fp4_quant,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    NvFp4LinearBackend,
    pad_nvfp4_activation_for_cutlass,
    slice_nvfp4_output,
)
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm


# ---------------------------------------------------------------------------
# Triton: fused residual-add + RMSNorm (avoids Python dispatch of RMSNorm)
# ---------------------------------------------------------------------------


@triton.jit
def _fused_add_rms_norm_kernel(
    x_ptr, residual_ptr, out_ptr, weight_ptr,
    eps,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Single-row fused residual-add + RMSNorm.

    Writes updated residual in-place, writes normed output to out_ptr.
    Avoids the overhead of the RMSNorm nn.Module dispatch.
    """
    sum_sq = tl.zeros([1], dtype=tl.float32)

    # Pass 1: residual add + sum of squares
    for start in tl.static_range(0, BLOCK, 1024):
        cols = start + tl.arange(0, 1024)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(residual_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        s = x + res
        tl.store(residual_ptr + cols, s.to(tl.bfloat16), mask=mask)
        sum_sq += tl.sum(s * s)

    rrms = tl.math.rsqrt(sum_sq / N + eps)

    # Pass 2: apply norm
    for start in tl.static_range(0, BLOCK, 1024):
        cols = start + tl.arange(0, 1024)
        mask = cols < N
        s = tl.load(residual_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normed = s * rrms * w
        tl.store(out_ptr + cols, normed.to(tl.bfloat16), mask=mask)


@triton.jit
def _rms_norm_kernel(
    x_ptr, out_ptr, weight_ptr,
    eps,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Single-row RMSNorm without residual add (for first layer)."""
    sum_sq = tl.zeros([1], dtype=tl.float32)

    for start in tl.static_range(0, BLOCK, 1024):
        cols = start + tl.arange(0, 1024)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x)

    rrms = tl.math.rsqrt(sum_sq / N + eps)

    for start in tl.static_range(0, BLOCK, 1024):
        cols = start + tl.arange(0, 1024)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normed = x * rrms * w
        tl.store(out_ptr + cols, normed.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Triton: fused SiLU-and-Mul (avoids custom op dispatch overhead)
# ---------------------------------------------------------------------------


@triton.jit
def _fused_silu_mul_kernel(
    gate_up_ptr, out_ptr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """SiLU(gate) * up for a single row. gate_up is [1, 2*D]."""
    for start in tl.static_range(0, BLOCK, 1024):
        cols = start + tl.arange(0, 1024)
        mask = cols < D
        gate = tl.load(gate_up_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(gate_up_ptr + D + cols, mask=mask, other=0.0).to(tl.float32)
        act = gate * tl.sigmoid(gate) * up
        tl.store(out_ptr + cols, act.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Pre-allocated decode buffers for one layer
# ---------------------------------------------------------------------------


class LayerDecodeBuffers:
    """Pre-allocated intermediate tensors for BS=1 decode in one layer.

    Eliminates ~10 torch.empty calls per layer per decode step.
    """

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

        # Normed hidden states (output of RMSNorm)
        self.normed = torch.empty(1, hidden_size, dtype=torch.bfloat16, device=device)

        # QKV output
        qkv_size = q_size + 2 * kv_size
        self.qkv_out = torch.empty(1, qkv_size, dtype=torch.bfloat16, device=device)

        # Gate+Up output
        self.gate_up_out = torch.empty(
            1, intermediate_size_per_tp * 2, dtype=torch.bfloat16, device=device
        )

        # SiLU output
        self.silu_out = torch.empty(
            1, intermediate_size_per_tp, dtype=torch.bfloat16, device=device
        )

        # FP4 quant outputs (for each of the 4 linear layers)
        # Pre-allocate the output tensors for scaled_fp4_quant.out
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
# Round-up helper for Triton BLOCK size
# ---------------------------------------------------------------------------


def _next_multiple_of_1024(n: int) -> int:
    return ((n + 1023) // 1024) * 1024


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

    Key optimizations vs the generic flat_decoder_layer_forward:
    1. Pre-allocated intermediate tensors (no torch.empty per call)
    2. Triton fused RMSNorm (avoids nn.Module dispatch)
    3. Direct C++ op calls with pre-allocated outputs (scaled_fp4_quant.out)
    4. Triton fused SiLU-and-mul (avoids custom op dispatch)
    """
    from vllm.distributed import tensor_model_parallel_all_reduce

    eps = input_layernorm.variance_epsilon
    N = bufs.hidden_size
    BLOCK = _next_multiple_of_1024(N)

    # --- 1. Pre-attention LayerNorm ---
    if residual is None:
        residual = hidden_states
        _rms_norm_kernel[(1,)](
            hidden_states, bufs.normed, input_layernorm.weight.data,
            eps, N=N, BLOCK=BLOCK,
        )
    else:
        _fused_add_rms_norm_kernel[(1,)](
            hidden_states, residual, bufs.normed,
            input_layernorm.weight.data,
            eps, N=N, BLOCK=BLOCK,
        )

    # --- 2. QKV projection (FP4 quant + GEMM) ---
    qkv_proj = self_attn.qkv_proj
    torch.ops._C.scaled_fp4_quant.out(
        bufs.normed,
        qkv_proj.input_global_scale_inv,
        True,  # is_sf_swizzled_layout
        output=bufs.qkv_fp4,
        output_scale=bufs.qkv_scale,
    )
    qkv = _nvfp4_gemm(
        bufs.qkv_fp4,
        bufs.qkv_scale.view(torch.float8_e4m3fn),
        qkv_proj,
        nvfp4_backend,
    )

    # --- 3. Split Q/K/V + RoPE ---
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q, k = self_attn.rotary_emb(positions, q, k)

    # --- 4. Attention ---
    attn_output = self_attn.attn(q, k, v)

    # --- 5. O projection + TP reduce ---
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
        hidden_states = tensor_model_parallel_all_reduce(hidden_states)

    # --- 6. Post-attention LayerNorm ---
    _fused_add_rms_norm_kernel[(1,)](
        hidden_states, residual, bufs.normed,
        post_attention_layernorm.weight.data,
        eps, N=N, BLOCK=BLOCK,
    )

    # --- 7. Gate+Up projection ---
    gate_up_proj = mlp.gate_up_proj
    torch.ops._C.scaled_fp4_quant.out(
        bufs.normed,
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

    # --- 8. SiLU-and-mul ---
    D = gate_up.shape[-1] // 2
    D_BLOCK = _next_multiple_of_1024(D)
    _fused_silu_mul_kernel[(1,)](
        gate_up, bufs.silu_out,
        D=D, BLOCK=D_BLOCK,
    )

    # --- 9. Down projection + TP reduce ---
    down_proj = mlp.down_proj
    torch.ops._C.scaled_fp4_quant.out(
        bufs.silu_out,
        down_proj.input_global_scale_inv,
        True,
        output=bufs.down_fp4,
        output_scale=bufs.down_scale,
    )
    hidden_states = _nvfp4_gemm(
        bufs.down_fp4,
        bufs.down_scale.view(torch.float8_e4m3fn),
        down_proj,
        nvfp4_backend,
    )
    if tp_size > 1:
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
            x_fp4, weight,
            x_scale.view(-1).view(torch.uint8),
            weight_scale, alpha, use_mx=False,
        ).to(torch.bfloat16)
    else:
        out = cutlass_scaled_fp4_mm(*mm_args)

    return slice_nvfp4_output(out, output_size)
