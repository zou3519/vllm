# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Flat, function-level forward pass for Llama NVFP4.

The entire model forward is two functions:
  - flat_forward(): embedding → loop of transformer_layer → final norm
  - transformer_layer(): one decoder block with all params passed explicitly

No nn.Module dispatch in the hot path. All weights, buffers, and op handles
are passed as plain tensors / callables.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

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
# Triton fused kernels: residual-add + RMSNorm + FP4 quant (2 launches vs 3)
# Uses inline PTX cvt.rn.satfinite.e2m1x2.f32 for native E2M1 conversion.
# ---------------------------------------------------------------------------


@triton.jit
def _add_variance_kernel(
    hidden_ptr, residual_ptr, residual_out_ptr, variance_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    off = row * N
    _sum_sq = 0.0
    for start in tl.static_range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        h = tl.load(hidden_ptr + off + cols).to(tl.float32)
        r = tl.load(residual_ptr + off + cols).to(tl.float32)
        s = h + r
        tl.store(residual_out_ptr + off + cols, s.to(tl.bfloat16))
        _sum_sq += tl.sum(s * s)
    tl.store(variance_ptr + row, _sum_sq / N)


@triton.jit
def _norm_fp4_quant_kernel(
    residual_ptr, weight_ptr, variance_ptr, sf_scale_ptr,
    fp4_out_ptr, scale_out_ptr,
    N: tl.constexpr, SCALE_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_groups_per_row = N // 16
    row = pid // num_groups_per_row
    g = pid % num_groups_per_row

    variance = tl.load(variance_ptr + row)
    rrms = tl.math.rsqrt(variance + 1e-5)
    sf_scale = tl.load(sf_scale_ptr).to(tl.float32)

    base = row * N + g * 16
    pair_idx = tl.arange(0, 8)
    even_offs = base + pair_idx * 2
    odd_offs = even_offs + 1
    w_even_offs = g * 16 + pair_idx * 2
    w_odd_offs = w_even_offs + 1

    even_res = tl.load(residual_ptr + even_offs).to(tl.float32)
    odd_res = tl.load(residual_ptr + odd_offs).to(tl.float32)
    even_w = tl.load(weight_ptr + w_even_offs).to(tl.float32)
    odd_w = tl.load(weight_ptr + w_odd_offs).to(tl.float32)

    even_normed = even_res * rrms * even_w
    odd_normed = odd_res * rrms * odd_w

    block_max = tl.maximum(
        tl.max(tl.abs(even_normed)), tl.max(tl.abs(odd_normed))
    )
    sf_val = sf_scale * (block_max / 6.0)
    sf_fp8 = sf_val.to(tl.float8e4nv)
    sf_f32 = sf_fp8.to(tl.float32)
    quant_scale = tl.where(sf_f32 > 0.0, sf_scale / sf_f32, 0.0)

    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1;"
        " cvt.u16.u8 $0, tmp; }",
        "=h, r, r",
        [even_normed * quant_scale, odd_normed * quant_scale],
        dtype=tl.int16,
        is_pure=True,
        pack=1,
    )

    fp4_base = row * (N // 2) + g * 8
    tl.store(fp4_out_ptr + fp4_base + pair_idx, packed.to(tl.uint8))

    kTileIdx = g // 4
    innerKIdx = g % 4
    byte_offset = row * SCALE_STRIDE + kTileIdx * SCALE_STRIDE + innerKIdx
    tl.store(scale_out_ptr + byte_offset, sf_fp8.to(tl.uint8, bitcast=True))


# ---------------------------------------------------------------------------
# Data structures — hold extracted params for one NVFP4 linear projection
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class NvFp4Proj:
    """All tensors needed to run one NVFP4 linear projection."""
    weight: torch.Tensor           # [out, in/2] uint8
    weight_scale: torch.Tensor     # fp8 block scales (swizzled)
    alpha: torch.Tensor            # input_global_scale * weight_global_scale
    input_scale_inv: torch.Tensor  # 1 / input_global_scale
    weights_padding: int           # K-dimension padding bytes
    output_size: int               # unpadded output dim


@dataclass(slots=True)
class SharedDecodeBuffers:
    """One set of pre-allocated FP4 buffers shared across all layers."""
    qkv_fp4: torch.Tensor
    qkv_scale: torch.Tensor
    o_fp4: torch.Tensor
    o_scale: torch.Tensor
    gu_fp4: torch.Tensor
    gu_scale: torch.Tensor
    down_fp4: torch.Tensor
    down_scale: torch.Tensor

    # Intermediate buffers for fused norm+quant
    residual_buf: torch.Tensor
    variance: torch.Tensor

    @staticmethod
    def create(
        hidden_size: int,
        q_size: int,
        intermediate_size: int,
        device: torch.device,
    ) -> "SharedDecodeBuffers":
        qkv_fp4, qkv_sc = create_fp4_output_tensors(1, hidden_size, device, True)
        o_fp4, o_sc = create_fp4_output_tensors(1, q_size, device, True)
        gu_fp4, gu_sc = create_fp4_output_tensors(1, hidden_size, device, True)
        d_fp4, d_sc = create_fp4_output_tensors(1, intermediate_size, device, True)
        return SharedDecodeBuffers(
            qkv_fp4, qkv_sc, o_fp4, o_sc, gu_fp4, gu_sc, d_fp4, d_sc,
            residual_buf=torch.empty(1, hidden_size, dtype=torch.bfloat16, device=device),
            variance=torch.empty(1, dtype=torch.float32, device=device),
        )


# ---------------------------------------------------------------------------
# Core: nvfp4_gemm — GEMM with pre-quantized FP4 input
# ---------------------------------------------------------------------------


def nvfp4_gemm(
    x_fp4: torch.Tensor,
    x_scale: torch.Tensor,
    proj: NvFp4Proj,
    backend: NvFp4LinearBackend,
) -> torch.Tensor:
    xp = pad_nvfp4_activation_for_cutlass(x_fp4, proj.weights_padding)
    args = (xp, proj.weight, x_scale, proj.weight_scale, proj.alpha, torch.bfloat16)
    if backend.value.startswith("flashinfer-"):
        out = flashinfer_scaled_fp4_mm(*args, backend=backend.value[len("flashinfer-"):])
    elif backend == NvFp4LinearBackend.FBGEMM:
        out = torch.ops.fbgemm.f4f4bf16(
            xp, proj.weight, x_scale.view(-1).view(torch.uint8),
            proj.weight_scale, proj.alpha, use_mx=False,
        ).to(torch.bfloat16)
    else:
        out = cutlass_scaled_fp4_mm(*args)
    return slice_nvfp4_output(out, proj.output_size)


# ---------------------------------------------------------------------------
# Core: transformer_layer — one decoder block, all params explicit
# ---------------------------------------------------------------------------


def transformer_layer(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    # --- LayerNorm weights ---
    input_ln_w: torch.Tensor,
    post_attn_ln_w: torch.Tensor,
    eps: float,
    # --- NVFP4 projections ---
    qkv: NvFp4Proj,
    o: NvFp4Proj,
    gate_up: NvFp4Proj,
    down: NvFp4Proj,
    # --- Attention + RoPE (opaque callables) ---
    rotary_emb,      # callable(positions, q, k) -> (q, k)
    attn,            # callable(q, k, v) -> output
    # --- Sizes ---
    q_size: int,
    kv_size: int,
    # --- Pre-allocated FP4 buffers ---
    bufs: SharedDecodeBuffers,
    # --- Backend ---
    backend: NvFp4LinearBackend,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One decoder layer. Every input is an explicit parameter — no hidden state.

    With fused norm+quant kernels: 11 kernel launches (down from 13):
      1. _add_variance_kernel (residual add + variance)
      2. _norm_fp4_quant_kernel (normalize + E2M1 quantize via PTX)
      3. GEMM (QKV)
      4. RoPE
      5. KV cache write
      6. Attention
      7. scaled_fp4_quant (O input — not fused, different input size)
      8. GEMM (O)
      9-10. fused norm+quant (post-attention: 2 kernels)
      11. GEMM (gate_up)
      12. silu_and_mul_nvfp4_quant (fused)
      13. GEMM (down)
    = 11 unique kernel launches (steps 1+2 and 9+10 each replace 3 separate ops)
    """
    N = hidden_states.shape[-1]
    M = hidden_states.shape[0]
    scale_stride = bufs.qkv_scale.view(torch.uint8).shape[-1]

    # 1+2. Fused pre-attention norm + FP4 quant (2 kernels instead of 3)
    if residual is None:
        residual = hidden_states
        from vllm.model_executor.layers.layernorm import ir
        hidden_states = ir.ops.rms_norm(hidden_states, input_ln_w, eps)
        torch.ops._C.scaled_fp4_quant.out(
            hidden_states, qkv.input_scale_inv, True,
            output=bufs.qkv_fp4, output_scale=bufs.qkv_scale,
        )
    else:
        _add_variance_kernel[(M,)](
            hidden_states, residual, bufs.residual_buf, bufs.variance,
            N=N, BLOCK=min(N, 4096),
        )
        _norm_fp4_quant_kernel[(M * (N // 16),)](
            bufs.residual_buf, input_ln_w, bufs.variance,
            qkv.input_scale_inv,
            bufs.qkv_fp4, bufs.qkv_scale.view(torch.uint8),
            N=N, SCALE_STRIDE=scale_stride,
        )
        residual = bufs.residual_buf

    # 3. QKV GEMM
    qkv_out = nvfp4_gemm(
        bufs.qkv_fp4, bufs.qkv_scale.view(torch.float8_e4m3fn), qkv, backend,
    )

    # 3. Split Q/K/V + RoPE
    q, k, v = qkv_out.split([q_size, kv_size, kv_size], dim=-1)
    q, k = rotary_emb(positions, q, k)

    # 4. Attention (KV cache write + compute)
    attn_output = attn(q, k, v)

    # 5. O projection
    torch.ops._C.scaled_fp4_quant.out(
        attn_output, o.input_scale_inv, True,
        output=bufs.o_fp4, output_scale=bufs.o_scale,
    )
    hidden_states = nvfp4_gemm(
        bufs.o_fp4, bufs.o_scale.view(torch.float8_e4m3fn), o, backend,
    )

    # 6+7. Fused post-attention norm + FP4 quant (2 kernels instead of 3)
    _add_variance_kernel[(M,)](
        hidden_states, residual, bufs.residual_buf, bufs.variance,
        N=N, BLOCK=min(N, 4096),
    )
    _norm_fp4_quant_kernel[(M * (N // 16),)](
        bufs.residual_buf, post_attn_ln_w, bufs.variance,
        gate_up.input_scale_inv,
        bufs.gu_fp4, bufs.gu_scale.view(torch.uint8),
        N=N, SCALE_STRIDE=scale_stride,
    )
    residual = bufs.residual_buf

    # 8. Gate+Up GEMM
    gate_up_out = nvfp4_gemm(
        bufs.gu_fp4, bufs.gu_scale.view(torch.float8_e4m3fn), gate_up, backend,
    )

    # 8+9. Fused SiLU+mul+FP4 quant → down projection
    torch.ops._C.silu_and_mul_nvfp4_quant(
        bufs.down_fp4, bufs.down_scale, gate_up_out, down.input_scale_inv,
    )
    hidden_states = nvfp4_gemm(
        bufs.down_fp4, bufs.down_scale.view(torch.float8_e4m3fn), down, backend,
    )

    return hidden_states, residual


# ---------------------------------------------------------------------------
# Core: flat_forward — the entire model as one function
# ---------------------------------------------------------------------------


def flat_forward(
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    # --- Embedding ---
    embed_fn,           # callable(input_ids) -> hidden_states, or None
    # --- Per-layer params (lists of length num_layers) ---
    input_ln_weights: list[torch.Tensor],
    post_attn_ln_weights: list[torch.Tensor],
    eps: float,
    qkv_projs: list[NvFp4Proj],
    o_projs: list[NvFp4Proj],
    gate_up_projs: list[NvFp4Proj],
    down_projs: list[NvFp4Proj],
    rotary_embs: list,   # list of callable
    attns: list,          # list of callable (Attention layers)
    q_size: int,
    kv_size: int,
    # --- Final norm ---
    final_norm_w: torch.Tensor,
    # --- Shared buffers ---
    bufs: SharedDecodeBuffers,
    # --- Backend ---
    backend: NvFp4LinearBackend,
    # --- Layer range ---
    start_layer: int,
    end_layer: int,
    # --- Optional: pre-computed hidden states (skip embedding) ---
    hidden_states_in: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full model forward: embedding → N × transformer_layer → final norm.

    Every parameter is passed explicitly. No nn.Module attribute access.
    """
    if hidden_states_in is not None:
        hidden_states = hidden_states_in
    else:
        hidden_states = embed_fn(input_ids)
    residual = None

    for i in range(start_layer, end_layer):
        hidden_states, residual = transformer_layer(
            positions, hidden_states, residual,
            input_ln_weights[i], post_attn_ln_weights[i], eps,
            qkv_projs[i], o_projs[i], gate_up_projs[i], down_projs[i],
            rotary_embs[i], attns[i],
            q_size, kv_size,
            bufs, backend,
        )

    # Final norm
    ops.fused_add_rms_norm(hidden_states, residual, final_norm_w, eps)
    return hidden_states


# ---------------------------------------------------------------------------
# Param extraction — pull all weights out of nn.Module into flat lists
# ---------------------------------------------------------------------------


def extract_nvfp4_proj(linear_module: torch.nn.Module) -> NvFp4Proj:
    """Extract NVFP4 params from a vLLM ColumnParallelLinear/RowParallelLinear."""
    return NvFp4Proj(
        weight=linear_module.weight,
        weight_scale=linear_module.weight_scale,
        alpha=linear_module.alpha,
        input_scale_inv=linear_module.input_global_scale_inv,
        weights_padding=getattr(linear_module, "weights_padding_cols", 0),
        output_size=linear_module.output_size_per_partition,
    )


def extract_all_layer_params(layers, start_layer, end_layer):
    """Extract flat param lists from nn.Module decoder layers."""
    input_ln_weights = []
    post_attn_ln_weights = []
    qkv_projs = []
    o_projs = []
    gate_up_projs = []
    down_projs = []
    rotary_embs = []
    attns = []

    for i in range(start_layer, end_layer):
        layer = layers[i]
        input_ln_weights.append(layer.input_layernorm.weight.data)
        post_attn_ln_weights.append(layer.post_attention_layernorm.weight.data)
        qkv_projs.append(extract_nvfp4_proj(layer.self_attn.qkv_proj))
        o_projs.append(extract_nvfp4_proj(layer.self_attn.o_proj))
        gate_up_projs.append(extract_nvfp4_proj(layer.mlp.gate_up_proj))
        down_projs.append(extract_nvfp4_proj(layer.mlp.down_proj))
        rotary_embs.append(layer.self_attn.rotary_emb)
        attns.append(layer.self_attn.attn)

    return (
        input_ln_weights, post_attn_ln_weights,
        qkv_projs, o_projs, gate_up_projs, down_projs,
        rotary_embs, attns,
    )
