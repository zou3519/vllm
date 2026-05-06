# Flat Llama — Optimized NVFP4 Llama for Low-Latency Decode

## Summary

A hand-optimized "flat" model for `nvidia/Llama-3.3-70B-Instruct-NVFP4`
on GB300 (TP=1, BS=1). Bypasses vLLM's nn.Module dispatch — the forward
pass is a single function with all parameters as plain tensors.

**Current results (verified correct):**
```
             TPIT       Speedup
Compiled:    12.50ms    baseline
Flat+GEMV:    9.52ms    1.31×
```

## Maintenance

When making changes, **always sanity-check the whole README** — per-layer
breakdown, bandwidth analysis, architecture diagram, and optimization
history can all go stale. **Commit immediately and update this log** when
a new discovery improves performance.

## How to run

```bash
CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
vllm serve nvidia/Llama-3.3-70B-Instruct-NVFP4 \
    --hf-overrides '{"architectures": ["FlatLlamaForCausalLM"]}' \
    --compilation-config '{"mode": "none", "cudagraph_mode": "full"}'
```

## Model architecture (unfused)

Without any fusions, the standard Llama decode layer is:

```
1. Residual add
2. RMSNorm
3. FP4 quantize (for QKV input)
4. QKV GEMM
5. RoPE (Q, K)
6. KV cache write
7. Attention decode
8. Merge attention states
9. FP4 quantize (for O input)
10. O GEMM
11. Residual add
12. RMSNorm
13. FP4 quantize (for Gate+Up input)
14. Gate+Up GEMM
15. SiLU activation
16. Element-wise multiply (gate × up)
17. FP4 quantize (for Down input)
18. Down GEMM
```

That's 18 ops and ~13 kernel launches per layer.

## What was fused/optimized

| Fusion | Ops merged | Kernels saved | Method |
|---|---|---|---|
| Residual+RMSNorm+FP4 quant | 1+2+3, 11+12+13 | 2→1 (×2 sites) | Single-CTA Triton kernel |
| RoPE+KV cache write | 5+6 | 2→1 | Triton kernel (64 programs) |
| O GEMM → BF16-GEMV | 9+10 | 2→1 | Raw CUDA BF16-input GEMV (no FP4 quant) |
| SiLU+mul+FP4 quant | 15+16+17 | 3→1 | Triton kernel (row-major scales) |
| Down GEMM → GEMV | 18 | same count | Raw CUDA (22.8μs vs 48μs CUTLASS) |

Result: **10 kernels per layer** (down from ~13).

## Per-layer decode breakdown

Measured via standalone kernel benchmarks, then validated against e2e TPIT.

```
 #  Op                               μs     Pct
────────────────────────────────────────────────────
 1  Norm+FP4 quant (pre-attn)        4.2    4.8%   fused (Triton single-CTA)
 2  QKV GEMM (CUTLASS)               9.0   10.3%
 3  RoPE+KV write (fused)            2.5    2.9%   fused (Triton)
 4  Attention decode (FlashInfer)     9.3   10.6%
 5  Merge states (FlashInfer)         2.3    2.6%
 6  O BF16-GEMV (CUDA)               ~5    5.7%   was FP4 quant+CUTLASS (10.9μs)
 7  Norm+FP4 quant (post-attn)       4.2    4.8%   fused (Triton single-CTA)
 8  Gate+Up GEMM (CUTLASS)          46.0   52.5%
 9  Fused silu+mul+FP4 quant         1.8    2.1%   fused (Triton, row-major)
10  Down GEMV (CUDA)                22.8   26.0%   was 48μs CUTLASS
────────────────────────────────────────────────────
    TOTAL per layer                ~87.6
    80 layers                        7.0ms
    + overhead                      ~2.5ms
    ≈ TPIT                          9.5ms  (measured: 9.52ms)
```

## Known issues and investigations

### FP8 KV cache dtype mismatch (FIXED)
The fused RoPE+KV kernel checked `kv_cache.dtype == torch.float8_e4m3fn`
but vLLM/FlashInfer allocates FP8 cache as `torch.uint8`. Fixed by checking
for both types. Without this fix, the kernel wrote BF16 (2 bytes) into the
uint8 cache (1 byte per element), corrupting all subsequent attention.

### FP4-input GEMV for QKV/Gate+Up (NOT VIABLE — numerical fragility)
FP4-input GEMV (where activations are quantized to FP4 then decompressed
in the GEMV kernel) breaks model quality for QKV and Gate+Up projections,
even when the GEMV kernel produces output within 0.008 of CUTLASS on
identical FP4 data.

**Root cause 1 — Triton compiler FP4 data divergence**: The Triton fused
norm+quant kernel compiled with `ROW_MAJOR_SCALES=True` produces subtly
different FP4 quantization than with `ROW_MAJOR_SCALES=False`, due to
different compiler optimization choices affecting FP32 intermediate rounding.
Fix: always quantize with the CUTLASS-compatible kernel and unswizzle
activation scales for GEMV.

**Root cause 2 — MMA vs FMA rounding**: Even with identical FP4 data,
the GEMV (scalar FMA in FP16/FP32) produces ~0.008 max diff vs CUTLASS
(tensor core MMA). This error compounds across 80 layers through attention.
The model was calibrated for CUTLASS MMA rounding behavior.

### BF16-input GEMV for O projection (WORKS)
The BF16-input GEMV bypasses FP4 activation quantization entirely, feeding
BF16 attention output directly to the GEMV with FP4 weights. This avoids
both root causes above: no FP4 quant means no Triton compiler divergence,
and the higher-precision input compensates for MMA/FMA differences. Saves
~6μs per layer (eliminates FP4 quant kernel + faster than CUTLASS GEMM).

### Prefill kv_sharing bug (FIXED)
The `else` branch in `flat_forward` (general/prefill path) incorrectly set
`kv_sharing_target_layer_name = layer_name` instead of `None`, telling
attention to skip KV cache writes during prefill.

## Files

- `flat_llama.py` — model class (nn.Module for weight loading)
- `flat_llama_kernels.py` — `flat_forward()`, `transformer_layer()`, Triton kernels
- `flat_llama_gemv.py` — NVFP4 GEMV kernel (raw CUDA via load_inline)
- `flat_llama_triton_gemv.py` — experimental Triton GEMV (for benchmarking)
- `flat_llama_cute_kernels.py` — CuTe DSL kernels (deprecated)

## Optimization history

| Change | TPIT | Commit |
|---|---|---|
| Flat model (no nn.Module dispatch) | 12.43ms | initial |
| Single-CTA fused norm+quant | ~same | efb1989 |
| NVFP4 GEMV for down projection | 11.64ms | 09865c8 |
| Fused silu+mul+FP4 quant | 11.54ms | 21ffb97 |
| GEMV for O projection | 11.08ms | c375faf |
| All-GEMV decode (**BROKEN** — was 9.92ms) | — | a7d19f9 |
| FP8 KV cache dtype fix | fixed correctness | 712759a |
| Revert QKV/O/Gate+Up GEMV (precision) | 12.07ms | fa7e236 |
| O BF16-input GEMV (no FP4 quant) | 9.52ms | current |
