# Flat Llama Performance Analysis

## Current Performance (as of 2026-05-04)

**Model**: nvidia/Llama-3.3-70B-Instruct-NVFP4, TP=1, BS=1, GB300 (SM 10.3)

### End-to-end results

| Config | TTFT | TPIT | Total (128in+128out) |
|--------|------|------|---------------------|
| Flat Llama (no compile, FULL CG) | 85ms | 12.2ms | 1629ms |
| Standard (compiled, FULL CG) | 107ms | 12.5ms | 1696ms |
| **Flat wins by** | **20%** | **2.9%** | **3.9%** |

Verified across multiple configurations:
- in=1024, out=64: Flat 862ms vs Compiled 904ms (4.6% faster)
- in=512, out=32: Flat 471ms vs Compiled 499ms (5.6% faster)
- in=64, out=256: Flat 3160ms vs Compiled 3317ms (4.7% faster)

### Per-layer BS=1 decode breakdown (autotuned, from nsys)

```
Op                               μs     Pct    Notes
─────────────────────────────────────────────────────────────
Norm+FP4 quant (pre-attn)        4.2    3.0%   Triton 2-kernel (variance + norm+E2M1 PTX)
QKV GEMM                         9.0    6.5%   CUTLASS FP4, autotuned, 87% BW efficiency
RoPE+KV write (fused)            2.5    1.8%   Triton, 64 programs (one per Q head)
Attention decode                  9.3    6.7%   FlashInfer paged decode
Merge states                      2.3    1.7%   FlashInfer post-attention
FP4 quant (O input)              1.9    1.4%   Triton PTX E2M1
O GEMM                           9.0    6.5%   CUTLASS FP4, autotuned
Norm+FP4 quant (post-attn)       4.2    3.0%   Same as pre-attn
Gate+Up GEMM                    46.0   33.3%   CUTLASS FP4, 87% BW efficiency
Fused silu+mul+FP4 quant         1.8    1.3%   C++ built-in kernel
Down GEMM                       48.0   34.7%   CUTLASS FP4, only 42% BW efficiency
─────────────────────────────────────────────────────────────
TOTAL per layer                138.2
80 layers                       11.1ms
CUDA graph overhead              1.1ms
Measured TPIT                   12.2ms
```

### Category breakdown

| Category | Time/layer | % | Status |
|----------|-----------|---|--------|
| GEMMs (4 projections) | 112.0μs | 81% | Autotuned CUTLASS, near-optimal for 3/4 |
| Small ops (norm, quant, rope, silu) | 14.6μs | 11% | Fully fused, little room left |
| Attention + merge | 11.6μs | 8% | FlashInfer, hardware-limited |
| CUDA graph overhead | ~14μs/layer equiv | — | ~1.1ms total for 880 nodes |

### Theoretical limits

| Projection | Weight data | BW limit (6.6 TB/s) | Actual | Efficiency |
|-----------|------------|---------------------|--------|------------|
| QKV (8192→10240) | 47 MB | 7.1μs | 9.0μs | 79% |
| O (8192→8192) | 38 MB | 5.7μs | 9.0μs | 63% |
| Gate+Up (8192→57344) | 264 MB | 40.0μs | 46.0μs | **87%** |
| Down (28672→8192) | 132 MB | 20.0μs | 48.0μs | **42%** |

Note: QKV and O show <80% efficiency but the absolute time (9μs) is so small
that the overhead is fixed kernel launch + tile setup cost, not bandwidth waste.

### Remaining optimization opportunities

**1. Down GEMM bandwidth efficiency (42% → 80%): saves ~1.8ms/step**
The Down projection reads 132MB but takes 48μs (0.42 of 6.6 TB/s peak).
At 80% efficiency it would take ~25μs. This is likely a CUTLASS autotune
issue — the K=28672 dimension may not tile well with the selected config.
Potential fix: force a different CUTLASS tactic or write a specialized kernel.

**2. CUDA graph overhead reduction: saves ~0.5ms/step**
880 graph nodes × ~1.2μs/node = 1.1ms. Reducing kernel count from 11 to 9
per layer (merging norm+quant into 1 kernel) would save ~160 nodes = ~190μs.
The CuTe DSL single-kernel norm+quant works but hasn't been integrated yet.

**3. Nothing else is worth optimizing.**
Small ops at 14.6μs/layer are already minimal. Attention at 9.3μs is
hardware-limited. The GEMMs are the GPU's fundamental bottleneck.

### GEMV research conclusions

We extensively explored replacing CUTLASS FP4 GEMM with a custom GEMV:

| Approach | QKV time | vs CUTLASS | Verdict |
|----------|---------|------------|---------|
| Triton + software dequant | 56μs | 6.2x slower | Software E2M1 too slow |
| Triton + PTX hw dequant | 56μs | 6.2x slower | Per-program overhead |
| CuTe DSL scalar loads | 153μs | 17x slower | Scalar loads, no coalescing |
| CuTe DSL u32 loads | 117μs | 13x slower | Better but still slow |
| CuTe DSL no-chunk | 46μs | 5.1x slower | Best CuTe result |
| Raw CUDA + LUT dequant | 135μs | 15x slower | LUT compute-bound |
| Raw CUDA + PTX hw dequant | 166μs | 18x slower | Input reload overhead |
| CUTLASS autotuned | **9μs** | **baseline** | Tensor cores win |

**Why CUTLASS wins**: On Blackwell (SM 10.3), tensor cores natively support
FP4 (E2M1) multiply-accumulate. Even though M=64 tiles waste 63/64 at BS=1,
the tensor core throughput is so high (~600+ TOPS effective) that scalar
CUDA core approaches can't compete. The autotuned CUTLASS kernel achieves
~80% bandwidth efficiency on large matrices (Gate+Up).

**The only scenario where GEMV would win**: if the CUTLASS autotune picks a
bad configuration. The un-autotuned CUTLASS runs at 165-275μs per GEMM,
which our CuTe GEMV (46μs) easily beats by 3-6x. FlashInfer's autotuner
is the key enabler of CUTLASS performance.

### Optimizations implemented

| Optimization | Kernel savings | Time savings | Status |
|-------------|---------------|-------------|--------|
| Fused norm+FP4 quant (Triton PTX) | 3→2 kernels/norm | ~12μs/norm | ✅ Integrated |
| Fused RoPE+KV cache write | 2→1 kernels | ~16μs/layer | ✅ Integrated |
| Fused silu+mul+FP4 quant (C++) | 3→1 kernels | ~10μs/layer | ✅ Integrated |
| Triton FP4 quant (O input) | Same count | ~1μs/layer | ✅ Integrated |
| Pre-allocated FP4 buffers | 0 kernel savings | ~5ms total (no CG) | ✅ Integrated |
| SharedPrefillBuffers | 0 kernel savings | ~13ms TTFT | ✅ Integrated |
| FULL CUDA graphs | 0 kernel savings | ~25ms total | ✅ Via config flag |
| CuTe DSL single-kernel norm+quant | 2→1 kernels/norm | ~192μs/step | ⚠️ Code exists, not integrated |
| CuTe DSL GEMV | Replaces GEMM+quant | Break-even | ❌ Not worth it |

### Hardware specs

- GPU: NVIDIA GB300 (SM 10.3, Blackwell)
- HBM3e: 284 GB, measured bandwidth 6.6 TB/s
- SMs: 152
- FP4 tensor cores: native E2M1 MMA support (tcgen05)
- CUDA graph replay overhead: ~1.2μs per graph node

### Files

- `vllm/model_executor/models/flat_llama.py` — model class (nn.Module for loading)
- `vllm/model_executor/models/flat_llama_kernels.py` — flat_forward, transformer_layer, Triton kernels
- `vllm/model_executor/models/flat_llama_cute_kernels.py` — CuTe DSL norm+quant kernel
- `vllm/model_executor/models/registry.py` — FlatLlamaForCausalLM registration
- `benchmarks/benchmark_flat_llama.sh` — benchmark script
- `flat_llama_dev_log.md` — development history

## Corrected GEMM Efficiency Analysis (with scale data)

| Projection | Weight | Scale | Total | Theory | Actual | Efficiency |
|-----------|--------|-------|-------|--------|--------|------------|
| QKV (8192→10240) | 42MB | 5MB | 47MB | 7.1μs | 9.0μs | 79% |
| O (8192→8192) | 34MB | 4MB | 38MB | 5.7μs | 9.0μs | 64% |
| Gate+Up (8192→57344) | 235MB | 29MB | 264MB | 40.0μs | 46.0μs | **87%** |
| Down (28672→8192) | 117MB | 15MB | 132MB | 20.0μs | 48.0μs | **42%** |

QKV and O inefficiency is from fixed kernel launch/setup overhead (absolute
time is already <10μs). Gate+Up is near-optimal. Down's 42% efficiency is
structural: K=28672 requires 112 tensor core K-iterations per output tile
(vs 32 for Gate+Up), causing excessive pipeline fill/drain overhead.

### Down GEMM optimization attempts

- **Split-K (2 halves)**: Made it worse (2× kernel launches + overhead)
- **CuTe DSL cute-dsl backend**: 103μs without autotune, worse than CUTLASS 48μs
- **Custom GEMV approaches**: Cannot beat autotuned CUTLASS tensor cores
- **Root cause**: CUTLASS architectural limitation — large K with small N
  causes poor pipeline utilization. Would need CUTLASS-internal changes
  (e.g., persistent scheduling, larger K-tiles) to fix.
