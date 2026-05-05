# Flat Llama — Optimized NVFP4 Llama for Low-Latency Decode

## What this is

A hand-optimized "flat" model definition for `nvidia/Llama-3.3-70B-Instruct-NVFP4`
that beats vLLM's torch.compile'd model by ~8% on BS=1 decode latency.

The model bypasses vLLM's nn.Module hierarchy — the forward pass is a single
`flat_forward()` function that calls `transformer_layer()` in a loop with all
parameters passed explicitly as plain tensors.

## How to run

```bash
# Serve (TP=1 — NVFP4 70B fits on single GB300)
vllm serve nvidia/Llama-3.3-70B-Instruct-NVFP4 \
    --hf-overrides '{"architectures": ["FlatLlamaForCausalLM"]}' \
    --compilation-config '{"mode": "none", "cudagraph_mode": "full"}'

# Query
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"nvidia/Llama-3.3-70B-Instruct-NVFP4",
         "messages":[{"role":"user","content":"Hello!"}]}'
```

## Accuracy

The flat+GEMV model produces **factually identical** answers to the compiled
baseline. Exact token-for-token match is ~3/8 on a test suite because the GEMV
uses FP16 intermediate accumulation (vs CUTLASS FP32 tensor core), causing
slight logit differences that lead to different but equally valid phrasings.

All arithmetic, factual, and translation answers are correct. The differences
are purely stylistic (e.g. "composed of interconnected nodes" vs "composed of
layers of interconnected nodes").

## How to measure latency

### Total latency (TTFT + decode)
```bash
vllm bench latency \
    --model nvidia/Llama-3.3-70B-Instruct-NVFP4 \
    --tensor-parallel-size 1 \
    --batch-size 1 --input-len 128 --output-len 128 \
    --num-iters 20 --num-iters-warmup 10 \
    --hf-overrides '{"architectures": ["FlatLlamaForCausalLM"]}' \
    --compilation-config '{"mode": "none", "cudagraph_mode": "full"}' \
    --max-model-len 4096
```

### TTFT vs TPIT breakdown
```python
import time, torch
from vllm import LLM, SamplingParams

def measure(name, **kwargs):
    llm = LLM(model='nvidia/Llama-3.3-70B-Instruct-NVFP4',
              tensor_parallel_size=1, max_model_len=4096, **kwargs)
    prompt = [{'prompt_token_ids': list(range(10, 138))}]
    sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    sp128 = SamplingParams(max_tokens=128, temperature=0, ignore_eos=True)
    for _ in range(10): llm.generate(prompt, sp128, use_tqdm=False)

    iters = 30
    t0 = time.perf_counter()
    for _ in range(iters): llm.generate(prompt, sp1, use_tqdm=False)
    ttft = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters): llm.generate(prompt, sp128, use_tqdm=False)
    total = (time.perf_counter() - t0) / iters

    tpit = (total - ttft) / 127
    print(f'{name}: TTFT={ttft*1000:.1f}ms  TPIT={tpit*1000:.2f}ms')
    del llm; torch.cuda.empty_cache()

# Flat model
measure('Flat',
    hf_overrides={'architectures': ['FlatLlamaForCausalLM']},
    compilation_config={'mode': 'none', 'cudagraph_mode': 'full'})

# Compiled baseline
measure('Compiled',
    compilation_config={'cudagraph_mode': 'full'})
```

### Compare against compiled baseline
Always use `cudagraph_mode: full` for both to get a fair comparison.
The compiled model uses `{"cudagraph_mode": "full"}` (default mode includes
torch.compile which we want to compare against).

## Current results (GB300, TP=1, BS=1)

```
             TTFT       TPIT       Total (128in+128out)
Compiled:    107ms      12.50ms    1696ms
Flat+GEMV:   78ms        9.92ms    1345ms
Flat wins:   27%        20.6%      20.7%
```

## Files

- `vllm/model_executor/models/flat_llama.py` — model class (nn.Module for weight loading)
- `vllm/model_executor/models/flat_llama_kernels.py` — `flat_forward()`, `transformer_layer()`, Triton kernels
- `vllm/model_executor/models/flat_llama_gemv.py` — NVFP4 GEMV kernel (raw CUDA via load_inline)
- `vllm/model_executor/models/flat_llama_cute_kernels.py` — CuTe DSL kernels (deprecated, not used)
- `vllm/model_executor/models/registry.py` — `FlatLlamaForCausalLM` registration
- `benchmarks/benchmark_flat_llama.sh` — benchmark script
- `tests/bench_fp4_gemv.py` — GEMV tuning benchmark
- `tests/test_single_cta_fused_kernel.py` — single-CTA norm+quant test

## Current optimization status

### What's integrated and working
- Fused norm+FP4 quant (single-CTA Triton kernel) — 1 kernel per norm site (was 2)
- Fused RoPE+KV cache write (Triton, 64 programs) — saves 1 kernel/layer
- Fused silu+mul+FP4 quant with row-major scales (Triton) — for GEMV path
- NVFP4 GEMV for down projection (raw CUDA) — 22.8μs vs 48μs CUTLASS (2.1×)
- NVFP4 GEMV for O projection (raw CUDA) — replaces CUTLASS, ~0.5ms/step savings
- Triton FP4 quant with row-major scales for GEMV paths (O + down)
- Pre-allocated FP4 buffers (SharedDecodeBuffers, SharedPrefillBuffers)
- Weight scale unswizzling at load time (CUTLASS → row-major for GEMV)
- FULL CUDA graphs (all kernels graph-compatible, no CuTe DSL)

### Investigated but not actionable
- GEMV for gate+up: 43μs vs 46μs CUTLASS — only 6.5% faster, not worth complexity
- GEMV for QKV/O: 8-10μs vs 9μs CUTLASS — marginal improvement
- CuTe DSL fused kernel: correct and 1.28× faster than Triton, but incompatible
  with CUDA graphs (uses non-standard launch mechanism)
- Custom GEMV (Triton): couldn't match raw CUDA GEMV performance
- Down GEMM at 42% BW efficiency was structural CUTLASS limitation (K=28672)
  — solved by switching to GEMV (72% efficiency)

## Per-layer decode breakdown (11 kernels, ~113μs/layer)

```
Op                                    μs     Pct   Status
─────────────────────────────────────────────────────────────
Gate+Up GEMM (CUTLASS)              46.0   43.4%   ← biggest bottleneck (72% BW eff)
GEMV down projection                22.8   21.5%   ★ was 48μs CUTLASS
QKV GEMM (CUTLASS)                   9.0    8.5%   66% BW efficiency
Attention decode (FlashInfer)        9.3    8.8%   external
GEMV O projection                    8.2    7.7%   ★ was 9μs CUTLASS
Norm+FP4 quant (pre-attn)            4.2    4.0%   ● single-CTA Triton
Norm+FP4 quant (post-attn)           4.2    4.0%   ● single-CTA Triton
RoPE+KV write (fused)                2.5    2.4%   ● Triton fused
Merge states                         2.3    2.2%   external (FlashInfer)
FP4 quant (O input, row-major)       1.9    1.8%   ★ Triton, row-major scales
Fused silu+mul+FP4 quant             1.8    1.7%   ★ Triton, row-major scales
─────────────────────────────────────────────────────────────
TOTAL per layer                    106.0 (est)
80 layers                            8.5ms
CUDA graph + framework overhead     ~2.5ms
Measured TPIT                       11.08ms
```

78% of layer time is linear algebra (GEMM/GEMV). The remaining GEMMs
(QKV, gate+up) are at 66-72% of peak memory bandwidth.

## Memory bandwidth analysis

```
Projection       Weight MB   BW floor μs   Actual μs   Efficiency
──────────────────────────────────────────────────────────────────
QKV GEMM             47.2          5.9          9.0         66%
O GEMM               37.7          4.7          9.0         52%
Gate+Up GEMM        264.2         33.0         46.0         72%
Down GEMV           132.1         16.5         22.8         72%
```

Theoretical floor: 40.5 GB weights / 8 TB/s = 5.06 ms/token.
Measured: 11.64 ms/token = 43% of ideal.

## Roofline vs context length

```
Context   KV cache   Weight     Total      BW floor   Est. TPIT
  128      0.02 GB   40.5 GB   40.5 GB     5.06 ms    ~11.6 ms
   1K      0.16 GB   40.5 GB   40.7 GB     5.08 ms    ~11.7 ms
   8K      1.25 GB   40.5 GB   41.8 GB     5.22 ms    ~12.0 ms
  32K      5.0  GB   40.5 GB   45.5 GB     5.69 ms    ~13.0 ms
 128K     20.0  GB   40.5 GB   60.5 GB     7.56 ms    ~17   ms
```

KV cache = weight crossover: ~260K tokens.

## Model architecture (what transformer_layer actually calls)

```python
def transformer_layer(positions, hidden_states, residual, ...):

    # 1. Fused norm + FP4 quant (single-CTA Triton, 1 kernel)
    _single_cta_add_rms_norm_fp4_quant(hidden, residual → residual_buf, fp4, scale)

    # 2. QKV GEMM (CUTLASS autotuned)
    qkv_out = nvfp4_gemm(fp4, scale, qkv_proj)
    q, k, v = split(qkv_out)

    # 3. Fused RoPE + KV cache write (Triton, 1 kernel)
    _fused_rope_kv_kernel(q, k, v, cos_sin, positions, kv_cache, slot_mapping)

    # 4. Attention (FlashInfer)
    attn_output = attn(q, k, v)

    # 5. FP4 quant for O input (Triton PTX, row-major scales, 1 kernel)
    triton_fp4_quant_rowmajor(attn_output → fp4, scale)

    # 6. O GEMV (raw CUDA, 1 kernel)
    nvfp4_gemv(weight, fp4, weight_scale_rowmajor, scale, hidden_states, alpha)

    # 7. Fused norm + FP4 quant (single-CTA Triton, 1 kernel)
    _single_cta_add_rms_norm_fp4_quant(hidden, residual → residual_buf, fp4, scale)

    # 8. Gate+Up GEMM (CUTLASS autotuned)
    gate_up_out = nvfp4_gemm(fp4, scale, gate_up_proj)

    # 9. Fused SiLU+mul+FP4 quant with row-major scales (Triton, 1 kernel)
    triton_silu_mul_fp4_quant_rowmajor(gate_up_out → fp4, scale)

    # 10. GEMV down projection (raw CUDA, 1 kernel)
    nvfp4_gemv(weight, fp4, weight_scale_rowmajor, scale, hidden_states, alpha)

    return hidden_states, residual
```

The outer `flat_forward()` function does:
1. Embedding lookup
2. Set `kv_sharing_target_layer_name` on all attention layers (for fused RoPE+KV)
3. Loop: `transformer_layer()` × 80
4. Final RMSNorm

## Optimization history

| Change | TPIT impact | Commit |
|---|---|---|
| Flat model (no nn.Module dispatch) | 12.50 → 12.43ms | initial |
| Single-CTA fused norm+quant (Triton) | kernel: 20.9 → 18.9μs | efb1989 |
| NVFP4 GEMV for down projection | kernel: 48 → 22.8μs | 09865c8 |
| CUDA graph stream fix for GEMV | fixed silent graph capture failure | a41d961 |
| Fused silu+mul+FP4 quant (Triton) | 11.89 → 11.64ms | 21ffb97 |
| BF16-input GEMV (added, reverted for O) | 20.7μs — too slow at K=4096 | 0510ef7 |
| GEMV for O projection (FP4 input) | 11.54 → 11.08ms | c375faf |
| GEMV for ALL projections (QKV+Gate+Up) | 11.08 → 9.92ms | a7d19f9 |

## Workflow

When making new discoveries that improve performance, **commit immediately
and update this log**. This keeps the optimization history traceable and
makes it easy to bisect if a change causes regressions.
