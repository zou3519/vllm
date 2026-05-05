# Flat Llama — Optimized NVFP4 Llama for Low-Latency Decode

## What this is

A hand-optimized "flat" model definition for `nvidia/Llama-3.3-70B-Instruct-NVFP4`
that beats vLLM's torch.compile'd model by 4-6% on BS=1 decode latency.

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
Flat:    85ms       12.2ms     1629ms
Compiled: 107ms     12.5ms     1696ms
Flat wins: 20%      2.9%       3.9%
```

## Files

- `vllm/model_executor/models/flat_llama.py` — model class (nn.Module for weight loading)
- `vllm/model_executor/models/flat_llama_kernels.py` — `flat_forward()`, `transformer_layer()`, Triton kernels
- `vllm/model_executor/models/flat_llama_cute_kernels.py` — CuTe DSL single-kernel norm+quant (WIP)
- `vllm/model_executor/models/registry.py` — `FlatLlamaForCausalLM` registration
- `benchmarks/benchmark_flat_llama.sh` — benchmark script
- `flat_llama_perf_analysis.md` — detailed per-op breakdown
- `flat_llama_dev_log.md` — development history
- `cute_dsl_cuda_graph_issue.md` — CuTe DSL integration issues

## Current optimization status

### What's integrated and working
- Fused norm+FP4 quant (Triton 2-kernel with PTX E2M1) — saves 1 kernel/norm
- Fused RoPE+KV cache write (Triton, 64 programs) — saves 1 kernel/layer
- Fused silu+mul+FP4 quant (C++ built-in) — saves 2 kernels/layer
- Triton FP4 quant for O input (faster than C++ at BS=1)
- Pre-allocated FP4 buffers (SharedDecodeBuffers, SharedPrefillBuffers)
- FULL CUDA graphs (no torch.compile)

### Active work: CuTe DSL single-kernel norm+quant
- Kernel is correct and 1.28x faster than Triton 2-kernel (20.5μs vs 26μs)
- CUDA graph stream issue fixed (was launching on stream 0)
- Current blocker: `_compiled_fn(*args)` Python dispatch overhead (~40μs/call)
- Tried `from_dlpack` (5.4μs × 8 = 43μs) and `make_ptr` — both have overhead
- The CuTe compiled function's `__call__` itself may be the bottleneck
- Potential fix: extract raw CUfunction and launch via cuLaunchKernel

### Investigated but not actionable
- Custom GEMV (Triton, CuTe DSL, raw CUDA) — cannot beat autotuned CUTLASS
- Down GEMM at 42% BW efficiency — structural CUTLASS limitation (K=28672)
- CuTe DSL GEMV — 46μs for QKV (break-even with CUTLASS+quant)

## Per-layer decode breakdown (11 kernels, 138μs/layer)

```
Op                               μs     Pct
──────────────────────────────────────────────
Norm+FP4 quant (pre-attn)        4.2    3.0%   ← CuTe DSL could make this 1 kernel
QKV GEMM                         9.0    6.5%
RoPE+KV write (fused)            2.5    1.8%
Attention decode                  9.3    6.7%
Merge states                      2.3    1.7%
FP4 quant (O input)              1.9    1.4%
O GEMM                           9.0    6.5%
Norm+FP4 quant (post-attn)       4.2    3.0%   ← CuTe DSL could make this 1 kernel
Gate+Up GEMM                    46.0   33.3%
Fused silu+mul+FP4 quant         1.8    1.3%
Down GEMM                       48.0   34.7%
──────────────────────────────────────────────
TOTAL per layer                138.2
80 layers                       11.1ms
CUDA graph overhead              1.1ms
Measured TPIT                   12.2ms
```

GEMMs = 81% of time. Small ops are fully optimized. The two norm+quant points
(4.2μs each, 2 kernels each) are the only remaining fusion opportunity.

## Model architecture (what transformer_layer actually calls)

```python
def transformer_layer(positions, hidden_states, residual,
                      input_ln_w, post_attn_ln_w, eps,
                      qkv, o, gate_up, down,  # NvFp4Proj dataclasses
                      rotary_emb, attn,        # callables
                      q_size, kv_size, bufs, backend):

    # 1-2. Fused norm + FP4 quant (Triton, 2 kernels)
    _add_variance_kernel(hidden, residual → residual_buf, variance)
    _norm_fp4_quant_kernel(residual_buf, weight, variance → fp4, scale)

    # 3. QKV GEMM (CUTLASS autotuned)
    qkv_out = nvfp4_gemm(fp4, scale, qkv_proj)
    q, k, v = split(qkv_out)

    # 4. Fused RoPE + KV cache write (Triton, 1 kernel)
    _fused_rope_kv_kernel(q, k, v, cos_sin, positions, kv_cache, slot_mapping)

    # 5. Attention (FlashInfer)
    attn_output = attn(q, k, v)  # kv_sharing skips redundant cache write

    # 6. FP4 quant for O input (Triton PTX, 1 kernel)
    triton_fp4_quant(attn_output → fp4, scale)

    # 7. O GEMM (CUTLASS autotuned)
    hidden_states = nvfp4_gemm(fp4, scale, o_proj)

    # 8-9. Fused norm + FP4 quant (Triton, 2 kernels)
    _add_variance_kernel(hidden, residual → residual_buf, variance)
    _norm_fp4_quant_kernel(residual_buf, weight, variance → fp4, scale)

    # 10. Gate+Up GEMM (CUTLASS autotuned)
    gate_up_out = nvfp4_gemm(fp4, scale, gate_up_proj)

    # 11. Fused SiLU+mul+FP4 quant (C++, 1 kernel)
    silu_and_mul_nvfp4_quant(gate_up_out → fp4, scale)

    # 12. Down GEMM (CUTLASS autotuned)
    hidden_states = nvfp4_gemm(fp4, scale, down_proj)

    return hidden_states, residual
```

The outer `flat_forward()` function does:
1. Embedding lookup
2. Set `kv_sharing_target_layer_name` on all attention layers (for fused RoPE+KV)
3. Loop: `transformer_layer()` × 80
4. Final RMSNorm
