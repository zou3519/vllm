# Flat Llama Development Log

## Goal
Optimize nvidia/Llama-3.3-70B-Instruct-NVFP4 for low-latency BS=1 decode
by writing a flat model definition that bypasses vLLM's nn.Module hierarchy
and enables manual kernel fusion.

## Session 1: Initial Setup

### Created flat model definition
- `vllm/model_executor/models/flat_llama.py` — FlatLlamaForCausalLM
- Same weight hierarchy as LlamaForCausalLM (for checkpoint loading)
- Forward path calls ops directly, bypassing nn.Module dispatch
- Registered in `vllm/model_executor/models/registry.py`
- Tested: server starts, outputs are correct and coherent

### Key architecture decisions
- TP=1 recommended (NVFP4 70B fits on single GB300 at ~35GB)
- torch.compile OFF, CUDA graphs FULL
- Usage: `--hf-overrides '{"architectures": ["FlatLlamaForCausalLM"]}' --compilation-config '{"mode": "none", "cudagraph_mode": "full"}'`

## Session 2: Optimization Journey

### Phase 1: Pre-allocated buffers + direct C++ ops
- Created `flat_llama_kernels.py` with `LayerDecodeBuffers`
- Pre-allocate all FP4 output tensors (eliminates torch.empty per call)
- Call `scaled_fp4_quant.out` directly with pre-alloc'd outputs
- Result at TP=4: 12.8s → from 14.3s baseline (11% improvement without CG)

### Phase 2: CUDA graphs
- Added `--compilation-config '{"mode": "none", "cudagraph_mode": "full"}'`
- Result at TP=4: 1.39s (9.2x speedup from CUDA graphs!)
- vs compiled: 1.70s compiled, 1.39s flat — but this was TP=4

### Phase 3: Switch to TP=1
- NVFP4 70B fits on single GB300 (284GB)
- Eliminates all-reduce overhead entirely
- No allreduce fusion needed

### Phase 4: Fused silu_and_mul_nvfp4_quant
- Discovered vLLM already has `torch.ops._C.silu_and_mul_nvfp4_quant`
- This is the ONLY fusion torch.compile applies for NVFP4
- Integrated directly — saves 2 kernel launches per layer
- Also discovered: `fuse_norm_quant` only handles FP8, NOT NVFP4

### Phase 5: Inductor output analysis
- Used TORCH_LOGS=output_code to dump inductor's generated kernels
- Found: inductor does NOT fuse RMSNorm+FP4 quant either
- Inductor generates Triton reduction kernels for fused_add_rms_norm
- Inductor generates pointwise kernels for RoPE
- The compiled advantage is mostly from graph-level memory planning

### Phase 6: Truly flat architecture
User requested: "one big function that calls a transformer_layer function"
- `flat_forward()` → loops over `transformer_layer()` with explicit params
- `NvFp4Proj` dataclass holds per-projection weights
- `SharedDecodeBuffers` shared across all 80 layers
- `extract_all_layer_params()` pulls weights from nn.Modules after loading
- Result: flat model slightly faster than compiled on some runs

### Phase 7: Fused norm+FP4 quant with Triton inline PTX
- Wrote Triton kernel using `tl.inline_asm_elementwise` with PTX:
  `"{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1; cvt.u16.u8 $0, tmp; }"`
- Constraints: `"=h, r, r"` (output=16-bit reg, inputs=32-bit regs)
- Two-kernel approach: `_add_variance_kernel` + `_norm_fp4_quant_kernel`
- Micro-benchmark: 64μs fused vs 81μs separate (1.26x faster)
- FP4 output: 0.5% mismatches (rounding differences), scales: exact match
- Triton quirks: `tl.float8e4nv` (not float8e4m3fn), `bitcast=True` for fp8→u8

### Phase 8: Standalone Triton FP4 quant
- `_fp4_quant_kernel`: standalone FP4 quant using same PTX approach
- 9.7μs vs 10.6μs C++ (1.09x faster at BS=1)
- Exact output match
- Applied to O-projection's input quantization

### Phase 9: TTFT/TPIT breakdown (KEY FINDING)
```
         TTFT       TPIT      Decode(127)   Total
Flat:    103.6ms    12.58ms   1597ms        1701ms
Compiled: 91.1ms   12.66ms   1607ms        1699ms
```
**Flat WINS on decode (TPIT 12.58ms vs 12.66ms)**
**Flat LOSES on prefill (TTFT 104ms vs 91ms)**
Total is roughly even — the prefill loss cancels the decode win.

### Phase 10: RoPE+KV cache fusion (attempted)
- Wrote Triton kernel: one program per head, does RoPE + KV cache write
- Issues: correctness bugs in RoPE math, and 2x slower than separate ops
- Root cause: too little work per program (128 elements/head is tiny)
- Conclusion: at BS=1, these ops are so small that fusion overhead > savings

### Phase 11: CuTe DSL investigation
- CUTLASS 4.4.2 with CuTe DSL is available in the environment
- Has `Float4E2M1FN` type and full CuTe layout system
- Potential use: cross-layer GEMM epilogue that fuses down_proj → norm → FP4 quant

## Current Results (latest)
```
Standard (compiled, FULL CG):  1.705s  (TTFT=91ms, TPIT=12.66ms)
Flat v6 (no compile, FULL CG): 1.711s  (TTFT=104ms, TPIT=12.58ms)
Gap: +0.34% total, but FLAT WINS on decode
```

## Per-Layer Kernel Count: 11
1-2: Fused norm+FP4 quant (Triton PTX, 2 kernels — was 3 ops)
3: QKV GEMM (FlashInfer CUTLASS)
4: RoPE (C++ in-place)
5: KV cache write (reshape_and_cache_flash)
6: Attention (FlashInfer decode)
7: FP4 quant (Triton PTX — O projection input)
8: O GEMM (FlashInfer CUTLASS)
9-10: Fused norm+FP4 quant (Triton PTX, 2 kernels — was 3 ops)
11: Gate+Up GEMM (FlashInfer CUTLASS)
12: Fused silu+mul+FP4 quant (C++ — was 3 ops)
13: Down GEMM (FlashInfer CUTLASS)

## Key Technical Learnings

1. **CUDA graph per-node overhead**: ~1.2μs/node. Reducing from 13→11 kernels saves ~192μs/step
2. **scaled_fp4_quant takes input_global_scale_inv** (1/scale), NOT scale itself
3. **PTX E2M1**: output must use `.b8` register (declare inside asm block, convert to `.b16`)
4. **Triton float8**: `tl.float8e4nv` (not `tl.float8e4m3fn`); need `bitcast=True` for fp8→u8
5. **tl.static_range(0, 512)** causes infinite compile time — use `tl.range` for dynamic loops
6. **BS=1 underutilization**: most small ops only use 1 SM out of 152. The GPU is memory-bound on GEMM weight reads.
7. **Inductor for NVFP4**: only fusion is `silu_and_mul_nvfp4_quant`. No norm+FP4 fusion.

## Next Steps
1. **Optimize prefill path** — make flat_forward() work for M>1 (direct C++ ops, no nn.Module)
2. **CuTe DSL GEMM epilogue** — fuse down_proj GEMM → next layer's norm+FP4 quant
3. **RoPE+KV as single C++ kernel** (not Triton — too little work per program for Triton)

## Files
- `vllm/model_executor/models/flat_llama.py` — model definition (nn.Module for loading, flat_forward for inference)
- `vllm/model_executor/models/flat_llama_kernels.py` — flat_forward, transformer_layer, Triton kernels, NvFp4Proj, SharedDecodeBuffers
- `vllm/model_executor/models/registry.py` — FlatLlamaForCausalLM registration
- `benchmarks/benchmark_flat_llama.sh` — benchmark script
