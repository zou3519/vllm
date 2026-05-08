# Flat GPT-OSS - Optimized MXFP4 GPT-OSS for Low-Latency Decode

## Summary

A hand-flattened decode path for `openai/gpt-oss-120b` on a Blackwell-class
GPU (TP=1, BS=1). The flat model bypasses most per-layer `nn.Module` dispatch:
`flat_forward()` caches layer parameters as plain tensors and calls the concrete
attention, normalization, quantization, and MoE kernels directly.

The current optimization target is the flat model with torch.compile disabled
and full decode CUDA graphs enabled:

```text
--hf-overrides '{"architectures": ["FlatGptOssForCausalLM"]}'
-cc.mode=none
-cc.cudagraph_mode=full_decode_only
--max-cudagraph-capture-size 1
--max-num-batched-tokens 8192
```

Current verified results:

```text
Path                                      TPIT       Speedup
----------------------------------------------------------------
Regular vLLM + torch.compile             2.539576   baseline
Flat no-compile FULL_DECODE_ONLY         2.467747   1.029x
Initial flat no-compile target           5.386773   0.471x
```

The flat path is about 2.8% faster than regular compiled vLLM on the same
5x512-token Prometheus ITL benchmark, and about 2.18x faster than the first
valid no-compile flat target.

## Maintenance

When changing this path, sanity-check this whole file. The run command,
per-layer breakdown, torch.compile comparison, roofline discussion, and
optimization history can all go stale.

Always log every experiment in `results.tsv`. Keep `--max-num-batched-tokens
8192` for target comparisons; smaller values benchmark faster but are not the
target command.

## How to run

```bash
VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 \
vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 1 \
    --kv-cache-dtype fp8 \
    --no-enable-prefix-caching \
    --max-cudagraph-capture-size 1 \
    --max-num-batched-tokens 8192 \
    --stream-interval 20 \
    --tool-call-parser openai \
    --enable-auto-tool-choice \
    --hf-overrides '{"architectures": ["FlatGptOssForCausalLM"]}' \
    -cc.mode=none \
    -cc.cudagraph_mode=full_decode_only
```

Startup should show:

```text
CompilationMode.NONE
CUDAGraphMode.FULL_DECODE_ONLY
cudagraph_capture_sizes: [1]
Using 'FLASHINFER_TRTLLM_MXFP4_MXFP8' Mxfp4 MoE backend
```

## Model architecture

The decode layer is approximately:

```text
1. Residual path setup
2. Input RMSNorm or fused residual-add + RMSNorm
3. QKV linear
4. Split Q, K, V
5. RoPE on Q and K
6. FP8 KV-cache write
7. Optional FP8 Q quantization
8. Attention decode
9. O projection
10. Post-attention residual-add + RMSNorm
11. MXFP8 activation quantization for MoE
12. Router linear
13. FlashInfer TRTLLM MXFP4/MXFP8 MoE routing
14. MoE GEMM1
15. SwiGLU
16. MoE GEMM2
17. MoE finalize / top-k reduction
```

GPT-OSS here has 36 layers, 128 experts, top-k=4, hidden size 2880, and
FlashInfer/TRTLLM pads the MoE hidden/intermediate dimensions to 3072 for the
MXFP4 kernels.

## What was fused or optimized

```text
Area                         Change
---------------------------------------------------------------------------
Flat forward                 Cache all layer tensors and run a direct
                             function instead of per-module dispatch.

Post-attn norm + MXFP8       Fused residual add, RMSNorm, residual update,
quant                        MXFP8 quantization, and scale write in one
                             Triton kernel.

RoPE + KV cache + Q quant    Fused Q/K RoPE, FP8 KV-cache write, and optional
                             FP8 Q quantization in one Triton kernel.

RoPE/KV micro-optimizations  Skip q/k contiguous copies, skip storing rotated
                             K back to the temporary tensor on decode, and
                             early-return nonzero RoPE tiles before KV work.

MoE                          Use FlashInfer TRTLLM MXFP4/MXFP8 monolithic
                             MoE. The native FlashInfer routing/GEMM/finalize
                             path remains faster than the custom finalizers
                             tried so far.

Attention dependency         Reuse the existing KV-cache tensor as the
                             attention dummy dependency after the fused cache
                             write, avoiding an empty allocation.
```

## Per-token trace comparison

Profiler aggregate for the flat path was `56.092ms` self CUDA over the captured
trace; regular compiled vLLM was `57.767ms`. The MoE child kernels are
essentially the same in both paths. The flat path mainly wins around custom
post-attention norm/quant and RoPE/KV/Q handling.

```text
Area                         Flat path                         Regular torch.compile
------------------------------------------------------------------------------------------------
MoE GEMMs/routing/finalize   Same FlashInfer/TRTLLM child      Same kernels, nearly same timings
                             kernels

Post-attn norm + MXFP8       _fused_add_rms_norm_mxfp8_        Compiled Triton norm kernels plus
quant                        quant_kernel, 1.822ms             tensorrt_llm quantize rows, about
                                                               2.200ms

RoPE + KV cache + Q quant    _rope_and_cache_kernel,           reshape_and_cache_kernel_flash plus
                             2.303ms                           Triton pointwise RoPE/Q work and
                                                               wrapper overhead

MoE wrapper                  vllm::flashinfer_trtllm_fp4_      vllm::moe_forward
                             block_scale_moe

Compilation/glue             More Python-visible aten::linear  More compiled FX graph wrapper rows
                             bookkeeping
```

Flat trace highlights:

```text
Kernel / op                                      Total       Calls    Avg
--------------------------------------------------------------------------
FlashInfer MoE wrapper self CUDA                 4.216ms       36   117.1us
MoE MXFP4/MXFP8 GEMM1 child kernels             10.910ms      720    15.2us
MoE GEMM2 child kernels                          7.270ms      720    10.1us
MoE routing                                      3.398ms      720     4.7us
MoE finalize                                     2.340ms      720     3.3us
RoPE + KV cache + Q quant                        2.303ms      720     3.2us
Post-attn add + RMSNorm + MXFP8 quant            1.822ms      720     2.5us
Attention decode kernels                         2.862ms      684     4.2us
```

## Roofline notes

BS=1 decode is dominated by launch overhead, active weight reads, scale reads,
and small-M MoE inefficiency. The user's target roofline is below 1ms TPIT for
this workload; the current verified flat path is still 2.47ms, so there is a
large gap.

The custom scalar BF16xMXFP4 GEMV prototypes were far from the roofline
(15-22ms TPIT), which strongly suggests the next viable custom MoE path must
use tensor-core/blockscaled GEMM primitives, not scalar dequant-FMA loops.
FlashInfer's current TRTLLM path is already substantially better for the MoE
work. Future wins likely require fusing around that tensor-core path, changing
the weight/routing layout, or writing a tensor-core BS=1/top-k=4 specialized
MoE kernel.

## Known issues and investigations

### FlashInfer MXFP4/MXFP8 env var is required

Use:

```bash
VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1
```

Without this, the flat path may not select the intended TRTLLM MXFP4/MXFP8 MoE
backend.

### Autotune cache matters

FlashInfer autotuning is cached under the vLLM/FlashInfer cache directory.
Disabling FlashInfer autotune or forcing alternative tactics regressed TPIT in
the experiments logged in `results.tsv`.

### PDL should stay enabled

Disabling PDL for the flat FlashInfer MoE quality-passed but regressed TPIT to
`2.592595ms`.

### Custom scalar MoE kernels are not viable

Several raw CUDA BF16xMXFP4 BS=1 MoE prototypes were tried. The initial
half-split `w13` interpretation was wrong and failed quality; after fixing
the interleaved `w13` row pairing, the kernels quality-passed but measured
`15-22ms` TPIT. The scalar GEMV strategy is not competitive with FlashInfer's
tensor-core kernels.

### Custom MoE finalizer did not help

Using `do_finalize=False` from FlashInfer and reducing top-4 expert rows in a
custom Triton finalizer quality-passed, but regressed to `2.502972ms`. The
native FlashInfer finalizer remains better.

### Allocation-only MoE tweaks did not help

Letting FlashInfer allocate the output (`2.475045ms`) and using persistent
per-layer decode output buffers (`2.487326ms`) both regressed versus the best
flat result.

## Files

- `vllm/model_executor/models/flat_gpt_oss.py` - flat model class for weight loading.
- `vllm/model_executor/models/flat_gpt_oss_forward.py` - flat forward pass and custom op wrappers.
- `vllm/model_executor/models/flat_gpt_oss_kernels.py` - Triton kernels for post-attn norm/MXFP8 quant and RoPE/KV/Q quant.
- `flat_config.txt` - serve commands and early baseline metadata.
- `results.tsv` - full experiment log.

## Optimization history

Full history is in `results.tsv`. Important milestones:

```text
Change                                                       TPIT       Status    Commit
-----------------------------------------------------------------------------------------
Initial valid flat no-compile FULL_DECODE_ONLY target        5.386773   keep      3296ed7
Skip q/k contiguous copies on decode                         2.477788   keep      69dd19b
Skip rotated-K temp store after KV-cache write               2.468724   keep      b441f1c
Early-return nonzero RoPE tiles                              2.468310   keep      536f746
Reuse KV cache tensor as attention dependency                2.467747   keep      06d27a9
Regular non-flat GptOssForCausalLM with torch.compile        2.539576   baseline  regular

Invalid faster max-num-batched-tokens=512 experiment         2.456266   discard   1ce977f+mbt512
Disable PDL                                                  2.592595   discard   5efa65395
4096-wide RoPE/KV/Q tile                                     2.495630   discard   745a39a97
FlashInfer do_finalize=false + custom Triton finalizer       2.502972   discard   a89e50464
FlashInfer output=None                                       2.475045   discard   bcf36f490
Persistent decode MoE output buffers                         2.487326   discard   13322ebb7
```

## Next likely directions

The obvious simple launch/allocation changes have mostly been exhausted. The
highest-risk/highest-upside direction is a real tensor-core BS=1/top-k=4 MoE
kernel or a deeper fusion around FlashInfer's existing TRTLLM MoE path. A
replacement should preserve the tensor-core blockscaled math; scalar dequant
GEMV is too slow.
