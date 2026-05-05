# CuTe DSL CUDA Graph Incompatibility

## Summary

The CuTe DSL fused norm+FP4 quant kernel is **correct and 1.28x faster** than
the Triton 2-kernel approach (20.5μs vs 26.3μs), but **incompatible with CUDA
graphs**. Output is correct in eager mode but produces gibberish with CUDA
graph capture/replay.

## Root cause

CuTe DSL's compiled function (`CudaDialectJitCompiledFunction`) uses its own
execution mechanism (`run_compiled_program` via a ctypes `capi_func`) rather
than standard `cudaLaunchKernel`. CUDA graphs can only capture standard CUDA
API calls, so the CuTe kernel launch is not recorded in the graph.

During CUDA graph replay, the CuTe kernel simply doesn't run — the graph
replays whatever was captured, which doesn't include the CuTe kernel. This
causes the norm+quant step to be skipped, producing garbage output.

## Evidence

```
# Eager mode (cudagraph_mode=none): CORRECT
$ curl .../chat/completions -d '{"messages":[{"content":"What is 3+4?"}]}'
→ "7"

# CUDA graph mode (cudagraph_mode=full): GARBAGE
$ curl .../chat/completions -d '{"messages":[{"content":"What is 3+4?"}]}'
→ "7!!!!" or "3 + 3+"
```

Standalone correctness tests pass with changing tensors (not a kernel bug).

## Kernel details

- File: `vllm/model_executor/models/flat_llama_cute_kernels.py`
- 1 CTA × 256 threads (8 warps), processes 8192 elements
- Phase 1: residual add + sum-of-squares (per-thread, warp reduction via
  `warp_reduction_sum`, cross-warp via shared memory)
- Phase 2: RMSNorm + E2M1 FP4 quantization via inline PTX
- Uses `cute.compile()` for JIT compilation, `from_dlpack()` for tensor wrapping

## Performance

```
Triton 2-kernel: 26.3μs (2 launches, CUDA graph compatible)
CuTe 1-kernel:   20.5μs (1 launch, NOT CUDA graph compatible)
Speedup:          1.28x
Potential savings: 0.9ms per decode step (160 calls × 5.8μs)
```

## Python wrapper overhead

`from_dlpack()` costs 5.4μs per call × 8 tensors = 43μs. With tensor
pointer caching, the wrapper overhead drops to near-zero for repeated
calls with the same pre-allocated buffers.

## Potential fixes

1. **Extract the raw CUDA kernel** from `CudaDialectJitCompiledFunction`:
   - `fn.jit_module.cuda_library` contains the compiled CUDA module
   - `fn.capi_func` is a ctypes wrapper
   - `fn.kernel_info` has the kernel name
   - If we can get the `CUfunction` handle, we can launch it via
     `cuLaunchKernel` which CUDA graphs CAN capture.

2. **Use CuTe DSL's export feature**: `fn.export_to_c()` or
   `fn.dump_to_object()` might give us a standalone binary we can load
   as a regular CUDA module.

3. **Write an equivalent raw CUDA kernel** using `torch.utils.cpp_extension.load_inline`
   with `-gencode=arch=compute_103a,code=sm_103a` (the `a` suffix is needed
   for E2M1 PTX instructions on Blackwell).

4. **Write an equivalent Triton kernel** that does the full norm+quant in one
   launch. The previous attempt with atomic barriers was 51μs (slower than
   2-kernel at 26μs). A better approach might use cooperative groups or
   a different parallelism strategy.

## CuTe DSL compiled function internals

```python
fn = cute.compile(_launch_fused_kernel, *args)

type(fn)  # CudaDialectJitCompiledFunction
fn.kernel_info  # OrderedDict with kernel name
fn.capi_func    # ctypes CFunctionType
fn.jit_module   # CudaDialectJitModule
fn.jit_module.cuda_library  # the compiled CUDA module
fn.jit_module.capi_func     # same ctypes function
fn.export_to_c()            # might export standalone C code
fn.dump_to_object()         # might dump binary
```
