# auto_optimize

**FULL_DECODE_ONLY** — this program only targets the BS=1 decode path.
Prefill is out of scope.

You are a GPU inference optimization engineer. Your job is to minimize
BS=1 decode latency (TPIT) for a flat vLLM model.

## Setup

Read `flat_config.txt` in the repo root for the serve command, file
paths, and baseline TPITs (written by `flat_model.md`).

Also read any model-specific optimization summary in the repo root, e.g.
`flat_<model>_optimization.md`, plus the current `results.tsv` if it exists.
Treat that history as the source of truth for what already failed, what was
kept, and what the current best verified TPIT is.

Assume the current shell is already inside the correct conda environment.
That environment should already have `torch` and `vllm` installed. Do not
create a virtualenv and do not install packages. Confirm the environment
before doing any work:
```bash
python -c "import torch, vllm; print(torch.__version__); print(vllm.__file__)"
command -v vllm
```
If either `torch` or `vllm` is missing, stop immediately and shout to the
human that the conda environment is broken and needs to be fixed.

To set up:

1. **Read `flat_config.txt`** and the flat model forward file. Understand
   every op in `transformer_layer()` — what it does, what shapes flow
   through it.
2. **Compute the roofline**. For each linear projection, calculate:
   ```
   weight_bytes = M × K_packed + M × (K_packed / 8)   # FP4 data + scales
   roofline_μs  = weight_bytes / peak_HBM_bandwidth
   ```
   Sum across projections × num_layers for the total model roofline.
3. **Verify the flat no-compile FULL_DECODE_ONLY baseline TPIT** from
   `flat_config.txt` by re-measuring (see Benchmarking below). This loop
   optimizes that flat path, not a torch.compile variant.
4. **Initialize results.tsv** with just the header row if it does not already
   exist. If it exists, append to it; do not erase old experiments.
5. **Confirm and go**: tell the user the flat baseline TPIT, the original
   model TPIT (from `flat_config.txt`), the roofline, and the gap.
   Then start optimizing.

## Profiling

Use vLLM's built-in torch profiler integration to get kernel-level
traces. See https://docs.vllm.ai/en/stable/contributing/profiling/#openai-server
for how to profile the server.

For the iterative loop, TPIT measurement (below) is usually sufficient.
Use profiling when you need to understand WHERE time is spent within a
layer.

## Benchmarking

Start the server using the flat serve command from `flat_config.txt`, or the
user's original `vllm serve` command plus the flat `--hf-overrides`. Always add
`--max-cudagraph-capture-size 1`, `-cc.mode=none`, and
`-cc.cudagraph_mode=full_decode_only` for this auto-optimize loop. Redirect
output to a log file and run in background. Wait for "Application startup
complete" in the log. Before each start, check for and stop stale vLLM
server/EngineCore processes from previous runs; after each run, stop the server
and verify none remain.

Do not change target-defining serve flags while comparing optimizations. In
particular, keep `--max-num-batched-tokens` exactly as specified by the target
serve command, even if smaller values look faster. If you intentionally measure
an off-target serve flag, log it as invalid/off-target and do not count it as
the best TPIT.

For the auto-optimize target, torch.compile must be disabled while full decode
CUDA graphs remain enabled. The startup log must show
`CompilationMode.NONE` and `CUDAGraphMode.FULL_DECODE_ONLY` (or equivalent
effective config). If vLLM overrides cudagraph mode to `NONE`, treat that as a
target setup failure, not as a valid benchmark.

**Measure engine-side TPIT** with vLLM's Prometheus metrics, but validate the
metric count:

1. Send one warmup streaming request.
2. Read `/metrics` and save
   `vllm:inter_token_latency_seconds_sum/count` and
   `vllm:time_to_first_token_seconds_sum/count`.
3. Send 3-5 identical streaming requests with fixed prompt, `max_tokens`,
   temperature, and seed settings.
4. Read `/metrics` again and compute delta means:
   `mean_itl_ms = 1000 * delta_sum / delta_count`.
5. Check that `delta_count == num_requests * (generated_tokens_per_request - 1)`.
   If it does not, an engine output event may contain multiple token IDs; do
   not report the histogram mean as TPIT without explaining the mismatch.

For context, periodically benchmark the regular non-flat model with the normal
vLLM torch.compile path using the same prompt, token count, stream interval, and
Prometheus count validation. This comparison answers whether the flat path is
actually better than the production compiled path, not just better than an
earlier flat baseline.

This measures engine-core inter-token latency. It is the right signal for
kernel/runtime optimization when the count validation passes, but it is not
necessarily client-visible TTIT. Measure client-visible TTIT from streaming
responses only when each chunk corresponds to one generated token; if
`--stream-interval > 1`, either rerun with `--stream-interval 1` for client
timing or clearly label the result as chunk timing.

You can still send a streaming request with curl for sanity:
```bash
curl -s localhost:<port>/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "<model>", "messages": [{"role": "user", "content": "Write a paragraph about AI."}], "max_tokens": 150, "stream": true}'
```
Do not report SSE chunk spacing as TPIT when `--stream-interval > 1`. Run
3-5 times; TPIT should be consistent enough to distinguish real regressions
from runtime noise.

**Test quality** after every change with these prompts:
- "What is 2+2?" → should answer 4
- "Name the capital of France in one word" → Paris
- "Write a recipe for chocolate chip cookies" → coherent, detailed recipe

Red flags: repetition loops ("Hello!!!!!!!"), partially correct + garbage
("4! !!!!!!"), complete nonsense. Any of these means the optimization
broke precision — revert.

## What you CAN do

- Modify the flat model forward file — fuse kernels, replace ops, change
  buffer allocation, write custom kernels (Triton, CuTe DSL, or raw CUDA)
- Add new kernel files (e.g. `flat_<model>_gemv.py`)
- Write benchmarking scripts
- Search the internet for state-of-the-art kernel implementations,
  optimization techniques, and relevant papers
- Specialize on BS=1 when the target is explicitly BS=1 decode. BS=1-only
  branches, fixed top-k kernels, capture-size-1 assumptions, and specialized
  routing/finalize layouts are allowed if the generic path still works or the
  specialization is guarded by shape checks.
- Change internal weight/layout formats for performance if the flat loader or
  cached params can produce the new layout once, correctness is preserved, and
  the target serve command still loads the same checkpoint.
- Inline backend wrappers further when they hide concrete pointwise,
  reduction, routing, quantization, or allocation work. For example, a method
  named `forward_cuda` may still be Python orchestration; inspect until you
  see the real CUDA/custom ops.

## What you CANNOT do

- Modify the flat model weight-loading file (`flat_<model>.py`) — it must
  continue to load the same checkpoint
- Change the model's numerical output significantly — quality must be preserved
- Install new packages

## The goal

**Get TPIT as close to the memory-bandwidth roofline as possible.** BS=1
decode is entirely memory-bound — the roofline is total weight reads
divided by peak HBM bandwidth. Typical efficiency is 60-70%; getting
above 70% is good.

## Logging results

Log every experiment to `results.tsv` (tab-separated). During the optimization
loop, append results immediately so the process can be killed without losing
measurements. Keep it uncommitted while iterating unless the user asks for a
snapshot or you are explicitly wrapping up a documented optimization phase.

```
commit	tpit_ms	status	description
```

- commit: short git hash (7 chars)
- tpit_ms: measured TPIT in ms (0.0 for crashes)
- status: `keep`, `discard`, or `crash`
- description: what this experiment tried

Example:
```
commit	tpit_ms	status	description
a1b2c3d	12.50	keep	baseline
b2c3d4e	12.07	keep	fused norm+quant (Triton single-CTA)
c3d4e5f	12.55	discard	triton GEMV for QKV (slower than CUTLASS)
d4e5f6g	0.0	crash	custom attention kernel (illegal memory access)
```

## The experiment loop

LOOP FOREVER:

1. **Profile**: look at the per-layer breakdown. Identify the slowest op.
2. **Pick an optimization**: kernel fusion, GEMV replacement, or op elimination.
3. **Implement and commit**.
4. **Kill the old server, start a new one** with the updated code.
5. **Test quality** with the three test prompts.
6. **Measure TPIT** (3-5 runs).
7. **Log** the result to results.tsv.
8. If TPIT improved AND quality is preserved: **keep** the commit.
9. If TPIT regressed OR quality degraded: **revert with `git revert`** so the
   failed experiment remains visible in history. Avoid destructive reset unless
   the human explicitly asks for it.
10. Go to 1.

## Optimization ideas (roughly ordered by impact)

### Kernel fusions
Fuse adjacent small kernels into one launch. Low risk, no quality impact.
- LayerNorm + quantize → one Triton kernel
- Activation + multiply + quantize → one Triton kernel
- RoPE + KV cache write → one Triton kernel

### GEMV for large-K projections
For BS=1, the GEMM is really a matrix-vector multiply. A custom GEMV
kernel can beat CUTLASS when K is large (the GEMV streams weights from
HBM with better cache utilization than tensor cores at M=1).

For quantized MoE, this is only true if the custom path still uses the right
math primitive. Scalar BF16xMXFP4 dequant-FMA GEMV prototypes were much slower
than FlashInfer TRTLLM tensor-core MoE. A serious MoE replacement should use
tensor-core/blockscaled GEMM primitives, CuTe/CUTLASS-style MMA, or a proven
low-latency MoE kernel, not one threadblock doing scalar dot products per row.

### BF16-input GEMV (skip activation quantize)
Instead of quantize → FP4 GEMV, feed BF16 activations directly to a GEMV
that only dequantizes the weights. Eliminates a kernel launch and avoids
activation quantization error. The BF16 input is tiny (one vector) so
the 4× larger read is negligible vs weight reads.

### Pre-allocated buffers
Allocate intermediate tensors once and reuse across layers. Eliminates
per-layer `torch.empty` overhead inside CUDA graphs.

Preallocation is a hypothesis, not a guaranteed win. Reusing MoE outputs,
letting FlashInfer allocate padded output, and persistent decode MoE output
buffers all regressed in prior GPT-OSS experiments. Benchmark every allocation
change instead of assuming it helps.

### Fuse pointwise/reduction work around heavyweight kernels
Low-latency decode usually wants fewer launches. Look for pointwise and
reduction work adjacent to a heavyweight kernel and either fuse it into an
existing custom kernel or into a new adjacent kernel. Good candidates include
RMSNorm, residual add, activation quantization, top-k packing, routing-weight
application, final reduction, RoPE, and KV-cache writes. KV-cache writes can
often be fused into surrounding RoPE/QKV handling.

If an existing backend kernel already fuses the work internally, do not split it
out unless you have a concrete replacement. The FlashInfer native MoE finalizer
beat a custom Triton `do_finalize=False` top-k finalizer in prior GPT-OSS work.

## Gotchas from prior work

These bugs wasted hours. Watch for them:

- **Quality degrades silently**: the model produces plausible-looking but
  wrong output (repetition loops, trailing garbage). Always test quality
  after every change — don't just measure TPIT.

- **GEMV precision vs CUTLASS**: custom GEMV kernels (scalar FMA) produce
  slightly different results from CUTLASS (tensor core MMA). This ~0.008
  max diff per layer compounds across deep models. Only use GEMV for
  projections whose output enters the residual stream (e.g. O projection,
  down projection). QKV and Gate+Up feed through attention which amplifies
  the error.

- **MoE scalar GEMV is not near-optimal**: for GPT-OSS MXFP4/MXFP8, raw CUDA
  scalar dequant-FMA MoE kernels were 15-22ms TPIT, far slower than FlashInfer.
  If replacing MoE, keep tensor-core blockscaled math and specialize on the
  actual shape (BS=1, top-k=4, padded dimensions) instead of writing a generic
  scalar GEMV.

- **GPT-OSS w13 row layout is interleaved**: gate/up rows are paired as
  `2*i` and `2*i+1`; do not assume half-split gate then up rows when writing
  custom kernels.

- **Triton constexpr divergence**: the same Triton kernel compiled with
  different `tl.constexpr` values produces subtly different floating-point
  results. If you have two code paths (e.g. CUTLASS vs GEMV) that both
  quantize the same data, they MUST use the same compiled kernel. If the
  GEMV needs a different scale layout, quantize once with the CUTLASS
  kernel and post-process the scales.

- **Server runs in subprocess**: the model runs in vLLM's EngineCore
  subprocess. Crashes and prints go to the log file, not your terminal.
  Always check the log.

- **FP8 KV cache stored as uint8**: vLLM stores FP8 KV cache as
  `torch.uint8`, not `torch.float8_e4m3fn`. If writing custom KV cache
  kernels, check for both dtypes.

- **CUDA graph capture**: torch.compile is disabled, but full decode CUDA
  graphs are enabled. Use `--max-cudagraph-capture-size 1` because this program
  only measures BS=1 decode. Do not add `torch.cuda.synchronize()`, `.item()`,
  or Python-side tensor value checks to the forward path unless the diagnostic
  is temporary, guarded against graph capture, and removed before committing.

- **Keep FULL_DECODE_ONLY even with torch.compile off**: the target is
  `CompilationMode.NONE` plus `CUDAGraphMode.FULL_DECODE_ONLY`, not eager-only
  execution. The full decode graph usually matters more than small Python
  cleanups.

- **Backend selection is part of the benchmark**: environment variables such
  as MoE backend selectors can materially change the op sequence. Assert or
  log the selected attention/MoE backend and specialize only for the backend
  used by the target serve command.

- **FlashInfer autotune/cache matters**: disabling FlashInfer autotune or
  forcing alternative cached tactics can regress. Log cache/tactic experiments
  separately. If an autotune cache miss happens because a shape changed, wait
  for startup tuning to complete before benchmarking.

- **PDL can matter**: for FlashInfer TRTLLM MoE on GPT-OSS, disabling PDL
  quality-passed but regressed TPIT. Leave PDL enabled unless the experiment is
  specifically testing it.

- **CUDA device mismatch**: when running standalone kernel benchmarks,
  use `CUDA_VISIBLE_DEVICES=<gpu>` to match the server's device. Using
  `device='cuda:3'` directly (without CUDA_VISIBLE_DEVICES) can cause
  illegal memory access errors due to compilation targeting the wrong
  device context.

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you
should continue. The human may be away and expects you to work
autonomously until manually stopped. If you run out of ideas, re-read
the forward file, study the roofline gaps, try combining previous
near-misses, or try more radical kernel designs.
