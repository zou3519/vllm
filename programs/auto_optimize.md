# auto_optimize

**FULL_DECODE_ONLY** — this program only targets the BS=1 decode path.
Prefill is out of scope.

You are a GPU inference optimization engineer. Your job is to minimize
BS=1 decode latency (TPIT) for a flat vLLM model.

## Setup

Read `flat_config.txt` in the repo root for the serve command, file
paths, and baseline TPITs (written by `flat_model.md`).

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
3. **Verify the baseline TPIT** from `flat_config.txt` by re-measuring
   (see Benchmarking below).
4. **Initialize results.tsv** with just the header row.
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

Start the server using the user's `vllm serve` command. Redirect output
to a log file and run in background. Wait for "Application startup
complete" in the log.

**Measure TPIT** by sending a streaming request:
```bash
curl -s localhost:<port>/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "<model>", "messages": [{"role": "user", "content": "Write a paragraph about AI."}], "max_tokens": 150, "stream": true}'
```
Parse the SSE stream: `TPIT = (total_time - first_token_time) / (tokens - 1)`.
Run 3-5 times — TPIT should be consistent (±0.1ms) with CUDA graphs.

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

Log every experiment to `results.tsv` (tab-separated). Do NOT commit
this file — leave it untracked.

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
9. If TPIT regressed OR quality degraded: **revert** (`git reset --hard HEAD~1`).
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

### BF16-input GEMV (skip activation quantize)
Instead of quantize → FP4 GEMV, feed BF16 activations directly to a GEMV
that only dequantizes the weights. Eliminates a kernel launch and avoids
activation quantization error. The BF16 input is tiny (one vector) so
the 4× larger read is negligible vs weight reads.

### Pre-allocated buffers
Allocate intermediate tensors once and reuse across layers. Eliminates
per-layer `torch.empty` overhead inside CUDA graphs.

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

- **CUDA graph capture**: no `torch.cuda.synchronize()`, `.item()`, or
  Python-side tensor value checks in the forward path. These break graph
  capture. Guard diagnostic code with
  `if not torch.cuda.is_current_stream_capturing()`.

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
