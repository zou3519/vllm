---
name: make-flat-model
description: Create and validate a flat vLLM model implementation with explicit tensor parameters, no nn.Module dispatch in the forward path, parity checks, and flat-vs-original benchmarking.
---

# flat_model

**FULL_DECODE_ONLY test target** — flat-model benchmarking and optimization
normally focus on BS=1 decode with full decode CUDA graphs. The flat model
definition itself should still preserve prefill/decode behavior unless the user
explicitly asks for a decode-only specialization.

Create a "flat" model definition for a vLLM-supported model. The forward
pass becomes a single explicit function over plain tensor parameters, with no
`nn.Module` dispatch. The result must produce identical output to the original
up to floating point tolerances.

By default, preserve both prefill and decode behavior and do not assume BS=1.
BS=1 decode is a common downstream optimization target, but only specialize for
decode-only behavior when the user explicitly asks.

Flat-model testing and optimization should use torch.compile disabled and full
decode CUDA graphs enabled:
- `-cc.mode=none`
- `-cc.cudagraph_mode=full_decode_only`

When comparing flat vs original or later optimization results, keep target
serve flags fixed, especially throughput/capacity flags such as
`--max-num-batched-tokens`.

## Setup

The user will give you a `vllm serve` command, e.g.:
```bash
vllm serve <model_name> --arg1 ... --arg2 ...
```

Assume the current shell is already inside the correct conda environment.
This workflow intentionally uses the active conda environment rather than the
repo's normal `uv`/`.venv` workflow, because flat-model work must run against
the serving environment selected by the user. That environment should already
have `torch` and `vllm` installed. Do not create a virtualenv and do not
install packages. Confirm the environment before doing any work:
```bash
python -c "import torch, vllm; print(torch.__version__); print(vllm.__file__)"
command -v vllm
```
If either `torch` or `vllm` is missing, stop immediately and report that the
active conda environment is broken, including the command output.

1. **Read the original model** in `vllm/model_executor/models/`. Understand
   the layer structure, parameter names, and forward pass.
   Also inspect the Hugging Face reference implementation for the target model
   when available. It is often written in native torch ops and can clarify math,
   tensor shapes, and operation ordering. If you need the reference
   implementation and it is not available locally, ask the human to download or
   provide it. Use vLLM as the source of truth for serving integration, weight
   loading, attention/KV-cache behavior, and the selected backend path.
2. **Ensure weights are downloaded.** If not, ask the user to run:
   `huggingface-cli download <model_name>`
3. **Create branch**: `git checkout -b <user-prefix>/flat_<model>`

## Implementation

Create two files in `vllm/model_executor/models/`:

Registration checklist:
- Define the new model class name, e.g. `Flat<Model>ForCausalLM`.
- Use the same class name in `--hf-overrides '{"architectures": [...]}'`.
- Add any required import or registry wiring so vLLM can discover the class.
- Confirm the server imports the edited repo source, not an older installed
  package; when needed, put the repo root first in `PYTHONPATH`.

### `flat_<model>.py` — Weight loading only

Mirrors the original model's nn.Module hierarchy so vLLM's weight loader
can populate tensors. Copy `load_weights()` from the original model.
The backbone `forward()` calls `flat_forward()` from the forward file.

### `flat_<model>_forward.py` — Flat forward pass

- `extract_all_layer_params()` or cached-param setup — pull weights out of
  nn.Modules into flat lists
- `transformer_layer()` — one decoder block, all params explicit, no `self`
- `flat_forward()` — embedding → loop of transformer_layer → final norm

Pull weights/constants out of nn.Modules inside `flat_forward()` or a small
extraction helper and cache them on the model.

Inline the whole hidden layer body in `transformer_layer()`. This function
should not call any helper Python functions. For example, do not add helper
functions for RMSNorm, rotary embedding, activation, residual handling, or
linear wrappers. The only non-Python call sites inside `transformer_layer()`
should be:
- native PyTorch operators (`torch.*`, Tensor methods such as `.to()`/`.view()`,
  indexing, and assignment)
- `torch.nn.functional.*` ops
- vLLM's attention op
- the concrete CUDA ops used by vLLM's selected MoE backend

Note that you should not include `torch.ops.*` calls because those
are custom operators.

Add short comments at the original module boundaries in the flat forward, e.g.
embedding, each decoder layer, attention qkv/rotary/KV-cache/attention/o_proj,
MLP router/experts, and final norm.

Specialize on the current target hardware and serve command when that removes
branches from `transformer_layer()`. Prefer the original model's native PyTorch
path for ordinary math, but keep attention and MoE on the vLLM CUDA path. If a
linear layer's quant method is known for the target, inline that specialized
linear path instead of preserving generic dispatch. The goal is correctness
first.

Inline the selected MoE `forward_cuda` path too. Do not call
`FusedMoE.forward_cuda()` or runner/custom-op wrappers when they only dispatch
through Python to a concrete backend. Pull the backend's tensor weights,
scales, biases, routing constants, and workspace/output allocation into
`flat_forward()`'s cached params, then call the concrete CUDA ops directly in
`transformer_layer()`. For example, for GPT-OSS on FlashInfer TRTLLM
MXFP4/MXFP8, inline the NoDPEP monolithic prepare, FlashInfer MXFP8 activation
quantization, and `trtllm_fp4_block_scale_moe(...)` call. This exposes
pointwise/reduction-shaped work such as activation quantization, scale
reshaping, routing casts, top-k packing, weighting, and final combine instead
of hiding it behind `forward_cuda`.

Do the same inspection for any method named `forward_cuda`: the name does not
mean it is a single CUDA op. If it only validates, reshapes, allocates, selects
backends, or calls another wrapper before reaching the real kernel, inline that
body until `transformer_layer()` shows the concrete PyTorch ops and concrete
CUDA/custom ops that actually run for the target serve command.

Record backend selectors that are required for the chosen path. For example,
GPT-OSS MXFP4/MXFP8 needs `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1` to select
the intended FlashInfer TRTLLM MoE path. The flat serve command in
`flat_config.txt` should include these environment variables, not rely on
ambient shell state.

For MoE models, capture the exact loaded layout and padded dimensions in the
flat parameter cache or summary notes. GPT-OSS `w13` gate/up rows are
interleaved pairs (`2*i`, `2*i+1`), and FlashInfer TRTLLM pads the relevant
dimensions to kernel-friendly sizes. These details matter for later BS=1
custom kernels and prevent wrong half-split assumptions.

Make the KV-cache write explicit in the flat definition. Prefer a torch-native
cache update specialized to the active attention backend/cache layout/hardware,
then call vLLM's attention op directly, ideally
`torch.ops.vllm.unified_attention_with_output(...)`, instead of calling the
`Attention` module wrapper.

Do not torch.compile the flat model when testing or benchmarking. Turn it off
via the serve command, e.g. add `-cc.mode=none` (CompilationMode.NONE) to the
flat-model `vllm serve` command, and also add
`-cc.cudagraph_mode=full_decode_only` so BS=1 decode still uses full CUDA
graphs. For BS=1 decode-only optimization, set
`--max-cudagraph-capture-size 1`. If the log says cudagraph mode was overridden
to `NONE`, treat that as a setup failure and fix the serve command/config before
benchmarking.

When the user later narrows optimization to BS=1 decode, BS=1-specific branches
are allowed in the flat forward or kernels if guarded by shape checks. The base
flat model should still preserve prefill/decode correctness unless the user
explicitly accepts a decode-only model.

## The Test Loop

LOOP FOREVER until the flat model produces correct output:

1. **Start the server** using the user's `vllm serve` command, but add
   `--hf-overrides '{"architectures": ["Flat<Model>ForCausalLM"]}'` to
   use your flat model. Redirect output to a log file and run in background.
   Wait for "Application startup complete" in the log.
   If the agent is getting killed by OOM attribution while the vLLM server is
   alive, start vLLM with `systemd-run --user --collect --unit=<name>` as a
   transient user service and pass required env vars explicitly, especially
   `PATH`, `HOME`, backend selector env vars, and offline/cache env vars. When
   running from a console entry point, put the repo root first in `PYTHONPATH`
   so the server imports the edited flat-model source instead of an older
   installed package. Check `systemctl --user status <name>.service` and
   confirm the server cgroup is under `user@<uid>.service/app.slice/`, then stop it with
   `systemctl --user stop <name>.service`.

2. **Test correctness** — send these prompts and check the answers:
   - "What is 2+2?" → should answer 4
   - "Name the capital of France in one word" → Paris
   - "Write a recipe for chocolate chip cookies" → coherent, detailed recipe
   Also run GSM8K as a later, slower correctness check once the flat model is
   stable. Ask the human for the correct expected GSM8K number for the target
   model/configuration, then compare the flat model against that number.

3. **If server crashes**: the model runs in a subprocess — errors go to the
   log file, not your terminal. Read `tail -50 /tmp/flat_model_server.log`,
   fix, restart.

4. **If output is wrong**: compare against original model's forward pass.
   Check parameter extraction, op ordering, KV cache handling.

5. **If output is correct**: measure TPIT for both the flat model and
   the original (non-flat) model using the user's `vllm serve` command
   without `--hf-overrides`. Also measure the regular model in its normal
   torch.compile configuration unless the user explicitly asks for eager-only
   comparison; this is the number that says whether the flat path beats
   production vLLM. TPIT/TTIT means streaming inter-token latency:
   time each generated token arrives after the previous generated token.
   Measure engine-side TPIT by reading `/metrics` before and after a fixed
   workload and using `vllm:inter_token_latency_seconds` only after validating
   the delta count. For N requests that each generate T tokens, the expected
   inter-token count is `N * (T - 1)`. If the metric count differs, an engine
   output event may contain multiple token IDs, so do not report the histogram
   mean as TPIT without explaining the mismatch. Measure client-visible TTIT
   from streaming responses only when each chunk corresponds to one generated
   token; if the serve command uses `--stream-interval > 1`, either rerun with
   `--stream-interval 1` for client timing or clearly label the result as
   chunk timing. Do not use TPOT ("time per output token") from serving
   benchmarks as the primary number; TPOT is an aggregate derived from request
   latency and can hide chunking behavior. If using `vllm bench serve`, report
   the `ITL` metric (`mean_itl_ms`/`median_itl_ms`), not `TPOT`, and verify it
   is token-level rather than chunk-level for the selected stream interval. Write
   `flat_config.txt` in the repo root:
   ```
   serve_cmd: <the user's original vllm serve command>
   flat_serve_cmd: <serve_cmd plus --hf-overrides and flat baseline flags>
   forward_file: vllm/model_executor/models/flat_<model>_forward.py
   model_file: vllm/model_executor/models/flat_<model>.py
   original_tpit_ms: <measured>
   flat_baseline_tpit_ms: <measured>
   tpit_benchmark: <exact metric/workload used>
   ```
   Also write or update a concise `flat_<model>_optimization.md` summary if the
   optimization history has become nontrivial. Include the run command, current
   flat-vs-regular torch.compile TPIT, fused ops, known failed ideas, and links
   to the relevant flat files and `results.tsv`.

   Commit everything (including `flat_config.txt` and the summary if created),
   done.

## Lessons to preserve for the optimizer

- Flat testing normally means `CompilationMode.NONE` plus
  `CUDAGraphMode.FULL_DECODE_ONLY`, not torch.compile and not eager-only.
- Capture size 1 is appropriate for BS=1 decode optimization, but target serve
  flags such as `--max-num-batched-tokens` should remain unchanged.
- Compare against regular vLLM with torch.compile; flat can improve over its
  own baseline while still losing to production compiled vLLM.
- Keep required backend env vars in `flat_config.txt`.
- Expose pointwise/reduction work in the flat forward. The optimizer needs to
  see residual add, RMSNorm, quantization, RoPE, KV-cache write, top-k/routing
  prep, and final reduction boundaries to decide what to fuse.
- Keep the flat forward shaped so later optimization can attempt aggressive
  vertical and horizontal fusion across adjacent decode-only operations,
  especially pointwise, reduction, quantization, RoPE, cache-write, routing, and
  same-shape independent streams.
- Do not make the flat structure hostile to tuning fused kernels: preserve clear
  tensor shapes, strides, backend constants, and cached one-time layout
  transforms so an optimizer can tune tile sizes, launch grids, branch
  specialization, and cached buffers before abandoning a quality-preserving
  fusion.
- Do not assume custom scalar GEMV is a good MoE replacement. For quantized MoE,
  future custom kernels should preserve tensor-core/blockscaled math or build
  directly on a proven low-latency backend.
