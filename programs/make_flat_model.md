# flat_model

Create a "flat" model definition for a vLLM-supported model. The forward
pass becomes a single function with all parameters as plain tensors — no
nn.Module dispatch. The result must produce identical output to the original.

The flat model definition should preserve the original model's forward behavior
for both prefill and decode, and it should not assume batch size 1 unless the
target serve command or user explicitly narrows the scope. Some downstream
auto-optimize workflows may benchmark or specialize for BS=1 decode, but that
is an optimization target, not the default scope of this program.

## Setup

The user will give you a `vllm serve` command, e.g.:
```bash
vllm serve <model_name> --arg1 ... --arg2 ...
```

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

1. **Read the original model** in `vllm/model_executor/models/`. Understand
   the layer structure, parameter names, and forward pass.
2. **Ensure weights are downloaded.** If not, ask the user to run:
   `huggingface-cli download <model_name>`
3. **Create branch**: `git checkout -b rzou/flat_<model>`

## Implementation

Create two files in `vllm/model_executor/models/`:

### `flat_<model>.py` — Weight loading only

Mirrors the original model's nn.Module hierarchy so vLLM's weight loader
can populate tensors. Copy `load_weights()` from the original model.
The backbone `forward()` calls `flat_forward()` from the forward file.

### `flat_<model>_forward.py` — Flat forward pass

- `transformer_layer()` — one decoder block, all params explicit, no `self`
- `flat_forward()` — embedding → loop of transformer_layer → final norm

Only define those two functions. Pull weights/constants out of nn.Modules inside
`flat_forward()` and cache them on the model.

Inline the whole hidden layer body in `transformer_layer()`. Do not add helper
functions for RMSNorm, rotary embedding, activation, residual handling, or
linear wrappers. The only non-Python call sites inside `transformer_layer()`
should be:
- native PyTorch operators (`torch.*`, Tensor methods such as `.to()`/`.view()`,
  indexing, and assignment)
- `torch.nn.functional.*` ops
- vLLM's attention op
- vLLM's MoE op

Add short comments at the original module boundaries in the flat forward, e.g.
embedding, each decoder layer, attention qkv/rotary/KV-cache/attention/o_proj,
MLP router/experts, and final norm.

Specialize on the current target hardware and serve command when that removes
branches from `transformer_layer()`. Prefer the original model's native PyTorch
path for ordinary math, but keep attention and MoE on the vLLM CUDA path. If a
linear layer's quant method is known for the target, inline that specialized
linear path instead of preserving generic dispatch. The goal is correctness
first.

Make the KV-cache write explicit in the flat definition. Prefer a torch-native
cache update specialized to the active attention backend/cache layout/hardware,
then call vLLM's attention op directly, ideally
`torch.ops.vllm.unified_attention_with_output(...)`, instead of calling the
`Attention` module wrapper.

Do not torch.compile the flat model when testing or benchmarking. Turn it off
via the serve command, e.g. add `-cc.mode=none` (CompilationMode.NONE) to the
flat-model `vllm serve` command.

## The Test Loop

LOOP FOREVER until the flat model produces correct output:

1. **Start the server** using the user's `vllm serve` command, but add
   `--hf-overrides '{"architectures": ["Flat<Model>ForCausalLM"]}'` to
   use your flat model, and add `-cc.mode=none` so the flat model is not
   torch-compiled. Redirect output to a log file and run in background. Wait
   for "Application startup complete" in the log.

2. **Test correctness** — send these prompts and check the answers:
   - "What is 2+2?" → should answer 4
   - "Name the capital of France in one word" → Paris
   - "Write a recipe for chocolate chip cookies" → coherent, detailed recipe

3. **If server crashes**: the model runs in a subprocess — errors go to the
   log file, not your terminal. Read `tail -50 /tmp/flat_model_server.log`,
   fix, restart.

4. **If output is wrong**: compare against original model's forward pass.
   Check parameter extraction, op ordering, KV cache handling.

5. **If output is correct**: measure TPIT/TTIT for both the flat model and
   the original (non-flat) model using the user's `vllm serve` command
   without `--hf-overrides`. TPIT/TTIT means streaming inter-token latency:
   time each generated token arrives after the previous generated token.
   Do not use TPOT ("time per output token") from serving benchmarks as the
   primary number; TPOT is an aggregate derived from request latency and can
   hide chunking behavior. If using `vllm bench serve`, report the `ITL`
   metric (`mean_itl_ms`/`median_itl_ms`), not `TPOT`. Write
   `flat_config.txt` in the repo root:
   ```
   serve_cmd: <the user's original vllm serve command>
   forward_file: vllm/model_executor/models/flat_<model>_forward.py
   model_file: vllm/model_executor/models/flat_<model>.py
   original_tpit_ms: <measured>
   flat_baseline_tpit_ms: <measured>
   ```
   Commit everything (including `flat_config.txt`), done.
