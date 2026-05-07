# flat_model

**FULL_DECODE_ONLY** — this program only targets the BS=1 decode path.
Prefill is out of scope.

Create a "flat" model definition for a vLLM-supported model. The forward
pass becomes a single function with all parameters as plain tensors — no
nn.Module dispatch. The result must produce identical output to the original.

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

- `extract_all_layer_params()` — pull weights out of nn.Modules into flat lists
- `transformer_layer()` — one decoder block, all params explicit, no `self`
- `flat_forward()` — embedding → loop of transformer_layer → final norm

Use standard torch ops. For attention and MoE, vLLM's existing operators
are fine. No custom kernels. The goal is correctness first.

## The Test Loop

LOOP FOREVER until the flat model produces correct output:

1. **Start the server** using the user's `vllm serve` command, but add
   `--hf-overrides '{"architectures": ["Flat<Model>ForCausalLM"]}'` to
   use your flat model. Redirect output to a log file and run in background.
   Wait for "Application startup complete" in the log.

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
