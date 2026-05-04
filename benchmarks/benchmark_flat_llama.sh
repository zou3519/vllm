#!/bin/bash
# Benchmark flat Llama vs baseline Llama for low-latency (batch size 1) decode.
#
# Usage:
#   bash benchmarks/benchmark_flat_llama.sh [MODEL] [TP]
#
# Example:
#   bash benchmarks/benchmark_flat_llama.sh nvidia/Llama-3.3-70B-Instruct-NVFP4 4

set -euo pipefail

MODEL="${1:-nvidia/Llama-3.3-70B-Instruct-NVFP4}"
TP="${2:-4}"
BATCH_SIZE="${3:-1}"
INPUT_LEN="${4:-128}"
OUTPUT_LEN="${5:-128}"
NUM_ITERS="${6:-10}"
NUM_WARMUP="${7:-3}"

COMMON_ARGS=(
    --model "$MODEL"
    --tensor-parallel-size "$TP"
    --batch-size "$BATCH_SIZE"
    --input-len "$INPUT_LEN"
    --output-len "$OUTPUT_LEN"
    --num-iters "$NUM_ITERS"
    --num-iters-warmup "$NUM_WARMUP"
    --max-model-len 4096
)

echo "============================================="
echo "Flat Llama Latency Benchmark"
echo "============================================="
echo "Model:      $MODEL"
echo "TP:         $TP"
echo "Batch size: $BATCH_SIZE"
echo "Input len:  $INPUT_LEN"
echo "Output len: $OUTPUT_LEN"
echo "Iterations: $NUM_ITERS (warmup: $NUM_WARMUP)"
echo ""

echo "---------------------------------------------"
echo "[1/4] Standard LlamaForCausalLM (default: compile + CG)"
echo "---------------------------------------------"
python -m vllm.entrypoints.cli.main bench latency \
    "${COMMON_ARGS[@]}" \
    --output-json /tmp/bench_standard.json

echo ""
echo "---------------------------------------------"
echo "[2/4] Standard LlamaForCausalLM (no compile, FULL CG)"
echo "---------------------------------------------"
python -m vllm.entrypoints.cli.main bench latency \
    "${COMMON_ARGS[@]}" \
    --compilation-config '{"mode": "none", "cudagraph_mode": "full"}' \
    --output-json /tmp/bench_standard_nocg.json

echo ""
echo "---------------------------------------------"
echo "[3/4] FlatLlamaForCausalLM (no compile, no CG)"
echo "---------------------------------------------"
python -m vllm.entrypoints.cli.main bench latency \
    "${COMMON_ARGS[@]}" \
    --hf-overrides '{"architectures": ["FlatLlamaForCausalLM"]}' \
    --compilation-config '{"mode": "none"}' \
    --output-json /tmp/bench_flat_nocg.json

echo ""
echo "---------------------------------------------"
echo "[4/4] FlatLlamaForCausalLM (no compile, FULL CG)"
echo "---------------------------------------------"
python -m vllm.entrypoints.cli.main bench latency \
    "${COMMON_ARGS[@]}" \
    --hf-overrides '{"architectures": ["FlatLlamaForCausalLM"]}' \
    --compilation-config '{"mode": "none", "cudagraph_mode": "full"}' \
    --output-json /tmp/bench_flat_cg.json

echo ""
echo "============================================="
echo "Results Summary"
echo "============================================="

python3 -c "
import json, os

configs = [
    ('Standard (compile+CG)',   '/tmp/bench_standard.json'),
    ('Standard (no compile, CG)', '/tmp/bench_standard_nocg.json'),
    ('Flat (no compile, no CG)', '/tmp/bench_flat_nocg.json'),
    ('Flat (no compile, CG)',    '/tmp/bench_flat_cg.json'),
]

results = {}
for name, path in configs:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        results[name] = data

if not results:
    print('No results found')
    exit()

best = min(r['avg_latency'] for r in results.values())

print()
print(f'  {\"Config\":<35s} {\"Avg (s)\":>10s} {\"p50 (s)\":>10s} {\"vs best\":>10s}')
print(f'  {\"-\"*35} {\"-\"*10} {\"-\"*10} {\"-\"*10}')

for name, data in results.items():
    avg = data['avg_latency']
    p50 = data['percentiles']['50']
    ratio = avg / best
    print(f'  {name:<35s} {avg:>10.3f} {p50:>10.3f} {ratio:>9.2f}x')

# Per-decode-step estimate (total - ~prefill) / output_tokens
output_len = $OUTPUT_LEN
print()
print(f'  Estimated per-decode-step (total / {output_len} output tokens):')
for name, data in results.items():
    ms_per_step = data['avg_latency'] / output_len * 1000
    print(f'    {name:<35s} {ms_per_step:>8.1f} ms/step')
print()
"
