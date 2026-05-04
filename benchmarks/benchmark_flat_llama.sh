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
NUM_ITERS="${6:-30}"
NUM_WARMUP="${7:-10}"

COMMON_ARGS=(
    --model "$MODEL"
    --tensor-parallel-size "$TP"
    --batch-size "$BATCH_SIZE"
    --input-len "$INPUT_LEN"
    --output-len "$OUTPUT_LEN"
    --num-iters "$NUM_ITERS"
    --num-iters-warmup "$NUM_WARMUP"
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
echo "[1/2] Baseline LlamaForCausalLM"
echo "---------------------------------------------"
python -m vllm.entrypoints.cli.main bench latency \
    "${COMMON_ARGS[@]}" \
    --output-json /tmp/bench_baseline.json

echo ""
echo "---------------------------------------------"
echo "[2/2] FlatLlamaForCausalLM (compile=none)"
echo "---------------------------------------------"
python -m vllm.entrypoints.cli.main bench latency \
    "${COMMON_ARGS[@]}" \
    --hf-overrides '{"architectures": ["FlatLlamaForCausalLM"]}' \
    --compilation-config '{"mode": "none"}' \
    --output-json /tmp/bench_flat.json

echo ""
echo "============================================="
echo "Results saved to:"
echo "  Baseline: /tmp/bench_baseline.json"
echo "  Flat:     /tmp/bench_flat.json"
echo "============================================="

python3 -c "
import json
with open('/tmp/bench_baseline.json') as f:
    baseline = json.load(f)
with open('/tmp/bench_flat.json') as f:
    flat = json.load(f)

bl = baseline['avg_latency']
fl = flat['avg_latency']
speedup = bl / fl

print()
print(f'  Baseline avg latency:  {bl:.4f}s')
print(f'  Flat avg latency:      {fl:.4f}s')
print(f'  Speedup:               {speedup:.3f}x')
print(f'  Baseline p50:          {baseline[\"percentiles\"][\"50\"]:.4f}s')
print(f'  Flat p50:              {flat[\"percentiles\"][\"50\"]:.4f}s')
print()
"
