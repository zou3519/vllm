#!/usr/bin/env python3
"""Benchmark BF16-input NVFP4 GEMV for Flat Llama QKV/O shapes.

This checks whether it is worth skipping activation FP4 quantization for
BS=1 decode projections whose input is naturally BF16.  The CUDA kernels are
built via the existing `flat_llama_gemv` load_inline extension.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
import triton

from vllm._custom_ops import create_fp4_output_tensors
from vllm.model_executor.models.flat_llama_gemv import (
    _ensure_compiled,
    nvfp4_gemv,
    nvfp4_gemv_bf16in,
)
from vllm.model_executor.models.flat_llama_kernels import triton_fp4_quant_rowmajor


CASES = {
    # output elements, packed FP4 bytes in the input dimension
    "qkv": (10240, 4096),
    "o": (8192, 4096),
}


def time_us(
    fn: Callable[[], None],
    warmup: int,
    iters: int,
    use_cudagraph: bool,
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if use_cudagraph:
        ms = triton.testing.do_bench_cudagraph(fn, rep=iters, return_mode="median")
        return ms * 1000.0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def bench_case(name: str, warmup: int, iters: int, use_cudagraph: bool) -> None:
    m, k_bytes = CASES[name]
    k_actual = k_bytes * 2
    k_scale = k_actual // 16
    device = torch.device("cuda")

    weight = torch.randint(0, 256, (m, k_bytes), dtype=torch.uint8, device=device)
    weight_scale = torch.randint(1, 4, (m, k_scale), dtype=torch.uint8, device=device)
    x_bf16 = torch.randn(k_actual, dtype=torch.bfloat16, device=device)
    scale_inv = torch.ones((), dtype=torch.float32, device=device)
    out = torch.empty(m, dtype=torch.bfloat16, device=device)

    x_fp4, x_scale = create_fp4_output_tensors(1, k_actual, device, False)

    # Pre-quantize once for the GEMV-only lower bound.
    triton_fp4_quant_rowmajor(
        x_bf16.view(1, k_actual), scale_inv, x_fp4, x_scale
    )
    torch.cuda.synchronize()

    quant_only_us = time_us(
        lambda: triton_fp4_quant_rowmajor(
            x_bf16.view(1, k_actual), scale_inv, x_fp4, x_scale
        ),
        warmup,
        iters,
        use_cudagraph,
    )
    fp4_gemv_us = time_us(
        lambda: nvfp4_gemv(
            weight, x_fp4.view(-1), weight_scale, x_scale.view(-1), out, 1.0
        ),
        warmup,
        iters,
        use_cudagraph,
    )
    quant_plus_fp4_us = time_us(
        lambda: (
            triton_fp4_quant_rowmajor(
                x_bf16.view(1, k_actual), scale_inv, x_fp4, x_scale
            ),
            nvfp4_gemv(
                weight, x_fp4.view(-1), weight_scale, x_scale.view(-1), out, 1.0
            ),
        ),
        warmup,
        iters,
        use_cudagraph,
    )
    bf16in_us = time_us(
        lambda: nvfp4_gemv_bf16in(weight, x_bf16, weight_scale, out, 1.0),
        warmup,
        iters,
        use_cudagraph,
    )

    payload_bytes = m * k_bytes + m * k_scale
    print(f"\n{name}: M={m}, K_actual={k_actual}")
    print(f"weight+scale payload: {payload_bytes / 1e6:.1f} MB")
    print(f"{'variant':<26} {'us':>8} {'TB/s payload':>13} {'vs fp4':>9}")
    print("-" * 62)
    for variant, us in [
        ("quant only", quant_only_us),
        ("fp4 GEMV only", fp4_gemv_us),
        ("quant + fp4 GEMV", quant_plus_fp4_us),
        ("bf16-input GEMV", bf16in_us),
    ]:
        tbs = payload_bytes / (us * 1e-6) / 1e12
        print(f"{variant:<26} {us:>8.3f} {tbs:>13.3f} {fp4_gemv_us / us:>8.3f}x")

    print(
        "bf16-input vs quant+fp4: "
        f"{quant_plus_fp4_us / bf16in_us:.3f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=[*CASES.keys(), "all"], default="all")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument(
        "--no-cudagraph",
        action="store_true",
        help="Use a normal event-timed Python launch loop instead of CUDA graph replay.",
    )
    args = parser.parse_args()

    _ensure_compiled()

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"benchmark mode: {'event loop' if args.no_cudagraph else 'CUDA graph replay'}")
    names = CASES.keys() if args.case == "all" else [args.case]
    for name in names:
        bench_case(name, args.warmup, args.iters, not args.no_cudagraph)


if __name__ == "__main__":
    main()
