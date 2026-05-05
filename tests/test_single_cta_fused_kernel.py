"""Test single-CTA fused add+RMSNorm+FP4 quant kernel against 2-kernel reference."""
import torch
import time


def create_test_tensors(N=8192, device="cuda"):
    """Create tensors matching Llama 70B dimensions."""
    hidden = torch.randn(1, N, dtype=torch.bfloat16, device=device)
    residual = torch.randn(1, N, dtype=torch.bfloat16, device=device)
    weight = torch.randn(N, dtype=torch.bfloat16, device=device)
    sf_scale_inv = torch.tensor([0.5], dtype=torch.float32, device=device)
    return hidden, residual, weight, sf_scale_inv


def run_two_kernel(hidden, residual, weight, sf_scale_inv, N=8192):
    """Reference: 2-kernel approach."""
    from vllm.model_executor.models.flat_llama_kernels import (
        _add_variance_kernel,
        _norm_fp4_quant_kernel,
    )
    from vllm._custom_ops import create_fp4_output_tensors

    device = hidden.device
    residual_out = torch.empty(1, N, dtype=torch.bfloat16, device=device)
    variance = torch.empty(1, dtype=torch.float32, device=device)
    fp4_out, scale_out = create_fp4_output_tensors(1, N, device, True)
    scale_bytes = scale_out.view(torch.uint8)
    scale_stride = scale_out.shape[1] * 4

    _add_variance_kernel[(1,)](
        hidden, residual, residual_out, variance,
        N=N, BLOCK=min(N, 4096),
    )
    _norm_fp4_quant_kernel[(N // 16,)](
        residual_out, weight, variance, sf_scale_inv,
        fp4_out, scale_bytes,
        N=N, SCALE_STRIDE=scale_stride,
    )
    return residual_out, fp4_out, scale_out


def run_single_cta(hidden, residual, weight, sf_scale_inv, N=8192):
    """Test: single-CTA fused kernel."""
    from vllm.model_executor.models.flat_llama_kernels import (
        triton_single_cta_fused_add_rms_norm_fp4_quant,
    )
    from vllm._custom_ops import create_fp4_output_tensors

    device = hidden.device
    residual_out = torch.empty(1, N, dtype=torch.bfloat16, device=device)
    fp4_out, scale_out = create_fp4_output_tensors(1, N, device, True)

    triton_single_cta_fused_add_rms_norm_fp4_quant(
        hidden, residual, residual_out,
        weight, sf_scale_inv,
        fp4_out, scale_out,
    )
    return residual_out, fp4_out, scale_out


def test_correctness(N=8192):
    """Verify single-CTA kernel matches 2-kernel reference."""
    print(f"Testing correctness (N={N})...")
    hidden, residual, weight, sf_scale_inv = create_test_tensors(N)

    ref_res, ref_fp4, ref_scale = run_two_kernel(
        hidden, residual, weight, sf_scale_inv, N)
    test_res, test_fp4, test_scale = run_single_cta(
        hidden, residual, weight, sf_scale_inv, N)

    # Residual output should match exactly (same BF16 add)
    res_match = torch.equal(ref_res, test_res)
    print(f"  residual_out match: {res_match}")
    if not res_match:
        diff = (ref_res.float() - test_res.float()).abs()
        print(f"    max diff: {diff.max().item():.6e}, mean: {diff.mean().item():.6e}")

    # FP4 packed bytes
    fp4_match = torch.equal(ref_fp4, test_fp4)
    print(f"  fp4_out match: {fp4_match}")
    if not fp4_match:
        mismatches = (ref_fp4 != test_fp4).sum().item()
        total = ref_fp4.numel()
        print(f"    mismatches: {mismatches}/{total} ({mismatches/total*100:.1f}%)")

    # Scales
    scale_match = torch.equal(ref_scale, test_scale)
    print(f"  scale_out match: {scale_match}")
    if not scale_match:
        ref_sb = ref_scale.view(torch.uint8)
        test_sb = test_scale.view(torch.uint8)
        mismatches = (ref_sb != test_sb).sum().item()
        total = ref_sb.numel()
        print(f"    mismatches: {mismatches}/{total} ({mismatches/total*100:.1f}%)")

    all_match = res_match and fp4_match and scale_match
    print(f"  OVERALL: {'PASS' if all_match else 'FAIL'}")
    return all_match


def benchmark(N=8192, warmup=200, iters=1000):
    """Benchmark both approaches."""
    hidden, residual, weight, sf_scale_inv = create_test_tensors(N)

    from vllm.model_executor.models.flat_llama_kernels import (
        _add_variance_kernel,
        _norm_fp4_quant_kernel,
        triton_single_cta_fused_add_rms_norm_fp4_quant,
    )
    from vllm._custom_ops import create_fp4_output_tensors

    device = hidden.device
    residual_out = torch.empty(1, N, dtype=torch.bfloat16, device=device)
    variance = torch.empty(1, dtype=torch.float32, device=device)
    fp4_out, scale_out = create_fp4_output_tensors(1, N, device, True)
    scale_bytes = scale_out.view(torch.uint8)
    scale_stride = scale_out.shape[1] * 4

    # Warmup + benchmark: 2-kernel
    for _ in range(warmup):
        _add_variance_kernel[(1,)](
            hidden, residual, residual_out, variance,
            N=N, BLOCK=min(N, 4096))
        _norm_fp4_quant_kernel[(N // 16,)](
            residual_out, weight, variance, sf_scale_inv,
            fp4_out, scale_bytes, N=N, SCALE_STRIDE=scale_stride)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _add_variance_kernel[(1,)](
            hidden, residual, residual_out, variance,
            N=N, BLOCK=min(N, 4096))
        _norm_fp4_quant_kernel[(N // 16,)](
            residual_out, weight, variance, sf_scale_inv,
            fp4_out, scale_bytes, N=N, SCALE_STRIDE=scale_stride)
    end.record()
    torch.cuda.synchronize()
    two_kernel_us = start.elapsed_time(end) / iters * 1000
    print(f"2-kernel:    {two_kernel_us:.1f} μs")

    # Warmup + benchmark: single-CTA
    fp4_out2, scale_out2 = create_fp4_output_tensors(1, N, device, True)
    for _ in range(warmup):
        triton_single_cta_fused_add_rms_norm_fp4_quant(
            hidden, residual, residual_out,
            weight, sf_scale_inv, fp4_out2, scale_out2)
    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        triton_single_cta_fused_add_rms_norm_fp4_quant(
            hidden, residual, residual_out,
            weight, sf_scale_inv, fp4_out2, scale_out2)
    end.record()
    torch.cuda.synchronize()
    single_cta_us = start.elapsed_time(end) / iters * 1000
    print(f"single-CTA:  {single_cta_us:.1f} μs")

    speedup = two_kernel_us / single_cta_us
    print(f"speedup:     {speedup:.2f}x")
    print(f"savings:     {two_kernel_us - single_cta_us:.1f} μs/call"
          f" → {(two_kernel_us - single_cta_us) * 160 / 1000:.1f} ms/step (160 calls)")


if __name__ == "__main__":
    test_correctness()
    print()
    benchmark()
