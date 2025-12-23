#!/usr/bin/env python3
"""
Unit tests for fused LayerNorm + Residual kernel.

This script validates the correctness of the CUDA kernel by comparing
its output with PyTorch's native LayerNorm + add implementation.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import time
from typing import Tuple

# Import our fused kernel
from fused_kernels.fused_ln_residual import fused_layernorm_residual


def pytorch_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Reference implementation using PyTorch native operations."""
    # LayerNorm
    normalized = F.layer_norm(x, (x.size(-1),), weight, bias, eps)
    # Residual add
    output = normalized + residual
    return output


def test_correctness(
    batch: int = 2,
    seq_len: int = 128,
    hidden_size: int = 2560,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-5,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> Tuple[bool, float, float]:
    """
    Test correctness of fused kernel against PyTorch reference.
    
    Args:
        batch: Batch size
        seq_len: Sequence length
        hidden_size: Hidden dimension size
        dtype: Data type (float32, float16)
        eps: LayerNorm epsilon
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
    
    Returns:
        (passed, max_abs_error, max_rel_error)
    """
    device = torch.device("cuda")
    
    # Generate random inputs
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)
    residual = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)
    weight = torch.ones(hidden_size, device=device, dtype=dtype)
    bias = torch.zeros(hidden_size, device=device, dtype=dtype)
    
    # Compute reference output
    with torch.no_grad():
        output_ref = pytorch_reference(x, residual, weight, bias, eps)
    
    # Compute fused kernel output
    with torch.no_grad():
        output_fused = fused_layernorm_residual(x, residual, weight, bias, eps)
    
    # Compare results
    abs_error = torch.abs(output_fused - output_ref)
    max_abs_error = abs_error.max().item()
    
    rel_error = abs_error / (torch.abs(output_ref) + 1e-8)
    max_rel_error = rel_error.max().item()
    
    passed = torch.allclose(output_fused, output_ref, rtol=rtol, atol=atol)
    
    return passed, max_abs_error, max_rel_error


def benchmark_performance(
    batch: int = 2,
    seq_len: int = 128,
    hidden_size: int = 2560,
    dtype: torch.dtype = torch.float16,
    eps: float = 1e-5,
    num_warmup: int = 10,
    num_iterations: int = 100,
) -> Tuple[float, float, float]:
    """
    Benchmark performance of fused kernel vs PyTorch reference.
    
    Returns:
        (pytorch_time_ms, fused_time_ms, speedup)
    """
    device = torch.device("cuda")
    
    # Generate inputs
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)
    residual = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)
    weight = torch.ones(hidden_size, device=device, dtype=dtype)
    bias = torch.zeros(hidden_size, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(num_warmup):
        _ = pytorch_reference(x, residual, weight, bias, eps)
        _ = fused_layernorm_residual(x, residual, weight, bias, eps)
    
    torch.cuda.synchronize()
    
    # Benchmark PyTorch reference
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = pytorch_reference(x, residual, weight, bias, eps)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / num_iterations * 1000  # ms
    
    # Benchmark fused kernel
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = fused_layernorm_residual(x, residual, weight, bias, eps)
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / num_iterations * 1000  # ms
    
    speedup = pytorch_time / fused_time
    
    return pytorch_time, fused_time, speedup


def main():
    """Run all tests."""
    print("=" * 80)
    print("Fused LayerNorm + Residual Kernel Tests")
    print("=" * 80)
    
    # Test different configurations
    test_configs = [
        # (batch, seq_len, hidden_size, dtype, name)
        (2, 128, 2560, torch.float32, "FP32 - small batch"),
        (2, 128, 2560, torch.float16, "FP16 - small batch"),
        (1, 1, 2560, torch.float16, "FP16 - decode (batch=1, seq=1)"),
        (4, 256, 2560, torch.float16, "FP16 - large batch"),
        (1, 2048, 2560, torch.float16, "FP16 - long sequence"),
    ]
    
    print("\n" + "=" * 80)
    print("CORRECTNESS TESTS")
    print("=" * 80)
    
    all_passed = True
    for batch, seq_len, hidden_size, dtype, name in test_configs:
        print(f"\n{name}:")
        print(f"  Shape: [{batch}, {seq_len}, {hidden_size}], dtype: {dtype}")
        
        # Adjust tolerances for FP16
        rtol = 1e-3 if dtype == torch.float16 else 1e-4
        atol = 1e-3 if dtype == torch.float16 else 1e-5
        
        try:
            passed, max_abs_err, max_rel_err = test_correctness(
                batch, seq_len, hidden_size, dtype, rtol=rtol, atol=atol
            )
            
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  Status: {status}")
            print(f"  Max absolute error: {max_abs_err:.6e}")
            print(f"  Max relative error: {max_rel_err:.6e}")
            
            if not passed:
                all_passed = False
                print(f"  ERROR: Output mismatch (rtol={rtol}, atol={atol})")
        
        except Exception as e:
            all_passed = False
            print(f"  ✗ FAIL: {e}")
    
    # Performance benchmarks
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 80)
    
    bench_configs = [
        (1, 1, 2560, torch.float16, "FP16 - decode (batch=1, seq=1)"),
        (2, 128, 2560, torch.float16, "FP16 - small batch"),
        (4, 256, 2560, torch.float16, "FP16 - large batch"),
    ]
    
    for batch, seq_len, hidden_size, dtype, name in bench_configs:
        print(f"\n{name}:")
        print(f"  Shape: [{batch}, {seq_len}, {hidden_size}]")
        
        try:
            pytorch_time, fused_time, speedup = benchmark_performance(
                batch, seq_len, hidden_size, dtype
            )
            
            print(f"  PyTorch time: {pytorch_time:.4f} ms")
            print(f"  Fused time:   {fused_time:.4f} ms")
            print(f"  Speedup:      {speedup:.2f}x")
        
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Final summary
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
