"""
Fused Add kernel - minimal viable fusion for residual connections.

This is a much simpler and more realistic kernel than fused LN+residual,
designed to actually work with Pre-LN GPTNeoX architecture.
"""

import os
import torch
from torch.utils.cpp_extension import load

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_fused_add = None

def _get_extension():
    """Lazy load the CUDA extension."""
    global _fused_add
    if _fused_add is None:
        print("[INFO] JIT compiling fused_add CUDA kernel...")
        _fused_add = load(
            name="fused_add",
            sources=[
                os.path.join(_MODULE_DIR, "fused_add_cuda.cpp"),
                os.path.join(_MODULE_DIR, "fused_add.cu"),
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        print("[INFO] fused_add compilation completed.")
    return _fused_add


def fused_add(a: torch.Tensor, b: torch.Tensor, use_vectorized: bool = True) -> torch.Tensor:
    """
    Fused element-wise addition.
    
    This replaces:
        output = a + b
    
    With a single CUDA kernel call.
    
    Args:
        a: First tensor
        b: Second tensor (must have same shape as a)
        use_vectorized: Use vectorized memory access (float4)
    
    Returns:
        output: a + b
    """
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Both tensors must be on CUDA")
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.dtype != b.dtype:
        raise ValueError(f"Dtype mismatch: {a.dtype} vs {b.dtype}")
    
    ext = _get_extension()
    return ext.fused_add(a, b, use_vectorized)
