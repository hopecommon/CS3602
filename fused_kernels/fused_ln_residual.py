"""
Fused LayerNorm + Residual implementation with JIT compilation.

This module provides a fused CUDA kernel that combines LayerNorm and residual addition
into a single operation, reducing memory bandwidth and kernel launch overhead.
"""

import os
import torch
from torch.utils.cpp_extension import load

# Get the directory containing this file
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# JIT compile the CUDA extension
_fused_ln_residual = None

def _get_extension():
    """Lazy load the CUDA extension with JIT compilation."""
    global _fused_ln_residual
    if _fused_ln_residual is None:
        print("[INFO] JIT compiling fused_ln_residual CUDA kernel...")
        _fused_ln_residual = load(
            name="fused_ln_residual",
            sources=[
                os.path.join(_MODULE_DIR, "fused_ln_residual_cuda.cpp"),
                os.path.join(_MODULE_DIR, "fused_ln_residual.cu"),
            ],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
            ],
            verbose=True,
        )
        print("[INFO] JIT compilation completed.")
    return _fused_ln_residual


def fused_layernorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Fused LayerNorm + Residual Add operation.
    
    Computes: LayerNorm(x) + residual
    
    This is equivalent to:
        normalized = F.layer_norm(x, (x.size(-1),), weight, bias, eps)
        output = normalized + residual
    
    But implemented as a single CUDA kernel for better performance.
    
    Args:
        x: Input tensor of shape [batch, seq_len, hidden_size]
        residual: Residual tensor of shape [batch, seq_len, hidden_size]
        weight: LayerNorm weight (gamma) of shape [hidden_size]
        bias: LayerNorm bias (beta) of shape [hidden_size]
        eps: Small constant for numerical stability (default: 1e-5)
    
    Returns:
        Output tensor of shape [batch, seq_len, hidden_size]
    
    Examples:
        >>> x = torch.randn(2, 128, 2560, device='cuda', dtype=torch.float16)
        >>> residual = torch.randn_like(x)
        >>> weight = torch.ones(2560, device='cuda', dtype=torch.float16)
        >>> bias = torch.zeros(2560, device='cuda', dtype=torch.float16)
        >>> output = fused_layernorm_residual(x, residual, weight, bias)
        >>> output.shape
        torch.Size([2, 128, 2560])
    """
    # Validate inputs
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not residual.is_cuda:
        raise ValueError("residual must be a CUDA tensor")
    if not weight.is_cuda:
        raise ValueError("weight must be a CUDA tensor")
    if not bias.is_cuda:
        raise ValueError("bias must be a CUDA tensor")
    
    if x.dim() != 3:
        raise ValueError(f"x must be 3D tensor [batch, seq_len, hidden_size], got shape {x.shape}")
    
    if x.shape != residual.shape:
        raise ValueError(f"x and residual must have same shape, got {x.shape} and {residual.shape}")
    
    hidden_size = x.size(-1)
    if weight.shape != (hidden_size,):
        raise ValueError(f"weight must have shape [{hidden_size}], got {weight.shape}")
    if bias.shape != (hidden_size,):
        raise ValueError(f"bias must have shape [{hidden_size}], got {bias.shape}")
    
    # Ensure all tensors have the same dtype
    if not (x.dtype == residual.dtype == weight.dtype == bias.dtype):
        raise ValueError(f"All tensors must have the same dtype, got x: {x.dtype}, "
                        f"residual: {residual.dtype}, weight: {weight.dtype}, bias: {bias.dtype}")
    
    # Call CUDA kernel
    ext = _get_extension()
    return ext.fused_layernorm_residual(x, residual, weight, bias, eps)


def fused_layernorm_residual_forward(
    x: torch.Tensor,
    residual: torch.Tensor,
    normalized_shape: tuple,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Drop-in replacement for torch.nn.functional.layer_norm + residual add.
    
    This function signature matches F.layer_norm more closely for easier integration.
    
    Args:
        x: Input tensor
        residual: Residual tensor
        normalized_shape: Shape to normalize over (typically (hidden_size,))
        weight: LayerNorm weight
        bias: LayerNorm bias
        eps: Small constant for numerical stability
    
    Returns:
        Output tensor: LayerNorm(x) + residual
    """
    if len(normalized_shape) != 1:
        raise ValueError(f"Only support normalizing over last dimension, got {normalized_shape}")
    
    # Reshape if needed to ensure 3D input
    original_shape = x.shape
    if x.dim() == 2:
        x = x.unsqueeze(0)
        residual = residual.unsqueeze(0)
    elif x.dim() > 3:
        batch_dims = x.shape[:-1]
        x = x.reshape(-1, 1, x.size(-1))
        residual = residual.reshape(-1, 1, residual.size(-1))
    
    output = fused_layernorm_residual(x, residual, weight, bias, eps)
    
    # Reshape back to original shape
    if output.shape != original_shape:
        output = output.reshape(original_shape)
    
    return output
