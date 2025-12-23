"""
Fused CUDA kernels for StreamingLLM optimization.

This module provides hand-written CUDA kernels to reduce kernel launch overhead
and memory bandwidth usage by fusing common operations in GPTNeoX models.

Key optimizations:
- Fused LayerNorm + Residual Add: Combines normalization and residual connection
  into a single kernel to eliminate intermediate tensor storage.

Usage:
    >>> from fused_kernels import fused_layernorm_residual, apply_fused_kernels
    >>> # Direct kernel usage
    >>> output = fused_layernorm_residual(x, residual, weight, bias, eps=1e-5)
    >>> 
    >>> # Model integration (monkey-patching)
    >>> from transformers import AutoModelForCausalLM
    >>> model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-2.8b")
    >>> apply_fused_kernels(model, enabled=True)
"""

from .fused_ln_residual import fused_layernorm_residual, fused_layernorm_residual_forward
from .gptneox_integration import apply_fused_kernels, test_fused_layer

__all__ = [
    "fused_layernorm_residual",
    "fused_layernorm_residual_forward",
    "apply_fused_kernels",
    "test_fused_layer",
]
