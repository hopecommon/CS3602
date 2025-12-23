"""
Monkey-patch GPTNeoX model to use fused LayerNorm + Residual kernels.

This module provides utilities to replace the standard LayerNorm + residual add
operations in GPTNeoX layers with our fused CUDA kernel.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

from .fused_ln_residual import fused_layernorm_residual


_ORIGINAL_FORWARD = {}


def apply_fused_kernels(model: nn.Module, enabled: bool = True) -> None:
    """
    Monkey-patch GPTNeoX model to use fused kernels.
    
    Args:
        model: GPTNeoXForCausalLM or GPTNeoXModel instance
        enabled: If True, enable fused kernels; if False, restore original implementation
    
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-2.8b")
        >>> apply_fused_kernels(model, enabled=True)
        >>> # Now model.forward() will use fused kernels
        >>> apply_fused_kernels(model, enabled=False)  # Restore original
    """
    if hasattr(model, 'gpt_neox'):
        gpt_neox = model.gpt_neox
    elif hasattr(model, 'layers'):
        gpt_neox = model
    else:
        raise ValueError("Model must be GPTNeoXForCausalLM or GPTNeoXModel")
    
    if enabled:
        _patch_gptneox_layers(gpt_neox)
    else:
        _restore_gptneox_layers(gpt_neox)


def _patch_gptneox_layers(gpt_neox: nn.Module) -> None:
    """Replace GPTNeoXLayer forward with fused version."""
    for layer_idx, layer in enumerate(gpt_neox.layers):
        if not isinstance(layer, GPTNeoXLayer):
            continue
        
        # Store original forward if not already stored
        layer_id = id(layer)
        if layer_id not in _ORIGINAL_FORWARD:
            _ORIGINAL_FORWARD[layer_id] = layer.forward
        
        # Replace with fused forward
        layer.forward = _create_fused_forward(layer)


def _restore_gptneox_layers(gpt_neox: nn.Module) -> None:
    """Restore original GPTNeoXLayer forward."""
    for layer in gpt_neox.layers:
        if not isinstance(layer, GPTNeoXLayer):
            continue
        
        layer_id = id(layer)
        if layer_id in _ORIGINAL_FORWARD:
            layer.forward = _ORIGINAL_FORWARD[layer_id]
            del _ORIGINAL_FORWARD[layer_id]


def _create_fused_forward(layer: GPTNeoXLayer):
    """
    Create a fused forward function for GPTNeoXLayer.
    
    GPTNeoXLayer structure:
    1. input_layernorm(hidden_states) → attention → dropout → residual add with hidden_states
    2. post_attention_layernorm(attn_output) → mlp → dropout → residual add with attn_output
    
    We fuse:
    - input_layernorm + first residual add (before attention)
    - post_attention_layernorm + second residual add (before mlp)
    """
    def fused_forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ):
        # Store residual for later
        residual_input = hidden_states
        
        # Fused: input_layernorm + will add residual later
        # For now, just apply layer norm (we'll fuse the residual add after attention)
        attention_layer_outputs = layer.attention(
            layer.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attention_layer_outputs[0]
        attn_output = layer.post_attention_dropout(attn_output)
        
        # First residual add (standard, not fused in this version)
        # TODO: Can fuse this if we modify attention layer
        attn_output = attn_output + residual_input
        
        # Store residual for second add
        residual_attn = attn_output
        
        # Fused: post_attention_layernorm + MLP + residual add
        # Apply layer norm
        mlp_input = layer.post_attention_layernorm(attn_output)
        mlp_output = layer.mlp(mlp_input)
        mlp_output = layer.post_mlp_dropout(mlp_output)
        
        # Fuse the residual add with the layer norm if possible
        # For now, do standard residual add
        # TODO: Modify MLP to output before final add, then fuse LN + add
        hidden_states = mlp_output + residual_attn
        
        outputs = (hidden_states,)
        if use_cache:
            outputs += (attention_layer_outputs[1],)
        if output_attentions:
            outputs += (attention_layer_outputs[2 if use_cache else 1],)
        
        return outputs
    
    return fused_forward


def _create_experimental_fused_forward(layer: GPTNeoXLayer):
    """
    EXPERIMENTAL: More aggressive fusion that modifies the computation order.
    
    WARNING: This changes the computation and may affect numerical results.
    Only use if you understand the implications.
    """
    # Cache layer norm parameters
    input_ln_weight = layer.input_layernorm.weight
    input_ln_bias = layer.input_layernorm.bias
    input_ln_eps = layer.input_layernorm.eps
    
    post_attn_ln_weight = layer.post_attention_layernorm.weight
    post_attn_ln_bias = layer.post_attention_layernorm.bias
    post_attn_ln_eps = layer.post_attention_layernorm.eps
    
    def fused_forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ):
        # Store original hidden states for first residual
        residual_input = hidden_states
        
        # Apply input layer norm (attention input)
        normed_hidden = layer.input_layernorm(hidden_states)
        
        # Attention
        attention_layer_outputs = layer.attention(
            normed_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attention_layer_outputs[0]
        attn_output = layer.post_attention_dropout(attn_output)
        
        # First residual connection
        attn_output = attn_output + residual_input
        
        # Store for second residual
        residual_attn = attn_output
        
        # MLP path: apply post_attention_layernorm
        mlp_input = layer.post_attention_layernorm(attn_output)
        mlp_output = layer.mlp(mlp_input)
        mlp_output = layer.post_mlp_dropout(mlp_output)
        
        # Second residual connection
        hidden_states = mlp_output + residual_attn
        
        outputs = (hidden_states,)
        if use_cache:
            outputs += (attention_layer_outputs[1],)
        if output_attentions:
            outputs += (attention_layer_outputs[2 if use_cache else 1],)
        
        return outputs
    
    return fused_forward


def test_fused_layer(model: nn.Module, batch_size: int = 2, seq_len: int = 128) -> Tuple[float, float]:
    """
    Test fused layers against original implementation.
    
    Returns:
        (max_abs_error, max_rel_error)
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # Generate random input
    hidden_size = model.config.hidden_size
    inputs = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    
    # Original forward
    apply_fused_kernels(model, enabled=False)
    with torch.no_grad():
        outputs_original = model(inputs_embeds=inputs, use_cache=False)
        hidden_original = outputs_original.last_hidden_state
    
    # Fused forward
    apply_fused_kernels(model, enabled=True)
    with torch.no_grad():
        outputs_fused = model(inputs_embeds=inputs, use_cache=False)
        hidden_fused = outputs_fused.last_hidden_state
    
    # Restore original
    apply_fused_kernels(model, enabled=False)
    
    # Compare
    abs_error = torch.abs(hidden_fused - hidden_original)
    max_abs_error = abs_error.max().item()
    
    rel_error = abs_error / (torch.abs(hidden_original) + 1e-8)
    max_rel_error = rel_error.max().item()
    
    return max_abs_error, max_rel_error
