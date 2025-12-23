#!/usr/bin/env python3
"""
Debug first layer divergence.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fused_kernels.fused_add import fused_add
from fused_kernels.gptneox_fused_add import apply_fused_add


def main():
    print("=" * 80)
    print("DEBUG: First Layer Divergence")
    print("=" * 80)
    
    model_name = "EleutherAI/pythia-2.8b"
    print(f"\nLoading {model_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    
    # Disable dropout
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_text = "The future"
    inputs = tokenizer(test_text, return_tensors="pt").to("cuda")
    
    # Test simple case: just call first layer
    print("\nTesting first layer only...")
    
    # Get input embeddings
    with torch.no_grad():
        input_embeds = model.get_input_embeddings()(inputs['input_ids'])
    
    print(f"Input embeds shape: {input_embeds.shape}")
    print(f"Input embeds range: [{input_embeds.min().item():.6f}, {input_embeds.max().item():.6f}]")
    
    first_layer = model.gpt_neox.layers[0]
    
    # Test 1: Original forward
    print("\n1. Original forward:")
    apply_fused_add(model, enabled=False)
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    with torch.no_grad():
        outputs_orig = first_layer(input_embeds, use_cache=False)
    hidden_orig = outputs_orig[0]
    
    print(f"   Output shape: {hidden_orig.shape}")
    print(f"   Output range: [{hidden_orig.min().item():.6f}, {hidden_orig.max().item():.6f}]")
    print(f"   Output mean: {hidden_orig.mean().item():.6f}")
    print(f"   Output std: {hidden_orig.std().item():.6f}")
    
    # Test 2: Fused forward
    print("\n2. Fused forward:")
    apply_fused_add(model, enabled=True)
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    with torch.no_grad():
        outputs_fused = first_layer(input_embeds, use_cache=False)
    hidden_fused = outputs_fused[0]
    
    print(f"   Output shape: {hidden_fused.shape}")
    print(f"   Output range: [{hidden_fused.min().item():.6f}, {hidden_fused.max().item():.6f}]")
    print(f"   Output mean: {hidden_fused.mean().item():.6f}")
    print(f"   Output std: {hidden_fused.std().item():.6f}")
    
    # Restore
    apply_fused_add(model, enabled=False)
    
    # Compare
    print("\n3. Comparison:")
    abs_diff = (hidden_fused - hidden_orig).abs()
    print(f"   Max abs diff: {abs_diff.max().item():.6f}")
    print(f"   Mean abs diff: {abs_diff.mean().item():.6f}")
    print(f"   Positions with diff > 0.01: {(abs_diff > 0.01).sum().item()} / {abs_diff.numel()}")
    
    # Show first few divergent positions
    print("\n4. First divergent positions:")
    flat_orig = hidden_orig.flatten()
    flat_fused = hidden_fused.flatten()
    flat_diff = abs_diff.flatten()
    
    top_indices = torch.argsort(flat_diff, descending=True)[:10]
    for i, idx in enumerate(top_indices):
        idx = idx.item()
        print(f"   Position {idx}: orig={flat_orig[idx].item():.6f}, "
              f"fused={flat_fused[idx].item():.6f}, "
              f"diff={flat_diff[idx].item():.6f}")
    
    # Test 3: Manually trace through layer
    print("\n5. Manual trace:")
    
    # Attention path
    print("   a) After LayerNorm:")
    normed = first_layer.input_layernorm(input_embeds)
    print(f"      Range: [{normed.min().item():.6f}, {normed.max().item():.6f}]")
    
    print("   b) After Attention:")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    attn_out = first_layer.attention(normed, use_cache=False)[0]
    print(f"      Range: [{attn_out.min().item():.6f}, {attn_out.max().item():.6f}]")
    
    print("   c) After Dropout:")
    attn_out = first_layer.post_attention_dropout(attn_out)
    print(f"      Range: [{attn_out.min().item():.6f}, {attn_out.max().item():.6f}]")
    
    print("   d) After Residual Add (PyTorch):")
    residual_1 = attn_out + input_embeds
    print(f"      Range: [{residual_1.min().item():.6f}, {residual_1.max().item():.6f}]")
    
    print("   e) After Residual Add (Fused):")
    residual_1_fused = fused_add(attn_out, input_embeds)
    print(f"      Range: [{residual_1_fused.min().item():.6f}, {residual_1_fused.max().item():.6f}]")
    
    print("   f) Difference in first residual:")
    first_residual_diff = (residual_1_fused - residual_1).abs()
    print(f"      Max diff: {first_residual_diff.max().item():.6f}")
    print(f"      Mean diff: {first_residual_diff.mean().item():.6f}")
    
    if first_residual_diff.max().item() > 0.0001:
        print("\n✗ FOUND THE BUG: fused_add produces different results!")
        print("   Even though isolated tests pass, something is wrong in real usage.")
    else:
        print("\n✓ First residual add is correct, bug must be elsewhere")


if __name__ == "__main__":
    main()
