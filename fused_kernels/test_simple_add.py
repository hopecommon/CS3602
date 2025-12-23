#!/usr/bin/env python3
"""
Simple test: Does fused_add give same result as PyTorch + in actual model tensors?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM

from fused_kernels.fused_add import fused_add


def main():
    print("=" * 80)
    print("Simple Test: fused_add vs PyTorch + on real model tensors")
    print("=" * 80)
    
    # Load model just to get real tensors
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-2.8b",
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    
    # Get two real tensors from the model
    layer = model.gpt_neox.layers[0]
    
    # Create realistic tensors
    batch_size, seq_len = 1, 8
    hidden_size = 2560
    
    torch.manual_seed(42)
    a = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16) * 10
    b = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16) * 10
    
    print(f"\nTensor shapes: {a.shape}")
    print(f"Tensor dtype: {a.dtype}")
    print(f"Tensor device: {a.device}")
    print(f"a is_contiguous: {a.is_contiguous()}")
    print(f"b is_contiguous: {b.is_contiguous()}")
    print(f"a range: [{a.min().item():.3f}, {a.max().item():.3f}]")
    print(f"b range: [{b.min().item():.3f}, {b.max().item():.3f}]")
    
    # PyTorch
    ref = a + b
    
    # Fused
    fused = fused_add(a, b)
    
    # Compare
    print(f"\n Results:")
    print(f"ref range: [{ref.min().item():.3f}, {ref.max().item():.3f}]")
    print(f"fused range: [{fused.min().item():.3f}, {fused.max().item():.3f}]")
    
    abs_diff = (fused - ref).abs()
    print(f"\nMax abs diff: {abs_diff.max().item():.6e}")
    print(f"Mean abs diff: {abs_diff.mean().item():.6e}")
    print(f"Num diff > 0: {(abs_diff > 0).sum().item()} / {abs_diff.numel()}")
    
    if abs_diff.max().item() > 1e-5:
        print("\n✗ DIFFERENT!")
        # Show where
        indices = torch.where(abs_diff > 1e-5)
        print(f"Divergent positions (first 5):")
        for i in range(min(5, len(indices[0]))):
            idx0, idx1, idx2 = indices[0][i].item(), indices[1][i].item(), indices[2][i].item()
            print(f"  [{idx0},{idx1},{idx2}]: ref={ref[idx0,idx1,idx2].item():.6f}, "
                  f"fused={fused[idx0,idx1,idx2].item():.6f}")
    else:
        print("\n✓ IDENTICAL (within FP16 precision)")
    
    # Test 2: In-place check
    print("\n" + "=" * 80)
    print("Test 2: Check if kernel modifies inputs")
    print("=" * 80)
    
    a_orig = a.clone()
    b_orig = b.clone()
    
    result = fused_add(a, b)
    
    a_changed = not torch.equal(a, a_orig)
    b_changed = not torch.equal(b, b_orig)
    
    print(f"a modified: {a_changed}")
    print(f"b modified: {b_changed}")
    
    if a_changed or b_changed:
        print("✗ Kernel modifies inputs! This is a BUG")
    else:
        print("✓ Inputs unchanged")


if __name__ == "__main__":
    main()
