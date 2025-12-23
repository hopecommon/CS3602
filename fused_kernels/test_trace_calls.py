#!/usr/bin/env python3
"""
Final diagnosis: Are we calling fused_add at all?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Monkey-patch fused_add to log calls
original_fused_add = None
call_count = 0

def logging_fused_add(a, b, **kwargs):
    global call_count
    call_count += 1
    result = original_fused_add(a, b, **kwargs)
    if call_count <= 5:  # Only log first few
        print(f"  [TRACE] fused_add called #{call_count}: shape={a.shape}, range=[{result.min().item():.3f}, {result.max().item():.3f}]")
    return result

def main():
    global original_fused_add, call_count
    
    print("=" * 80)
    print("Final Diagnosis: Trace fused_add calls")
    print("=" * 80)
    
    # Patch fused_add
    from fused_kernels import fused_add as fused_add_module
    original_fused_add = fused_add_module.fused_add
    fused_add_module.fused_add = logging_fused_add
    
    # Also patch in gptneox_fused_add
    import fused_kernels.gptneox_fused_add as gptneox_module
    gptneox_module.fused_add = logging_fused_add
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-2.8b",
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    
    # Disable dropout
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
    inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
    
    # Apply fused
    from fused_kernels.gptneox_fused_add import apply_fused_add
    apply_fused_add(model, enabled=True)
    
    print("\nRunning model with fused_add...")
    print("(Should see ~64 calls: 2 per layer × 32 layers)")
    call_count = 0
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)
    
    print(f"\nTotal fused_add calls: {call_count}")
    
    if call_count == 0:
        print("✗ BUG: fused_add was NEVER called!")
        print("   The monkey-patching failed or forward wasn't replaced")
    elif call_count < 60:
        print(f"⚠ WARNING: Only {call_count} calls (expected ~64)")
    else:
        print(f"✓ fused_add is being called ({call_count} times)")
    
    # Restore
    fused_add_module.fused_add = original_fused_add
    gptneox_module.fused_add = original_fused_add


if __name__ == "__main__":
    main()
