#!/usr/bin/env python3
"""
Rigorous correctness test for fused_add.

This test:
1. Tests kernel in isolation with various tensor layouts
2. Tests layer-by-layer hidden state consistency in the model
3. Only then tests generation output

Focus on CORRECTNESS before PERFORMANCE.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

from fused_kernels.fused_add import fused_add
from fused_kernels.gptneox_fused_add import apply_fused_add


def test_kernel_correctness():
    """Test fused_add kernel with various tensor layouts."""
    print("=" * 80)
    print("TEST 1: Kernel Correctness (Various Layouts)")
    print("=" * 80)
    
    device = torch.device("cuda")
    dtype = torch.float16
    torch.manual_seed(42)
    
    test_cases = [
        ("Contiguous", lambda x: x),
        ("Permuted", lambda x: x.permute(0, 2, 1).contiguous().permute(0, 2, 1)),  # Permute and back
        ("View", lambda x: x.view(-1, x.shape[-1])),
        ("Sliced", lambda x: x[:, ::2, :].contiguous()),  # Sliced then made contiguous
    ]
    
    all_passed = True
    results = {}
    
    for name, transform in test_cases:
        # Create base tensors
        a_base = torch.randn(4, 128, 2560, device=device, dtype=dtype)
        b_base = torch.randn(4, 128, 2560, device=device, dtype=dtype)
        
        # Apply transformation
        try:
            a = transform(a_base)
            b = transform(b_base)
            
            # Reference
            ref = a + b
            
            # Fused (should handle via .contiguous() internally)
            fused = fused_add(a, b)
            
            # Compare
            abs_err = (fused - ref).abs().max().item()
            rel_err = ((fused - ref).abs() / (ref.abs() + 1e-8)).max().item()
            
            passed = torch.allclose(fused, ref, rtol=1e-3, atol=1e-3)
            status = "✓ PASS" if passed else "✗ FAIL"
            
            print(f"\n{name}:")
            print(f"  Contiguous: a={a.is_contiguous()}, b={b.is_contiguous()}")
            print(f"  Status: {status}")
            print(f"  Max abs error: {abs_err:.6e}")
            print(f"  Max rel error: {rel_err:.6e}")
            
            results[name] = {
                "passed": passed,
                "max_abs_error": abs_err,
                "max_rel_error": rel_err,
                "a_contiguous": a.is_contiguous(),
                "b_contiguous": b.is_contiguous(),
            }
            
            if not passed:
                all_passed = False
        
        except Exception as e:
            print(f"\n{name}:")
            print(f"  ✗ ERROR: {e}")
            results[name] = {"error": str(e)}
            all_passed = False
    
    return all_passed, results


def test_hidden_states_consistency():
    """Test layer-by-layer hidden state consistency."""
    print("\n" + "=" * 80)
    print("TEST 2: Hidden States Consistency (Layer-by-Layer)")
    print("=" * 80)
    
    model_name = "EleutherAI/pythia-2.8b"
    print(f"\nLoading {model_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    
    # CRITICAL: Disable dropout for deterministic results
    for module in model.modules():
        if hasattr(module, 'dropout'):
            if isinstance(module.dropout, torch.nn.Dropout):
                module.dropout.p = 0.0
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Fixed input for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    test_text = "The future of artificial intelligence"
    inputs = tokenizer(test_text, return_tensors="pt").to("cuda")
    
    print("\nTesting with output_hidden_states=True (dropout=0)...")
    
    # Original
    print("1. Running original implementation...")
    apply_fused_add(model, enabled=False)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    with torch.no_grad():
        outputs_orig = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states_orig = outputs_orig.hidden_states
    
    # Fused
    print("2. Running fused implementation...")
    apply_fused_add(model, enabled=True)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    with torch.no_grad():
        outputs_fused = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states_fused = outputs_fused.hidden_states
    
    # Restore
    apply_fused_add(model, enabled=False)
    
    # Compare layer by layer
    print(f"\n3. Comparing {len(hidden_states_orig)} layers...")
    
    layer_results = {}
    max_abs_err_overall = 0.0
    max_rel_err_overall = 0.0
    first_divergence_layer = None
    
    for i, (h_orig, h_fused) in enumerate(zip(hidden_states_orig, hidden_states_fused)):
        abs_err = (h_fused - h_orig).abs().max().item()
        rel_err = ((h_fused - h_orig).abs() / (h_orig.abs() + 1e-8)).max().item()
        
        passed = torch.allclose(h_fused, h_orig, rtol=1e-2, atol=1e-2)
        
        layer_results[f"layer_{i}"] = {
            "max_abs_error": abs_err,
            "max_rel_error": rel_err,
            "passed": passed,
        }
        
        max_abs_err_overall = max(max_abs_err_overall, abs_err)
        max_rel_err_overall = max(max_rel_err_overall, rel_err)
        
        if not passed and first_divergence_layer is None:
            first_divergence_layer = i
        
        if i % 8 == 0 or not passed:  # Print every 8 layers or divergent ones
            status = "✓" if passed else "✗"
            print(f"   Layer {i:2d}: {status} abs_err={abs_err:.6e}, rel_err={rel_err:.6e}")
    
    print(f"\n4. Summary:")
    print(f"   Max abs error across all layers: {max_abs_err_overall:.6e}")
    print(f"   Max rel error across all layers: {max_rel_err_overall:.6e}")
    
    if first_divergence_layer is not None:
        print(f"   ⚠ First divergence at layer: {first_divergence_layer}")
        overall_passed = False
    else:
        print(f"   ✓ All layers match within tolerance")
        overall_passed = True
    
    return overall_passed, layer_results, model, tokenizer


def test_generation_output(model, tokenizer):
    """Test generation output consistency."""
    print("\n" + "=" * 80)
    print("TEST 3: Generation Output")
    print("=" * 80)
    
    test_text = "The quick brown fox"
    inputs = tokenizer(test_text, return_tensors="pt").to("cuda")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Original
    print("\n1. Original implementation:")
    apply_fused_add(model, enabled=False)
    with torch.no_grad():
        outputs_orig = model.generate(
            **inputs, 
            max_new_tokens=10, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
    text_orig = tokenizer.decode(outputs_orig[0], skip_special_tokens=True)
    print(f"   Output: '{text_orig}'")
    
    # Fused
    print("\n2. Fused implementation:")
    apply_fused_add(model, enabled=True)
    torch.manual_seed(42)  # Reset seed
    with torch.no_grad():
        outputs_fused = model.generate(
            **inputs, 
            max_new_tokens=10, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
    text_fused = tokenizer.decode(outputs_fused[0], skip_special_tokens=True)
    print(f"   Output: '{text_fused}'")
    
    # Restore
    apply_fused_add(model, enabled=False)
    
    # Compare
    tokens_match = torch.equal(outputs_orig, outputs_fused)
    text_match = text_orig == text_fused
    
    print(f"\n3. Comparison:")
    print(f"   Tokens match: {'✓ YES' if tokens_match else '✗ NO'}")
    print(f"   Text match:   {'✓ YES' if text_match else '✗ NO'}")
    
    if not tokens_match:
        print(f"\n   Original tokens: {outputs_orig[0].tolist()}")
        print(f"   Fused tokens:    {outputs_fused[0].tolist()}")
    
    return tokens_match, {
        "tokens_match": tokens_match,
        "text_match": text_match,
        "original_text": text_orig,
        "fused_text": text_fused,
    }


def main():
    print("=" * 80)
    print("RIGOROUS Fused Add Correctness Test")
    print("=" * 80)
    
    all_results = {}
    
    # Test 1: Kernel correctness
    kernel_passed, kernel_results = test_kernel_correctness()
    all_results["kernel_correctness"] = {
        "passed": kernel_passed,
        "details": kernel_results,
    }
    
    if not kernel_passed:
        print("\n" + "=" * 80)
        print("✗ Kernel test failed - stopping here")
        print("=" * 80)
        
        # Save results
        output_file = Path(__file__).parent / "rigorous_test_results.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
        return 1
    
    # Test 2: Hidden states consistency
    try:
        hidden_passed, layer_results, model, tokenizer = test_hidden_states_consistency()
        all_results["hidden_states_consistency"] = {
            "passed": hidden_passed,
            "details": layer_results,
        }
        
        # Test 3: Generation output
        gen_passed, gen_results = test_generation_output(model, tokenizer)
        all_results["generation_output"] = {
            "passed": gen_passed,
            "details": gen_results,
        }
        
    except Exception as e:
        print(f"\n✗ Error during model testing: {e}")
        import traceback
        traceback.print_exc()
        all_results["error"] = str(e)
        
        output_file = Path(__file__).parent / "rigorous_test_results.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nPartial results saved to: {output_file}")
        
        return 1
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print(f"✓ Kernel correctness:         {'PASS' if kernel_passed else 'FAIL'}")
    print(f"{'✓' if hidden_passed else '✗'} Hidden states consistency: {'PASS' if hidden_passed else 'FAIL'}")
    print(f"{'✓' if gen_passed else '✗'} Generation output:        {'PASS' if gen_passed else 'FAIL'}")
    
    all_passed = kernel_passed and hidden_passed and gen_passed
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED - fused_add is numerically correct")
        print("   Now it's safe to benchmark performance.")
    else:
        print("\n✗ SOME TESTS FAILED - DO NOT USE fused_add yet")
        print("   Fix correctness issues before benchmarking.")
    
    # Save results
    output_file = Path(__file__).parent / "rigorous_test_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to: {output_file}")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
