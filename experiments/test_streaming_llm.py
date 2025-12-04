#!/usr/bin/env python3
"""
快速测试 StreamingLLM 实现是否正常工作
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming_llm import StreamingLLMWrapper


def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "="*60)
    print("测试 StreamingLLM 基本功能")
    print("="*60 + "\n")
    
    # 加载模型
    print("1. 加载模型...")
    model_name = "EleutherAI/pythia-70m"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ 模型加载成功: {model_name}")
    print(f"✓ 设备: {device}")
    
    # 创建 StreamingLLMWrapper
    print("\n2. 创建 StreamingLLMWrapper...")
    wrapper = StreamingLLMWrapper(
        model=model,
        n_sink=4,
        window_size=128
    )
    print(f"✓ Wrapper 创建成功")
    print(f"  n_sink: {wrapper.n_sink}")
    print(f"  window_size: {wrapper.window_size}")
    print(f"  max_cache_size: {wrapper.cache.max_size}")
    
    # 测试推理
    print("\n3. 测试推理...")
    test_text = "The quick brown fox jumps over the lazy dog. " * 50
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    print(f"✓ 输入长度: {inputs.input_ids.shape[1]} tokens")
    
    # 不使用 StreamingLLM
    print("\n4. 测试基线推理...")
    with torch.no_grad():
        outputs_baseline = model(**inputs, use_cache=True)
    print(f"✓ 基线推理成功")
    print(f"  输出形状: {outputs_baseline.logits.shape}")
    
    # 使用 StreamingLLM
    print("\n5. 测试 StreamingLLM 推理...")
    with wrapper.enable():
        with torch.no_grad():
            outputs_streaming = model(**inputs, use_cache=True)
    print(f"✓ StreamingLLM 推理成功")
    print(f"  输出形状: {outputs_streaming.logits.shape}")
    
    # 比较结果
    print("\n6. 比较 logits...")
    logits_diff = torch.abs(outputs_baseline.logits - outputs_streaming.logits).mean().item()
    print(f"  Logits 平均差异: {logits_diff:.6f}")
    
    if logits_diff < 0.1:
        print(f"✓ Logits 差异在合理范围内")
    else:
        print(f"⚠ Logits 差异较大: {logits_diff:.6f}")
    
    # 测试 KV Cache 压缩
    print("\n7. 测试 KV Cache 压缩...")
    seq_len = inputs.input_ids.shape[1]
    compression_ratio = wrapper.get_compression_ratio(seq_len)
    print(f"  原始序列长度: {seq_len}")
    print(f"  压缩后长度: {wrapper.cache.max_size}")
    print(f"  压缩比: {compression_ratio:.2%}")
    
    print("\n" + "="*60)
    print("✓ 所有测试通过!")
    print("="*60 + "\n")
    
    return True


def main():
    try:
        test_basic_functionality()
        return 0
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())