"""
评估工具函数

提供通用的 PPL 评估、数据加载等功能
"""

import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
from torch import Tensor
from datasets import load_dataset
from transformers import AutoTokenizer


def load_tokenized_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    text_column: str,
    max_samples: Optional[int],
    tokenizer,
    max_eval_tokens: Optional[int] = None,
    trust_remote_code: bool = False,
    use_streaming: bool = False,
) -> Tensor:
    """
    加载并 tokenize 数据集
    
    Args:
        dataset_name: 数据集名称 (如 'wikitext', 'pg19')
        dataset_config: 数据集配置 (如 'wikitext-103-v1')
        split: 数据集分割 (如 'test')
        text_column: 文本列名 (如 'text')
        max_samples: 最大样本数
        tokenizer: tokenizer 实例
        max_eval_tokens: 最大评估 token 数
        trust_remote_code: 是否信任远程代码
        use_streaming: 是否使用流式加载
    
    Returns:
        input_ids: tokenized 输入 [1, seq_len]
    """
    # 加载数据集
    dataset_kwargs = {
        "split": split,
        "trust_remote_code": trust_remote_code
    }
    if use_streaming:
        dataset_kwargs["streaming"] = True
    
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, **dataset_kwargs)
    else:
        dataset = load_dataset(dataset_name, **dataset_kwargs)
    
    # 限制样本数
    if not use_streaming and max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # 收集文本
    texts = []
    for idx, row in enumerate(dataset):
        if use_streaming and max_samples and idx >= max_samples:
            break
        text = row.get(text_column, "")
        if text and not text.isspace():
            texts.append(text)
    
    if not texts:
        raise ValueError(f"No non-empty rows found for dataset {dataset_name}")
    
    # 拼接并 tokenize
    concatenated = "\n\n".join(texts)
    encodings = tokenizer(concatenated, return_tensors="pt")
    input_ids = encodings.input_ids
    
    # 限制 token 数
    if max_eval_tokens is not None and max_eval_tokens > 0:
        input_ids = input_ids[:, :max_eval_tokens]
    
    return input_ids


def compute_perplexity(
    model,
    encoded_dataset: Tensor,
    device: torch.device,
    max_length: int,
    stride: int,
    use_streaming: bool = False,
    streaming_wrapper = None,
) -> Tuple[float, float, float]:
    """
    计算 perplexity
    
    Args:
        model: 语言模型
        encoded_dataset: tokenized 数据集 [1, seq_len]
        device: 设备
        max_length: 单次评估的最大长度
        stride: 滑动步长
        use_streaming: 是否使用 StreamingLLM
        streaming_wrapper: StreamingLLMWrapper 实例
    
    Returns:
        perplexity: PPL 值
        total_time: 总时间 (秒)
        prefill_time: prefill 时间 (秒)
    """
    nlls = []
    total_tokens = 0
    seq_len = encoded_dataset.size(1)
    
    # 重置显存统计
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    total_start = time.perf_counter()
    prefill_start = time.perf_counter()
    
    # 使用 StreamingLLM 或普通模式
    if use_streaming and streaming_wrapper is not None:
        context = streaming_wrapper.enable()
    else:
        from contextlib import nullcontext
        context = nullcontext()
    
    with context:
        for start_idx in range(0, seq_len, stride):
            begin_loc = max(start_idx + stride - max_length, 0)
            end_loc = min(start_idx + stride, seq_len)
            trg_len = end_loc - start_idx
            
            input_ids = encoded_dataset[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=target_ids, use_cache=True)
            
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood.detach().to("cpu"))
            total_tokens += trg_len
            
            if end_loc == seq_len:
                break
    
    prefill_time = time.perf_counter() - prefill_start
    total_time = time.perf_counter() - total_start
    
    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
    
    return ppl.item(), total_time, prefill_time


def save_results(
    results: Dict[str, Any],
    output_path: Path
):
    """
    保存结果到 JSON 文件
    
    Args:
        results: 结果字典
        output_path: 输出路径
    """
    import json
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n结果已保存到: {output_path}")


def print_results(results: Dict[str, Any]):
    """
    打印结果
    
    Args:
        results: 结果字典
    """
    import json
    print("\n" + "="*60)
    print("实验结果")
    print("="*60)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("="*60)