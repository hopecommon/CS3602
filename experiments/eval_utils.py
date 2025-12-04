"""
评估工具函数

提供通用的 PPL 评估、数据加载等功能
"""

import time
import random
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from torch import Tensor
from datasets import load_dataset
from transformers import AutoTokenizer

# 设置随机种子以确保可复现
random.seed(42)

# PG19 本地缓存目录
PG19_CACHE_DIR = Path("data/pg19")
PG19_CACHE_FILE = PG19_CACHE_DIR / "sample.json"


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
    # 特殊处理 PG19: 下载一条数据到本地，后续使用本地缓存
    if dataset_name.lower() == "pg19":
        print("检测到 PG19 数据集...")
        
        # 检查本地缓存
        if PG19_CACHE_FILE.exists():
            print(f"✓ 使用本地缓存: {PG19_CACHE_FILE}")
            with open(PG19_CACHE_FILE, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            texts = [cached_data.get(text_column, "")]
            print(f"  已加载缓存数据 (长度: {len(texts[0])} 字符)")
        else:
            print("本地缓存不存在，开始流式下载 PG19 数据集...")
            print("注意: 只下载一条样本以节省空间和时间")
            
            # 流式加载前 N 条作为候选
            dataset = load_dataset(
                dataset_name,
                split=split,
                streaming=True,
                trust_remote_code=trust_remote_code
            )
            
            # 收集前 10 条作为候选
            N = 10
            buffer = []
            print(f"正在流式加载前 {N} 条样本...")
            for i, example in enumerate(dataset):
                buffer.append(example)
                print(f"  已加载 {i+1}/{N} 条", end='\r')
                if i + 1 >= N:
                    break
            print()  # 换行
            
            # 随机选择一条 (固定种子确保可复现)
            random_one = random.choice(buffer)
            texts = [random_one.get(text_column, "")]
            print(f"✓ 已从前 {N} 条中随机选择 1 条样本 (种子=42)")
            print(f"  样本长度: {len(texts[0])} 字符")
            
            # 保存到本地缓存
            PG19_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(PG19_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(random_one, f, ensure_ascii=False, indent=2)
            print(f"✓ 已保存到本地缓存: {PG19_CACHE_FILE}")
            print(f"  后续运行将直接使用本地缓存，无需重新下载")
    
    else:
        # 其他数据集的正常加载逻辑
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
    max_cache_size: Optional[int] = None,
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
    if max_cache_size is not None:
        wrapper = streaming_wrapper if use_streaming else None
        return _compute_streaming_decode_perplexity(
            model=model,
            encoded_dataset=encoded_dataset,
            device=device,
            max_cache_size=max_cache_size,
            streaming_wrapper=wrapper,
        )

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


def _compute_streaming_decode_perplexity(
    model,
    encoded_dataset: Tensor,
    device: torch.device,
    max_cache_size: int,
    streaming_wrapper = None,
) -> Tuple[float, float, float]:
    """
    使用解码式评估 (逐 token) 计算 PPL 和时间

    Args:
        model: 语言模型
        encoded_dataset: tokenized 数据集
        device: 设备
        max_cache_size: 最多保留的 token 数 (n_sink + window_size)
        streaming_wrapper: 若提供则使用 StreamingLLM, 否则模拟 sliding window baseline
    """
    seq_len = encoded_dataset.size(1)
    if seq_len < 2:
        raise ValueError("Dataset is too short to compute perplexity (seq_len < 2)")

    max_cache_size = max(2, min(max_cache_size, seq_len))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    total_start = time.perf_counter()
    prefill_start = time.perf_counter()

    total_nll = 0.0
    total_tokens = 0

    if streaming_wrapper is None:
        prefill_len = min(max_cache_size, seq_len)
        input_ids = encoded_dataset[:, :prefill_len].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)

        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        if labels.numel() > 0:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="sum",
            )
            total_nll += loss.item()
            total_tokens += labels.numel()

        prefill_time = time.perf_counter() - prefill_start

        for pos in range(prefill_len - 1, seq_len - 1):
            start_idx = max(0, pos + 1 - max_cache_size)
            context = encoded_dataset[:, start_idx:pos + 1].to(device)
            target = encoded_dataset[:, pos + 1].to(device)

            with torch.no_grad():
                outputs = model(input_ids=context, use_cache=False)

            logits = outputs.logits[:, -1, :]
            loss = F.cross_entropy(
                logits,
                target,
                reduction="sum",
            )
            total_nll += loss.item()
            total_tokens += target.numel()
    else:
        past_key_values = None
        prefill_len = min(max_cache_size, seq_len)
        with streaming_wrapper.enable():
            input_ids = encoded_dataset[:, :prefill_len].to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, use_cache=True)

            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            if labels.numel() > 0:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    reduction="sum",
                )
                total_nll += loss.item()
                total_tokens += labels.numel()

            past_key_values = outputs.past_key_values
            streaming_wrapper.update(past_key_values)

            prefill_time = time.perf_counter() - prefill_start

            for pos in range(prefill_len - 1, seq_len - 1):
                current_input = encoded_dataset[:, pos:pos + 1].to(device)
                target = encoded_dataset[:, pos + 1].to(device)

                with torch.no_grad():
                    outputs = model(
                        input_ids=current_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                logits = outputs.logits[:, -1, :]
                loss = F.cross_entropy(
                    logits,
                    target,
                    reduction="sum",
                )
                total_nll += loss.item()
                total_tokens += target.numel()

                past_key_values = outputs.past_key_values
                streaming_wrapper.update(past_key_values)

    total_time = time.perf_counter() - total_start
    ppl = torch.exp(torch.tensor(total_nll / total_tokens))
    return ppl.item(), total_time, prefill_time


_compute_streaming_perplexity = _compute_streaming_decode_perplexity


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
