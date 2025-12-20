#!/usr/bin/env python3
"""
Per-Token Decoding Latency 评估脚本

按照 StreamingLLM 论文的标准方法测量每个 token 的解码延迟:
1. GPU warmup (100-200 tokens)
2. 只统计 cache 填满后的 tokens
3. 使用 torch.cuda.synchronize() 确保准确计时
4. 多次运行取平均和标准差
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from contextlib import nullcontext

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming_llm import StreamingLLMWrapper
from streaming_llm.cache_ops import commit_and_prune
from streaming_llm.speculative import exact_accept, greedy_match, propose_tokens


def parse_args():
    parser = argparse.ArgumentParser(
        description="测量 per-token decoding latency"
    )
    
    # 模型参数
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.environ.get("MODEL_NAME", "EleutherAI/pythia-2.8b"),
        help="模型名称"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="数据类型"
    )
    
    # 评估参数
    parser.add_argument(
        "--cache-size",
        type=int,
        default=1024,
        help="Cache size (window_size for StreamingLLM)"
    )
    parser.add_argument(
        "--n-sink",
        type=int,
        default=4,
        help="Sink token 数量 (仅 StreamingLLM)"
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=512,
        help="初始 prompt 长度"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=2000,
        help="生成的 token 数量 (用于统计)"
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=200,
        help="Warmup token 数量"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="重复运行次数"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["both", "baseline", "streaming"],
        default="both",
        help="评估模式: both=基线+Streaming, baseline=仅基线, streaming=仅 Streaming"
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        help="已存在的基线结果 JSON (mode=streaming 时可复用)"
    )

    # Speculative decoding
    parser.add_argument(
        "--decoder",
        type=str,
        choices=["normal", "speculative"],
        default="normal",
        help="解码策略: normal=逐 token, speculative=投机解码"
    )
    parser.add_argument(
        "--speculative-mode",
        type=str,
        choices=["exact", "greedy_match"],
        default="exact",
        help="SpecDec 模式: exact=Leviathan exact, greedy_match=确定性对齐"
    )
    parser.add_argument(
        "--draft-model-name",
        type=str,
        default=os.environ.get("DRAFT_MODEL_NAME", "EleutherAI/pythia-70m"),
        help="SpecDec draft 模型名称"
    )
    parser.add_argument(
        "--draft-k",
        type=int,
        default=8,
        help="SpecDec 每轮提案 token 数"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度 (0 表示 greedy)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (SpecDec 采样用)"
    )

    # Prefill from dataset
    parser.add_argument(
        "--prefill-file",
        type=Path,
        default=None,
        help="Optional JSON file with a long text field to prefill context"
    )
    parser.add_argument(
        "--prefill-field",
        type=str,
        default="text",
        help="JSON field name for prefill text"
    )
    parser.add_argument(
        "--prefill-tokens",
        type=int,
        default=0,
        help="Number of tokens to prefill (0 = all tokens from file)"
    )
    parser.add_argument(
        "--prefill-chunk",
        type=int,
        default=256,
        help="Chunk size when streaming prefill tokens"
    )
    
    # 输出参数
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/decoding/decoding_latency.json"),
        help="输出文件路径"
    )
    
    return parser.parse_args()


def generate_random_prompt(tokenizer, length: int) -> torch.Tensor:
    """生成随机 prompt"""
    # 使用固定种子确保可复现
    torch.manual_seed(42)
    vocab_size = len(tokenizer)
    # 避免特殊 token
    token_ids = torch.randint(100, vocab_size - 100, (1, length))
    return token_ids


def load_prefill_tokens(
    tokenizer,
    path: Path,
    field: str,
    max_tokens: int,
) -> torch.Tensor:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        text = data.get(field, "")
    elif isinstance(data, list) and data:
        text = data[0].get(field, "")
    else:
        text = ""
    if not text:
        raise ValueError(f"Prefill file {path} missing text field '{field}'.")
    tokens = tokenizer(text, return_tensors="pt").input_ids
    if max_tokens and tokens.shape[1] > max_tokens:
        tokens = tokens[:, :max_tokens]
    return tokens


def prefill_model(
    model,
    input_ids: torch.Tensor,
    chunk_size: int,
    streaming_wrapper=None,
):
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    past_key_values = None
    last_logits = None
    total_len = input_ids.shape[1]
    ctx = streaming_wrapper.enable() if streaming_wrapper is not None else nullcontext()
    with ctx:
        for start in range(0, total_len, chunk_size):
            chunk = input_ids[:, start : start + chunk_size]
            with torch.no_grad():
                outputs = model(
                    input_ids=chunk,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values
            last_logits = outputs.logits[:, -1, :]
            if streaming_wrapper is not None:
                past_key_values = streaming_wrapper.update(past_key_values)
    last_token = input_ids[:, -1:]
    return past_key_values, last_logits, last_token


def warmup_model(
    model,
    input_ids: torch.Tensor,
    device: torch.device,
    num_tokens: int,
    use_streaming: bool = False,
    streaming_wrapper = None,
    past_key_values=None,
) -> None:
    """
    GPU warmup 阶段
    
    Args:
        model: 语言模型
        input_ids: 初始输入 [1, seq_len]
        device: 设备
        num_tokens: warmup token 数量
        use_streaming: 是否使用 StreamingLLM
        streaming_wrapper: StreamingLLMWrapper 实例
    """
    print(f"  Warmup: 生成 {num_tokens} tokens...")
    
    if use_streaming and streaming_wrapper is not None:
        context = streaming_wrapper.enable()
    else:
        from contextlib import nullcontext
        context = nullcontext()
    
    with context:
        with torch.no_grad():
            current_ids = input_ids.to(device)
            
            for _ in range(num_tokens):
                outputs = model(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                # 获取下一个 token
                next_token = outputs.logits[:, -1:].argmax(dim=-1)
                current_ids = next_token
                past_key_values = outputs.past_key_values
                
                if use_streaming and streaming_wrapper is not None:
                    streaming_wrapper.update(past_key_values)
    
    # 清空 CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    print("  Warmup 完成")


def _init_spec_state(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    target_wrapper=None,
    draft_wrapper=None,
    prefill_ids: Optional[torch.Tensor] = None,
    prefill_chunk: int = 256,
):
    device = next(target_model.parameters()).device
    input_ids = input_ids.to(device)
    if prefill_ids is not None and prefill_ids.numel() > 0:
        target_past, target_next_logits, last_token = prefill_model(
            target_model,
            prefill_ids,
            chunk_size=prefill_chunk,
            streaming_wrapper=target_wrapper,
        )
        draft_past, draft_next_logits, _ = prefill_model(
            draft_model,
            prefill_ids,
            chunk_size=prefill_chunk,
            streaming_wrapper=draft_wrapper,
        )
        return target_past, draft_past, target_next_logits, draft_next_logits

    with torch.no_grad():
        target_out = target_model(input_ids=input_ids, use_cache=True)
        target_past = target_out.past_key_values
        target_next_logits = target_out.logits[:, -1, :]
        if target_wrapper is not None:
            target_past = target_wrapper.update(target_past)

        draft_out = draft_model(input_ids=input_ids, use_cache=True)
        draft_past = draft_out.past_key_values
        draft_next_logits = draft_out.logits[:, -1, :]
        if draft_wrapper is not None:
            draft_past = draft_wrapper.update(draft_past)

    return target_past, draft_past, target_next_logits, draft_next_logits


def _speculative_step(
    target_model,
    draft_model,
    target_past,
    draft_past,
    target_next_logits: torch.Tensor,
    draft_next_logits: torch.Tensor,
    proposal_len: int,
    temperature: float,
    spec_mode: str,
    target_wrapper=None,
    draft_wrapper=None,
    generator: Optional[torch.Generator] = None,
):
    draft_temperature = 0.0 if spec_mode == "greedy_match" else temperature
    proposed, draft_logits, draft_past, draft_next_logits = propose_tokens(
        model=draft_model,
        first_logits=draft_next_logits,
        past_key_values=draft_past,
        k=proposal_len,
        temperature=draft_temperature,
        generator=generator,
    )

    with torch.no_grad():
        target_out = target_model(
            input_ids=proposed,
            past_key_values=target_past,
            use_cache=True,
        )

    target_verify_logits = target_out.logits
    if spec_mode == "greedy_match":
        accepted_len, fallback_token, bonus_token = greedy_match(
            proposed=proposed,
            target_next_logits=target_next_logits,
            target_verify_logits=target_verify_logits,
        )
    else:
        accepted_len, fallback_token, bonus_token = exact_accept(
            proposed=proposed,
            draft_logits=draft_logits,
            target_next_logits=target_next_logits,
            target_verify_logits=target_verify_logits,
            temperature=temperature,
            generator=generator,
        )

    proposed_len = proposed.shape[1]
    target_past = commit_and_prune(
        target_out.past_key_values,
        accepted_len=accepted_len,
        proposed_len=proposed_len,
        streaming_wrapper=target_wrapper,
    )
    draft_past = commit_and_prune(
        draft_past,
        accepted_len=accepted_len,
        proposed_len=proposed_len,
        streaming_wrapper=draft_wrapper,
    )

    next_token = fallback_token if fallback_token is not None else bonus_token
    if next_token is None:
        raise RuntimeError("SpecDec step produced no next token.")

    with torch.no_grad():
        target_out = target_model(
            input_ids=next_token,
            past_key_values=target_past,
            use_cache=True,
        )
        target_past = target_out.past_key_values
        target_next_logits = target_out.logits[:, -1, :]
        if target_wrapper is not None:
            target_past = target_wrapper.update(target_past)

        draft_out = draft_model(
            input_ids=next_token,
            past_key_values=draft_past,
            use_cache=True,
        )
        draft_past = draft_out.past_key_values
        draft_next_logits = draft_out.logits[:, -1, :]
        if draft_wrapper is not None:
            draft_past = draft_wrapper.update(draft_past)

    generated = accepted_len + 1
    return (
        generated,
        accepted_len,
        proposed_len,
        target_past,
        draft_past,
        target_next_logits,
        draft_next_logits,
    )


def warmup_speculative(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    num_tokens: int,
    proposal_len: int,
    temperature: float,
    spec_mode: str,
    target_wrapper=None,
    draft_wrapper=None,
    prefill_ids: Optional[torch.Tensor] = None,
    prefill_chunk: int = 256,
    generator: Optional[torch.Generator] = None,
):
    print(f"  Warmup(SpecDec): 生成 {num_tokens} tokens...")

    target_ctx = target_wrapper.enable() if target_wrapper is not None else nullcontext()
    draft_ctx = draft_wrapper.enable() if draft_wrapper is not None else nullcontext()
    with target_ctx, draft_ctx:
        (
            target_past,
            draft_past,
            target_next_logits,
            draft_next_logits,
        ) = _init_spec_state(
            target_model=target_model,
            draft_model=draft_model,
            input_ids=input_ids,
            target_wrapper=target_wrapper,
            draft_wrapper=draft_wrapper,
            prefill_ids=prefill_ids,
            prefill_chunk=prefill_chunk,
        )

        generated = 0
        while generated < num_tokens:
            (
                step_gen,
                _accepted,
                _proposed,
                target_past,
                draft_past,
                target_next_logits,
                draft_next_logits,
            ) = _speculative_step(
                target_model=target_model,
                draft_model=draft_model,
                target_past=target_past,
                draft_past=draft_past,
                target_next_logits=target_next_logits,
                draft_next_logits=draft_next_logits,
                proposal_len=proposal_len,
                temperature=temperature,
                spec_mode=spec_mode,
                target_wrapper=target_wrapper,
                draft_wrapper=draft_wrapper,
                generator=generator,
            )
            generated += step_gen

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    print("  Warmup 完成")


def measure_decoding_latency(
    model,
    input_ids: torch.Tensor,
    device: torch.device,
    num_tokens: int,
    cache_size: int,
    use_streaming: bool = False,
    streaming_wrapper = None,
    past_key_values=None,
) -> Tuple[List[float], float]:
    """
    测量 per-token decoding latency
    
    Args:
        model: 语言模型
        input_ids: 初始输入 [1, seq_len]
        device: 设备
        num_tokens: 要生成的 token 数量
        cache_size: cache 大小 (仅用于参考,不影响测量逻辑)
        use_streaming: 是否使用 StreamingLLM
        streaming_wrapper: StreamingLLMWrapper 实例
    
    Returns:
        latencies: 每个 token 的延迟列表 (秒)
        peak_memory_mb: 峰值显存 (MB)
    """
    latencies = []
    
    if use_streaming and streaming_wrapper is not None:
        context = streaming_wrapper.enable()
    else:
        from contextlib import nullcontext
        context = nullcontext()
    
    with context:
        with torch.no_grad():
            current_ids = input_ids.to(device)
            
            for step in range(num_tokens):
                # 同步 CUDA 确保准确计时
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                t0 = time.perf_counter()
                
                outputs = model(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                # 同步 CUDA
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                t1 = time.perf_counter()
                latencies.append(t1 - t0)
                
                # 获取下一个 token
                next_token = outputs.logits[:, -1:].argmax(dim=-1)
                current_ids = next_token
                past_key_values = outputs.past_key_values
                
                if use_streaming and streaming_wrapper is not None:
                    streaming_wrapper.update(past_key_values)
    
    # 获取峰值显存
    peak_memory_mb = 0
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return latencies, peak_memory_mb


def measure_speculative_latency(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    device: torch.device,
    num_tokens: int,
    proposal_len: int,
    temperature: float,
    spec_mode: str,
    target_wrapper=None,
    draft_wrapper=None,
    prefill_ids: Optional[torch.Tensor] = None,
    prefill_chunk: int = 256,
    generator: Optional[torch.Generator] = None,
) -> Tuple[List[float], float, Dict]:
    latencies = []
    accepted_total = 0
    proposed_total = 0
    verify_steps = 0
    target_forwards = 0

    target_ctx = target_wrapper.enable() if target_wrapper is not None else nullcontext()
    draft_ctx = draft_wrapper.enable() if draft_wrapper is not None else nullcontext()
    with target_ctx, draft_ctx:
        (
            target_past,
            draft_past,
            target_next_logits,
            draft_next_logits,
        ) = _init_spec_state(
            target_model=target_model,
            draft_model=draft_model,
            input_ids=input_ids,
            target_wrapper=target_wrapper,
            draft_wrapper=draft_wrapper,
            prefill_ids=prefill_ids,
            prefill_chunk=prefill_chunk,
        )

        generated = 0
        while generated < num_tokens:
            remaining = num_tokens - generated
            step_k = min(proposal_len, max(1, remaining))

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            (
                step_gen,
                accepted_len,
                proposed_len,
                target_past,
                draft_past,
                target_next_logits,
                draft_next_logits,
            ) = _speculative_step(
                target_model=target_model,
                draft_model=draft_model,
                target_past=target_past,
                draft_past=draft_past,
                target_next_logits=target_next_logits,
                draft_next_logits=draft_next_logits,
                proposal_len=step_k,
                temperature=temperature,
                spec_mode=spec_mode,
                target_wrapper=target_wrapper,
                draft_wrapper=draft_wrapper,
                generator=generator,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            accepted_total += accepted_len
            proposed_total += proposed_len
            verify_steps += 1
            target_forwards += 2

            used = min(step_gen, remaining)
            if used > 0:
                per_token = (t1 - t0) / used
                latencies.extend([per_token] * used)
            generated += used

    peak_memory_mb = 0
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    accept_rate = accepted_total / proposed_total if proposed_total > 0 else 0.0
    avg_accepted = accepted_total / verify_steps if verify_steps > 0 else 0.0
    tokens_per_target = generated / target_forwards if target_forwards > 0 else 0.0

    stats = {
        "accept_rate": accept_rate,
        "avg_accepted_per_step": avg_accepted,
        "tokens_per_target_forward": tokens_per_target,
        "accepted_tokens": accepted_total,
        "proposed_tokens": proposed_total,
        "verify_steps": verify_steps,
        "target_forwards": target_forwards,
    }

    return latencies, peak_memory_mb, stats


def evaluate_baseline(
    model,
    draft_model,
    tokenizer,
    device: torch.device,
    args,
    spec_mode: str,
    prefill_ids: Optional[torch.Tensor],
    prefill_chunk: int,
    generator: Optional[torch.Generator] = None,
) -> Dict:
    """评估 Baseline (Full KV Cache)"""
    print("\n" + "="*60)
    print("评估 Baseline (Full KV Cache)")
    print("="*60)
    
    all_latencies = []
    all_memories = []
    spec_totals = {
        "accepted_tokens": 0,
        "proposed_tokens": 0,
        "verify_steps": 0,
        "target_forwards": 0,
    }
    spec_enabled = args.decoder == "speculative"
    
    for run in range(args.num_runs):
        print(f"\nRun {run + 1}/{args.num_runs}")

        run_generator = None
        if spec_enabled and generator is not None:
            run_generator = torch.Generator(device=device).manual_seed(
                args.seed + run
            )
        
        if prefill_ids is not None:
            input_ids = prefill_ids[:, -1:]
            prefill_past, _, _ = prefill_model(
                model,
                prefill_ids,
                chunk_size=prefill_chunk,
                streaming_wrapper=None,
            )
        else:
            input_ids = generate_random_prompt(tokenizer, args.prompt_length)
            prefill_past = None
        
        if spec_enabled:
            warmup_speculative(
                target_model=model,
                draft_model=draft_model,
                input_ids=input_ids,
                num_tokens=args.warmup_tokens,
                proposal_len=args.draft_k,
                temperature=args.temperature,
                spec_mode=spec_mode,
                prefill_ids=prefill_ids,
                prefill_chunk=prefill_chunk,
                generator=run_generator,
            )

            print(f"  测量: 生成 {args.num_tokens} tokens...")
            latencies, peak_memory, spec_stats = measure_speculative_latency(
                target_model=model,
                draft_model=draft_model,
                input_ids=input_ids,
                device=device,
                num_tokens=args.num_tokens,
                proposal_len=args.draft_k,
                temperature=args.temperature,
                spec_mode=spec_mode,
                prefill_ids=prefill_ids,
                prefill_chunk=prefill_chunk,
                generator=run_generator,
            )
            for key in spec_totals:
                spec_totals[key] += spec_stats.get(key, 0)
        else:
            warmup_model(
                model=model,
                input_ids=input_ids,
                device=device,
                num_tokens=args.warmup_tokens,
                use_streaming=False,
                past_key_values=prefill_past,
            )

            print(f"  测量: 生成 {args.num_tokens} tokens...")
            latencies, peak_memory = measure_decoding_latency(
                model=model,
                input_ids=input_ids,
                device=device,
                num_tokens=args.num_tokens,
                cache_size=args.cache_size,
                use_streaming=False,
                past_key_values=prefill_past,
            )
            spec_stats = None
        
        all_latencies.extend(latencies)
        all_memories.append(peak_memory)
        
        # 安全地计算平均值
        if len(latencies) > 0:
            print(f"  收集到 {len(latencies)} 个延迟测量")
            print(f"  平均延迟: {np.mean(latencies)*1000:.3f} ms/token")
        else:
            print(f"  错误: 没有收集到延迟数据!")
        print(f"  峰值显存: {peak_memory:.1f} MB")
    
    # 统计 - 添加空数组检查
    if len(all_latencies) > 0:
        mean_latency = np.mean(all_latencies)
        std_latency = np.std(all_latencies) if len(all_latencies) > 1 else 0.0
        median_latency = np.median(all_latencies)
    else:
        print("警告: 没有收集到任何延迟数据,使用默认值")
        mean_latency = 0.0
        std_latency = 0.0
        median_latency = 0.0
    
    if len(all_memories) > 0:
        mean_memory = np.mean(all_memories)
    else:
        mean_memory = 0.0
    
    print(f"\n总结 ({args.num_runs} runs):")
    print(f"  平均延迟: {mean_latency*1000:.3f} ± {std_latency*1000:.3f} ms/token")
    print(f"  中位数延迟: {median_latency*1000:.3f} ms/token")
    print(f"  平均显存: {mean_memory:.1f} MB")
    
    results = {
        "mean_latency_ms": mean_latency * 1000,
        "std_latency_ms": std_latency * 1000,
        "median_latency_ms": median_latency * 1000,
        "mean_memory_mb": mean_memory,
        "all_latencies_ms": [l * 1000 for l in all_latencies],
    }
    if spec_enabled:
        proposed = spec_totals["proposed_tokens"]
        accepted = spec_totals["accepted_tokens"]
        verify_steps = spec_totals["verify_steps"]
        target_forwards = spec_totals["target_forwards"]
        results["speculative"] = {
            "accept_rate": accepted / proposed if proposed > 0 else 0.0,
            "avg_accepted_per_step": accepted / verify_steps if verify_steps > 0 else 0.0,
            "tokens_per_target_forward": (
                (args.num_tokens * args.num_runs) / target_forwards
                if target_forwards > 0
                else 0.0
            ),
            "accepted_tokens": accepted,
            "proposed_tokens": proposed,
            "verify_steps": verify_steps,
            "target_forwards": target_forwards,
        }
    return results


def evaluate_streaming_llm(
    model,
    draft_model,
    tokenizer,
    device: torch.device,
    args,
    spec_mode: str,
    prefill_ids: Optional[torch.Tensor],
    prefill_chunk: int,
    generator: Optional[torch.Generator] = None,
) -> Dict:
    """评估 StreamingLLM"""
    print("\n" + "="*60)
    print("评估 StreamingLLM (我们的实现)")
    print("="*60)
    print(f"  n_sink: {args.n_sink}")
    print(f"  window_size: {args.cache_size}")
    
    all_latencies = []
    all_memories = []
    spec_totals = {
        "accepted_tokens": 0,
        "proposed_tokens": 0,
        "verify_steps": 0,
        "target_forwards": 0,
    }
    spec_enabled = args.decoder == "speculative"
    
    for run in range(args.num_runs):
        print(f"\nRun {run + 1}/{args.num_runs}")

        run_generator = None
        if spec_enabled and generator is not None:
            run_generator = torch.Generator(device=device).manual_seed(
                args.seed + run
            )
        
        # 创建 StreamingLLM wrapper
        streaming_wrapper = StreamingLLMWrapper(
            model=model,
            n_sink=args.n_sink,
            window_size=args.cache_size
        )
        draft_wrapper = None
        if spec_enabled:
            draft_wrapper = StreamingLLMWrapper(
                model=draft_model,
                n_sink=args.n_sink,
                window_size=args.cache_size,
            )

        if prefill_ids is not None:
            input_ids = prefill_ids[:, -1:]
            prefill_past = None
            if not spec_enabled:
                prefill_past, _, _ = prefill_model(
                    model,
                    prefill_ids,
                    chunk_size=prefill_chunk,
                    streaming_wrapper=streaming_wrapper,
                )
        else:
            input_ids = generate_random_prompt(tokenizer, args.prompt_length)
            prefill_past = None

        if spec_enabled:
            warmup_speculative(
                target_model=model,
                draft_model=draft_model,
                input_ids=input_ids,
                num_tokens=args.warmup_tokens,
                proposal_len=args.draft_k,
                temperature=args.temperature,
                spec_mode=spec_mode,
                target_wrapper=streaming_wrapper,
                draft_wrapper=draft_wrapper,
                prefill_ids=prefill_ids,
                prefill_chunk=prefill_chunk,
                generator=run_generator,
            )

            print(f"  测量: 生成 {args.num_tokens} tokens...")
            latencies, peak_memory, spec_stats = measure_speculative_latency(
                target_model=model,
                draft_model=draft_model,
                input_ids=input_ids,
                device=device,
                num_tokens=args.num_tokens,
                proposal_len=args.draft_k,
                temperature=args.temperature,
                spec_mode=spec_mode,
                target_wrapper=streaming_wrapper,
                draft_wrapper=draft_wrapper,
                prefill_ids=prefill_ids,
                prefill_chunk=prefill_chunk,
                generator=run_generator,
            )
            for key in spec_totals:
                spec_totals[key] += spec_stats.get(key, 0)
        else:
            warmup_model(
                model=model,
                input_ids=input_ids,
                device=device,
                num_tokens=args.warmup_tokens,
                use_streaming=True,
                streaming_wrapper=streaming_wrapper,
                past_key_values=prefill_past,
            )

            print(f"  测量: 生成 {args.num_tokens} tokens...")
            latencies, peak_memory = measure_decoding_latency(
                model=model,
                input_ids=input_ids,
                device=device,
                num_tokens=args.num_tokens,
                cache_size=args.cache_size,
                use_streaming=True,
                streaming_wrapper=streaming_wrapper,
                past_key_values=prefill_past,
            )
        
        all_latencies.extend(latencies)
        all_memories.append(peak_memory)
        
        # 安全地计算平均值
        if len(latencies) > 0:
            print(f"  收集到 {len(latencies)} 个延迟测量")
            print(f"  平均延迟: {np.mean(latencies)*1000:.3f} ms/token")
        else:
            print(f"  错误: 没有收集到延迟数据!")
        print(f"  峰值显存: {peak_memory:.1f} MB")
    
    # 统计 - 添加空数组检查
    if len(all_latencies) > 0:
        mean_latency = np.mean(all_latencies)
        std_latency = np.std(all_latencies) if len(all_latencies) > 1 else 0.0
        median_latency = np.median(all_latencies)
    else:
        print("警告: 没有收集到任何延迟数据,使用默认值")
        mean_latency = 0.0
        std_latency = 0.0
        median_latency = 0.0
    
    if len(all_memories) > 0:
        mean_memory = np.mean(all_memories)
    else:
        mean_memory = 0.0
    
    print(f"\n总结 ({args.num_runs} runs):")
    print(f"  平均延迟: {mean_latency*1000:.3f} ± {std_latency*1000:.3f} ms/token")
    print(f"  中位数延迟: {median_latency*1000:.3f} ms/token")
    print(f"  平均显存: {mean_memory:.1f} MB")
    
    results = {
        "mean_latency_ms": mean_latency * 1000,
        "std_latency_ms": std_latency * 1000,
        "median_latency_ms": median_latency * 1000,
        "mean_memory_mb": mean_memory,
        "all_latencies_ms": [l * 1000 for l in all_latencies],
    }
    if spec_enabled:
        proposed = spec_totals["proposed_tokens"]
        accepted = spec_totals["accepted_tokens"]
        verify_steps = spec_totals["verify_steps"]
        target_forwards = spec_totals["target_forwards"]
        results["speculative"] = {
            "accept_rate": accepted / proposed if proposed > 0 else 0.0,
            "avg_accepted_per_step": accepted / verify_steps if verify_steps > 0 else 0.0,
            "tokens_per_target_forward": (
                (args.num_tokens * args.num_runs) / target_forwards
                if target_forwards > 0
                else 0.0
            ),
            "accepted_tokens": accepted,
            "proposed_tokens": proposed,
            "verify_steps": verify_steps,
            "target_forwards": target_forwards,
        }
    return results


def main():
    args = parse_args()
    
    # 设置设备和数据类型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_mapping[args.dtype]
    
    if device.type == "cpu" and torch_dtype != torch.float32:
        print("警告: CPU 不支持 fp16/bf16,切换到 fp32")
        torch_dtype = torch.float32
    
    print(f"\n{'='*60}")
    print(f"Per-Token Decoding Latency 评估")
    print(f"{'='*60}")
    print(f"模型: {args.model_name}")
    print(f"设备: {device}")
    print(f"数据类型: {torch_dtype}")
    print(f"Cache Size: {args.cache_size}")
    print(f"Prompt Length: {args.prompt_length}")
    print(f"Num Tokens: {args.num_tokens}")
    print(f"Warmup Tokens: {args.warmup_tokens}")
    print(f"Num Runs: {args.num_runs}")
    print(f"Mode: {args.mode}")
    print(f"Decoder: {args.decoder}")
    if args.decoder == "speculative":
        print(f"SpecDec Mode: {args.speculative_mode}")
        print(f"Draft Model: {args.draft_model_name}")
        print(f"Draft k: {args.draft_k}")
        print(f"Temperature: {args.temperature}")
    if args.prefill_file is not None:
        print(f"Prefill File: {args.prefill_file}")
        print(f"Prefill Tokens: {args.prefill_tokens or 'all'}")
        print(f"Prefill Chunk: {args.prefill_chunk}")
    print(f"{'='*60}\n")
    
    # 加载 tokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prefill_ids = None
    if args.prefill_file is not None:
        prefill_ids = load_prefill_tokens(
            tokenizer=tokenizer,
            path=args.prefill_file,
            field=args.prefill_field,
            max_tokens=args.prefill_tokens,
        )
        print(f"Prefill tokens: {prefill_ids.shape[1]}")
    
    # 加载模型
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()

    draft_model = None
    if args.decoder == "speculative":
        print("加载 draft 模型...")
        draft_model = AutoModelForCausalLM.from_pretrained(
            args.draft_model_name,
            torch_dtype=torch_dtype,
        ).to(device)
        draft_model.eval()

    spec_mode = args.speculative_mode
    if args.decoder == "speculative" and args.temperature <= 0 and spec_mode == "exact":
        print("提示: temperature=0 时自动切换为 greedy_match")
        spec_mode = "greedy_match"

    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    baseline_results = None
    baseline_source = None
    
    if args.mode in {"both", "baseline"}:
        baseline_results = evaluate_baseline(
            model=model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            device=device,
            args=args,
            spec_mode=spec_mode,
            prefill_ids=prefill_ids,
            prefill_chunk=args.prefill_chunk,
            generator=generator,
        )
        baseline_source = "computed"
    elif args.baseline_results:
        if not args.baseline_results.exists():
            raise FileNotFoundError(f"基线结果文件不存在: {args.baseline_results}")
        cache_data = json.loads(args.baseline_results.read_text())
        baseline_results = cache_data.get("baseline")
        if baseline_results is None:
            raise ValueError("提供的基线结果文件不包含 baseline 数据")
        baseline_source = f"loaded:{args.baseline_results}"
    else:
        raise ValueError("Streaming 模式需要提供 --baseline-results")
    
    streaming_results = None
    comparison = None
    
    if args.mode in {"both", "streaming"}:
        streaming_results = evaluate_streaming_llm(
            model=model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            device=device,
            args=args,
            spec_mode=spec_mode,
            prefill_ids=prefill_ids,
            prefill_chunk=args.prefill_chunk,
            generator=generator,
        )
        
        if streaming_results["mean_latency_ms"] > 0:
            speedup = baseline_results["mean_latency_ms"] / streaming_results["mean_latency_ms"]
        else:
            print("警告: StreamingLLM 延迟为 0,无法计算加速比")
            speedup = 0.0
        
        if baseline_results["mean_memory_mb"] > 0:
            memory_reduction = (
                baseline_results["mean_memory_mb"] - streaming_results["mean_memory_mb"]
            ) / baseline_results["mean_memory_mb"]
        else:
            print("警告: Baseline 显存在 0,无法计算显存减少比例")
            memory_reduction = 0.0
        
        comparison = {
            "speedup": speedup,
            "memory_reduction_percent": memory_reduction * 100,
        }
    
    results = {
        "model": args.model_name,
        "device": str(device),
        "dtype": str(torch_dtype),
        "mode": args.mode,
        "baseline_source": baseline_source,
        "config": {
            "cache_size": args.cache_size,
            "n_sink": args.n_sink,
            "prompt_length": args.prompt_length,
            "num_tokens": args.num_tokens,
            "warmup_tokens": args.warmup_tokens,
            "num_runs": args.num_runs,
            "decoder": args.decoder,
            "speculative_mode": spec_mode if args.decoder == "speculative" else None,
            "draft_model_name": args.draft_model_name if args.decoder == "speculative" else None,
            "draft_k": args.draft_k if args.decoder == "speculative" else None,
            "temperature": args.temperature,
            "seed": args.seed,
            "prefill_file": str(args.prefill_file) if args.prefill_file else None,
            "prefill_tokens": args.prefill_tokens,
            "prefill_chunk": args.prefill_chunk,
        },
    }
    if baseline_results:
        results["baseline"] = baseline_results
    if streaming_results:
        results["streaming_llm"] = streaming_results
    if comparison:
        results["comparison"] = comparison
    
    if streaming_results and comparison:
        print(f"\n{'='*60}")
        print(f"最终对比")
        print(f"{'='*60}")
        print(
            f"Baseline:      {baseline_results['mean_latency_ms']:.3f} ± "
            f"{baseline_results['std_latency_ms']:.3f} ms/token"
        )
        print(
            f"StreamingLLM:  {streaming_results['mean_latency_ms']:.3f} ± "
            f"{streaming_results['std_latency_ms']:.3f} ms/token"
        )
        print(f"加速比:        {comparison['speedup']:.2f}x")
        print(f"显存减少:      {comparison['memory_reduction_percent']:.1f}%")
        print(f"{'='*60}\n")
    elif baseline_results and args.mode == "baseline":
        print(f"\n{'='*60}")
        print("基线评估完成")
        print(f"{'='*60}")
        print(
            f"Baseline: {baseline_results['mean_latency_ms']:.3f} ± "
            f"{baseline_results['std_latency_ms']:.3f} ms/token"
        )
        print(f"{'='*60}\n")
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
