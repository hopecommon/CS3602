import torch
import time
import json
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from streaming_llm.enable_streaming_llm import enable_streaming_llm
from streaming_llm.utils import load

# ================= 配置区域 =================
class Args:
    # 模型路径或名称
    model_name_or_path = "EleutherAI/pythia-2.8b"
    
    # 数据集配置
    dataset_name = "pg19" 
    # 本地数据文件路径 (如果不为空，则优先使用此文件)
    dataset_path = "../data/pg19/long_context_20000.json"
    task = "wikitext-2-raw-v1"
    split = "test"
    num_samples = 1
    
    # 输出目录
    output_dir = "results/pg19_local_test"
    
    # StreamingLLM 核心参数
    enable_start_recent_kv_cache = True
    start_size = 32      # Sink tokens
    recent_size = 2016   # Window size
    
    # 其他配置
    enable_pos_shift = False
    num_eval_tokens = 50000  # 评估长度

args = Args()
device = "cuda" if torch.cuda.is_available() else "cpu"
# ===========================================

def main():
    print(f"Running experiment on {device}...")
    print(f"Model: {args.model_name_or_path}")
    print(f"Window: {args.recent_size}, Sink: {args.start_size}")

    # 1. 加载数据
    if args.dataset_path:
        print(f"Loading local dataset from {args.dataset_path}")
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            # 处理可能的不同 json 格式
            if isinstance(content, dict) and "text" in content:
                data = {"text": [content["text"]]}
            elif isinstance(content, list): # jsonl list
                data = {"text": [item["text"] for item in content]}
            else:
                # 尝试作为纯文本读取
                f.seek(0)
                data = {"text": [f.read()]}
    elif args.dataset_name == "pg19":
        data = load_dataset(args.dataset_name, split=args.split)
    else:
        data = load_dataset(args.dataset_name, args.task, split=args.split)

    # 2. 加载模型
    model, tokenizer = load(args.model_name_or_path)

    nlls = []
    loss_fn = CrossEntropyLoss(reduction="none")
    past_key_values = None

    # 3. 设置 Streaming LLM (使用官方 API)
    if args.enable_start_recent_kv_cache:
        kv_cache = enable_streaming_llm(
            model,
            start_size=args.start_size,
            recent_size=args.recent_size
        )
    else:
        kv_cache = None

    # 4. Positional Shift (已由 enable_streaming_llm 处理，此处不再重复)
    # 如果未使用 enable_streaming_llm 但需要 pos_shift，可手动调用，但本脚本主要演示 StreamingLLM 功能


    # 5. 准备输出
    os.makedirs(args.output_dir, exist_ok=True)
    f_log = open(f"{args.output_dir}/log.txt", "w")

    # 6. 开始评估循环
    num_eval_tokens = 0
    # 简单的处理：只取第一个样本，因为通常 pg19 一个样本就很长
    text_list = data["text"] if isinstance(data, dict) else data
    
    for text in text_list[: args.num_samples]:
        encodings = tokenizer(text, return_tensors="pt")

        print(f"Sample initial tokens: {encodings.input_ids[:, :10]}")
        seq_len = encodings.input_ids.size(1)
        print(f"Total sequence length: {seq_len}")
        
        pbar = tqdm(range(0, seq_len - 1))
        
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for idx in pbar:
            input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = outputs.logits.view(-1, model.config.vocab_size)
                past_key_values = outputs.past_key_values
                label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
                neg_log_likelihood = loss_fn(logits, label)
                if kv_cache is not None:
                    past_key_values = kv_cache(past_key_values)
            nlls.append(neg_log_likelihood)
            
            # 性能监控更新
            if (idx + 1) % 100 == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    mem_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)
                else:
                    mem_usage = 0
                    
                elapsed = time.time() - start_time
                avg_speed = (idx + 1) / elapsed
                
                pbar.set_description(
                    f"nll: {neg_log_likelihood.item():.2f}, "
                    f"ppl: {torch.exp(neg_log_likelihood).item():.2f}, "
                    f"mem: {mem_usage:.0f}MB, spd: {avg_speed:.1f}t/s"
                )
            elif idx % 100 == 0: # 减少刷新频率
                 pbar.set_description(
                    f"nll: {neg_log_likelihood.item():.2f}, "
                    f"ppl: {torch.exp(neg_log_likelihood).item():.2f}"
                )
                
            print(neg_log_likelihood.item(), file=f_log, flush=True)
            num_eval_tokens += 1
            if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
                break
        
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break

    f_log.close()

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"\nFinal PPL: {ppl.item()}")
    with open(f"{args.output_dir}/ppl.txt", "w") as f:
        f.write(f"{ppl.item()}\n")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
