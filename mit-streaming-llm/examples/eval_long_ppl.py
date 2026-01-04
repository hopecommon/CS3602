import argparse
import json
import os
import time

import sys
from pathlib import Path

# Ensure the vendored MIT repo (`mit-streaming-llm/streaming_llm`) is importable when this
# script is invoked from another working directory (e.g., repo root).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss

from streaming_llm.kv_cache import StartRecentKVCache
from streaming_llm.utils import load

device = "cuda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIT StreamingLLM eval_long_ppl (offline-capable)")
    parser.add_argument("--model_name_or_path", type=str, default="models/llama/llama-7b")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--task", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs/debug")

    parser.add_argument("--enable_start_recent_kv_cache", action="store_true")
    parser.add_argument("--start_size", type=int, default=1)
    parser.add_argument("--recent_size", type=int, default=255)
    parser.add_argument("--enable_pos_shift", action="store_true")

    # Original flag kept for compatibility.
    parser.add_argument("--num_eval_tokens", type=int, default=None)
    # Additional offline-friendly knobs.
    parser.add_argument("--data-json", type=str, default=None, help="Path to a presampled JSON (expects a 'text' key).")
    parser.add_argument("--text-key", type=str, default="text")
    parser.add_argument(
        "--max-eval-tokens",
        type=int,
        default=None,
        help="Truncate the tokenized input to at most this many tokens (per sample).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="If set, write a single JSON file with metrics (PPL/runtime/etc).",
    )
    return parser.parse_args()


def _load_texts(args: argparse.Namespace) -> list[str]:
    if args.data_json:
        with open(args.data_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            if not obj:
                raise ValueError(f"Empty JSON list: {args.data_json}")
            obj = obj[0]
        if not isinstance(obj, dict) or args.text_key not in obj:
            raise ValueError(f"Expected a JSON object with key '{args.text_key}': {args.data_json}")
        text = obj[args.text_key]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Expected '{args.text_key}' to be a non-empty string in {args.data_json}")
        return [text]

    # Fallback to original datasets behavior when --data-json is not provided.
    from datasets import load_dataset

    data = load_dataset(args.dataset_name, args.task, split=args.split)
    return list(data["text"][: args.num_samples])


args = parse_args()
texts = _load_texts(args)

model, tokenizer = load(args.model_name_or_path)
model_device_map = getattr(model, "hf_device_map", None)

nlls = []
loss_fn = CrossEntropyLoss(reduction="none")
past_key_values = None

if args.enable_start_recent_kv_cache:
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "pythia" in model.config.model_type or "gpt_neox" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
    else:
        raise ValueError(f"got {model.config.model_type}")
    kv_cache = StartRecentKVCache(
        start_size=args.start_size,
        recent_size=args.recent_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
else:
    kv_cache = None

if args.enable_pos_shift:
    if "llama" in model.config.model_type:
        from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention

        enable_llama_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        from streaming_llm.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )

        enable_falcon_pos_shift_attention(model)
    elif "gpt_neox" in model.config.model_type:
        from streaming_llm.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )

        enable_gpt_neox_pos_shift_attention(model)
    elif "mpt" in model.config.model_type:
        pass
    else:
        raise ValueError(f"got {model.config.model_type}")

os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")

# Transformers >= 4.46 may use the new Cache API. The MIT StartRecentKVCache expects
# legacy past_key_values (tuple/list of tensors). Convert back and forth if needed.
try:
    from transformers.cache_utils import DynamicCache
except Exception:
    DynamicCache = None  # type: ignore[misc,assignment]


def _to_legacy(past):
    if past is None:
        return None
    if hasattr(past, "to_legacy_cache"):
        return past.to_legacy_cache()
    return past


def _from_legacy(legacy):
    if legacy is None:
        return None
    if DynamicCache is None:
        return legacy
    return DynamicCache.from_legacy_cache(legacy)


def _reset_cuda_peak_stats_all_devices() -> None:
    if not torch.cuda.is_available():
        return
    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_idx)


def _read_cuda_peak_stats() -> list[dict]:
    if not torch.cuda.is_available():
        return []
    stats: list[dict] = []
    for device_idx in range(torch.cuda.device_count()):
        stats.append(
            {
                "device": device_idx,
                "max_allocated_mb": torch.cuda.max_memory_allocated(device_idx) / 1024**2,
                "max_reserved_mb": torch.cuda.max_memory_reserved(device_idx) / 1024**2,
            }
        )
    return stats


if torch.cuda.is_available() and device == "cuda":
    _reset_cuda_peak_stats_all_devices()
    total_start_evt = torch.cuda.Event(enable_timing=True)
    total_end_evt = torch.cuda.Event(enable_timing=True)
    first_start_evt = torch.cuda.Event(enable_timing=True)
    first_end_evt = torch.cuda.Event(enable_timing=True)
    total_start_evt.record()
else:
    total_start_time = time.perf_counter()
first_token_latency_sec = 0.0
first_recorded = False

num_eval_tokens = 0
for text in texts:
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)

    if args.max_eval_tokens is not None and args.max_eval_tokens > 0:
        encodings.input_ids = encodings.input_ids[:, : args.max_eval_tokens]

    print(encodings.input_ids[:, :10])

    seq_len = encodings.input_ids.size(1)
    print(f"seq_len: {seq_len}")
    pbar = tqdm(range(0, seq_len - 1))

    for idx in pbar:
        input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
        if torch.cuda.is_available() and device == "cuda" and not first_recorded:
            first_start_evt.record()
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
                legacy = _to_legacy(past_key_values)
                legacy = kv_cache(legacy)
                past_key_values = _from_legacy(legacy)
        if torch.cuda.is_available() and device == "cuda" and not first_recorded:
            first_end_evt.record()
        nlls.append(neg_log_likelihood)
        pbar.set_description(
            f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
        )
        print(neg_log_likelihood.item(), file=f, flush=True)
        num_eval_tokens += 1
        if not first_recorded:
            first_recorded = True
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break
    if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
        break

f.close()

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
with open(f"{args.output_dir}/ppl.txt", "w") as f:
    f.write(f"{ppl.item()}\n")

if torch.cuda.is_available() and device == "cuda":
    total_end_evt.record()
    torch.cuda.synchronize()
    runtime_sec = total_start_evt.elapsed_time(total_end_evt) / 1000.0
    if first_recorded:
        first_token_latency_sec = first_start_evt.elapsed_time(first_end_evt) / 1000.0
    # NOTE: `streaming_llm.utils.load()` uses `device_map="auto"`, which can shard the
    # model across multiple GPUs. If we only read the default device's peak, the
    # reported number can be misleadingly small. Report peak across all CUDA devices.
    cuda_peak = _read_cuda_peak_stats()
    try:
        peak_memory_mb = float(max(s["max_allocated_mb"] for s in cuda_peak)) if cuda_peak else 0.0
    except Exception:
        peak_memory_mb = 0.0
else:
    runtime_sec = time.perf_counter() - total_start_time
    peak_memory_mb = 0.0
    cuda_peak = []

tpot_ms = float(runtime_sec / max(num_eval_tokens, 1) * 1000.0)

if args.output_json:
    out = {
        "method": "mit_eval_long_ppl",
        "model": args.model_name_or_path,
        "dataset": args.dataset_name,
        "task": args.task,
        "split": args.split,
        "data_json": args.data_json,
        "text_key": args.text_key,
        "max_eval_tokens": args.max_eval_tokens,
        "num_eval_tokens": args.num_eval_tokens,
        "start_size": args.start_size,
        "recent_size": args.recent_size,
        "enable_pos_shift": bool(args.enable_pos_shift),
        "hf_device_map": model_device_map,
        "metrics": {
            "perplexity": float(ppl.item()),
            "runtime_sec": float(runtime_sec),
            "first_token_latency_sec": float(first_token_latency_sec),
            "eval_tokens": int(num_eval_tokens),
            "tpot_ms": float(tpot_ms),
            "peak_memory_mb": float(peak_memory_mb),
        },
        "cuda_peak": cuda_peak,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
