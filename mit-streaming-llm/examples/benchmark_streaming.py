import argparse
import json
import time
from typing import List, Optional, Tuple

import sys
from pathlib import Path

# Ensure the vendored MIT repo (`mit-streaming-llm/streaming_llm`) is importable when this
# script is invoked from another working directory (e.g., repo root).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from streaming_llm.enable_streaming_llm import enable_streaming_llm


def _optional_tqdm(iterable, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc)
    except Exception:
        return iterable


def _infer_kv_seq_dims(model_type: str) -> Tuple[int, int]:
    if "llama" in model_type:
        return 2, 2
    if "mpt" in model_type:
        return 3, 2
    if "pythia" in model_type:
        return 2, 2
    if "falcon" in model_type:
        return 1, 1
    if "gpt_neox" in model_type:
        return 2, 2
    raise ValueError(f"Unsupported model_type: {model_type}")


def _kv_seq_len(past_key_values, k_seq_dim: int) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length())
        except Exception:
            pass
    return int(past_key_values[0][0].size(k_seq_dim))


def _reset_cuda_peak_stats() -> None:
    if not torch.cuda.is_available():
        return
    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_idx)


def _read_cuda_peak_stats() -> List[dict]:
    if not torch.cuda.is_available():
        return []
    stats = []
    for device_idx in range(torch.cuda.device_count()):
        stats.append(
            {
                "device": device_idx,
                "max_allocated_mb": torch.cuda.max_memory_allocated(device_idx)
                / 1024**2,
                "max_reserved_mb": torch.cuda.max_memory_reserved(device_idx) / 1024**2,
            }
        )
    return stats


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _build_prefix_ids(
    tokenizer, *, prefix_tokens: int, seed_text: str = " hello"
) -> torch.LongTensor:
    piece = tokenizer.encode(seed_text, add_special_tokens=False)
    if len(piece) == 0:
        raise ValueError("Tokenizer produced empty tokenization for seed_text.")
    ids: List[int] = []
    while len(ids) < prefix_tokens:
        ids.extend(piece)
    ids = ids[:prefix_tokens]
    return torch.tensor([ids], dtype=torch.long)


def _build_prefix_ids_from_json(
    tokenizer,
    *,
    json_path: str,
    text_key: str,
    prefix_tokens: int,
    take: str,
) -> torch.LongTensor:
    with open(json_path, "r") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or text_key not in obj:
        raise ValueError(f"Expected a JSON object with key '{text_key}': {json_path}")
    text = obj[text_key]
    if not isinstance(text, str):
        raise ValueError(f"Expected '{text_key}' to be a string, got {type(text)}")

    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 0:
        raise ValueError("Tokenizer produced empty tokenization for input text.")

    if len(ids) < prefix_tokens:
        repeated: List[int] = []
        while len(repeated) < prefix_tokens:
            repeated.extend(ids)
        ids = repeated

    if take == "head":
        ids = ids[:prefix_tokens]
    elif take == "tail":
        ids = ids[-prefix_tokens:]
    else:
        raise ValueError(f"Unsupported take={take}, expected head|tail")
    return torch.tensor([ids], dtype=torch.long)


def _load_model_and_tokenizer(model_name_or_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else 0
    model.eval()
    model.to(device)
    return model, tokenizer


def _enable_pos_shift_only(model) -> None:
    model_type = model.config.model_type
    if "llama" in model_type:
        from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention

        enable_llama_pos_shift_attention(model)
        return
    if "gpt_neox" in model_type:
        from streaming_llm.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )

        enable_gpt_neox_pos_shift_attention(model)
        return
    if "falcon" in model_type:
        from streaming_llm.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )

        enable_falcon_pos_shift_attention(model)
        return
    if "mpt" in model_type or "pythia" in model_type:
        return
    raise ValueError(f"Unsupported model_type for pos_shift: {model_type}")


@torch.no_grad()
def _prefill_in_chunks(
    model,
    input_ids: torch.LongTensor,
    *,
    chunk_size: int,
    kv_cache=None,
    k_seq_dim: int,
    show_progress: bool = False,
) -> Tuple[Optional[Tuple[torch.Tensor, ...]], int]:
    # Transformers >= 4.46 may use the new Cache API (e.g., DynamicCache) for `past_key_values`.
    # The MIT StartRecentKVCache expects the legacy cache format (list/tuple of tensors).
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        DynamicCache = None  # type: ignore[misc,assignment]

    def to_legacy(past):
        if past is None:
            return None
        if hasattr(past, "to_legacy_cache"):
            return past.to_legacy_cache()
        return past

    def from_legacy(legacy):
        if legacy is None:
            return None
        if DynamicCache is None:
            return legacy
        return DynamicCache.from_legacy_cache(legacy)

    past_key_values = None
    total = input_ids.size(1)
    for start in _optional_tqdm(range(0, total, chunk_size), show_progress, "prefill"):
        end = min(total, start + chunk_size)
        chunk = input_ids[:, start:end]
        outputs = model(input_ids=chunk, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        if kv_cache is not None:
            legacy = to_legacy(past_key_values)
            legacy = kv_cache(legacy)
            past_key_values = from_legacy(legacy)
    return past_key_values, _kv_seq_len(past_key_values, k_seq_dim)


@torch.no_grad()
def _decode_full_or_streaming(
    model,
    *,
    past_key_values,
    gen_tokens: int,
    kv_cache=None,
    k_seq_dim: int,
    show_progress: bool = False,
) -> Tuple[int, int]:
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        DynamicCache = None  # type: ignore[misc,assignment]

    def to_legacy(past):
        if past is None:
            return None
        if hasattr(past, "to_legacy_cache"):
            return past.to_legacy_cache()
        return past

    def from_legacy(legacy):
        if legacy is None:
            return None
        if DynamicCache is None:
            return legacy
        return DynamicCache.from_legacy_cache(legacy)

    eos = getattr(model.config, "eos_token_id", None)
    next_token = torch.tensor([[eos if eos is not None else 0]], device=model.device)
    for _ in _optional_tqdm(range(gen_tokens), show_progress, "decode"):
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        if kv_cache is not None:
            legacy = to_legacy(past_key_values)
            legacy = kv_cache(legacy)
            past_key_values = from_legacy(legacy)
    return gen_tokens, _kv_seq_len(past_key_values, k_seq_dim)


@torch.no_grad()
def _decode_recompute(
    model,
    *,
    prefix_ids: torch.LongTensor,
    gen_tokens: int,
    window_tokens: int,
    keep_start_tokens: int,
    show_progress: bool = False,
) -> int:
    eos = getattr(model.config, "eos_token_id", None)
    generated: List[int] = []
    for _ in _optional_tqdm(range(gen_tokens), show_progress, "decode(recompute)"):
        if keep_start_tokens > 0:
            start_part = prefix_ids[:, :keep_start_tokens]
        else:
            start_part = None

        if len(generated) > 0:
            generated_part = torch.tensor([generated], device=model.device, dtype=torch.long)
            full = torch.cat([prefix_ids.to(model.device), generated_part], dim=1)
        else:
            full = prefix_ids.to(model.device)

        if start_part is not None:
            tail_source = full[:, keep_start_tokens:]
            tail = tail_source[:, -window_tokens:]
            context = torch.cat([start_part.to(model.device), tail], dim=1)
        else:
            context = full[:, -window_tokens:]

        outputs = model(input_ids=context, use_cache=False)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1).item()
        if eos is not None and next_token == eos:
            break
        generated.append(next_token)
    return len(generated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="streaming",
        choices=["streaming", "full", "recompute"],
    )
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2048)
    parser.add_argument("--prefix_tokens", type=int, default=4096)
    parser.add_argument("--prefill_chunk_size", type=int, default=512)
    parser.add_argument("--gen_tokens", type=int, default=512)
    parser.add_argument("--seed_text", type=str, default=" hello")
    parser.add_argument(
        "--data_json",
        type=str,
        default=None,
        help="Optional JSON file containing a long text field for prefill.",
    )
    parser.add_argument("--data_text_key", type=str, default="text")
    parser.add_argument(
        "--data_take",
        type=str,
        default="head",
        choices=["head", "tail"],
        help="Take the head/tail tokens from the text for prefill.",
    )
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="Show a progress bar for prefill/decode (prints nothing by default).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="If set, write a JSON file with benchmark metrics.",
    )

    parser.add_argument("--recompute_window_tokens", type=int, default=2048)
    parser.add_argument("--recompute_keep_start", action="store_true")
    parser.add_argument(
        "--enable_pos_shift",
        action="store_true",
        help="Enable position shift attention (needed if you want to exceed max_position_embeddings in full mode).",
    )

    args = parser.parse_args()

    model, tokenizer = _load_model_and_tokenizer(args.model_name_or_path, args.device)
    model_type = model.config.model_type
    k_seq_dim, _v_seq_dim = _infer_kv_seq_dims(model_type)

    kv_cache = None
    if args.mode == "streaming":
        kv_cache = enable_streaming_llm(
            model,
            start_size=args.start_size,
            recent_size=args.recent_size,
            enable_pos_shift=args.enable_pos_shift,
        )
    elif args.mode == "full" and args.enable_pos_shift:
        _enable_pos_shift_only(model)

    if args.data_json is not None:
        prefix_ids = _build_prefix_ids_from_json(
            tokenizer,
            json_path=args.data_json,
            text_key=args.data_text_key,
            prefix_tokens=args.prefix_tokens,
            take=args.data_take,
        ).to(model.device)
    else:
        prefix_ids = _build_prefix_ids(
            tokenizer, prefix_tokens=args.prefix_tokens, seed_text=args.seed_text
        ).to(model.device)

    _reset_cuda_peak_stats()
    _sync()
    if args.mode == "recompute":
        past_key_values = None
        kv_len_after_prefill = 0
        prefill_s = 0.0
        prefill_peak = _read_cuda_peak_stats()
    else:
        t0 = time.perf_counter()
        past_key_values, kv_len_after_prefill = _prefill_in_chunks(
            model,
            prefix_ids,
            chunk_size=args.prefill_chunk_size,
            kv_cache=kv_cache if args.mode == "streaming" else None,
            k_seq_dim=k_seq_dim,
            show_progress=args.show_progress,
        )
        _sync()
        prefill_s = time.perf_counter() - t0
        prefill_peak = _read_cuda_peak_stats()

    _reset_cuda_peak_stats()
    if args.mode in {"streaming", "full"}:
        _sync()
        t1 = time.perf_counter()
        produced, kv_len_after_decode = _decode_full_or_streaming(
            model,
            past_key_values=past_key_values,
            gen_tokens=args.gen_tokens,
            kv_cache=kv_cache if args.mode == "streaming" else None,
            k_seq_dim=k_seq_dim,
            show_progress=args.show_progress,
        )
        _sync()
        decode_s = time.perf_counter() - t1
    else:
        keep_start = args.start_size if args.recompute_keep_start else 0
        _sync()
        t1 = time.perf_counter()
        produced = _decode_recompute(
            model,
            prefix_ids=prefix_ids,
            gen_tokens=args.gen_tokens,
            window_tokens=args.recompute_window_tokens,
            keep_start_tokens=keep_start,
            show_progress=args.show_progress,
        )
        _sync()
        decode_s = time.perf_counter() - t1
        kv_len_after_decode = 0

    decode_peak = _read_cuda_peak_stats()

    decode_toks_per_s = float(produced / decode_s) if decode_s > 0 else 0.0
    peak_mem_mb = 0.0
    if decode_peak:
        try:
            peak_mem_mb = float(max(s["max_allocated_mb"] for s in decode_peak))
        except Exception:
            peak_mem_mb = 0.0

    print(f"mode: {args.mode}")
    print(f"model_type: {model_type}")
    print(f"prefix_tokens: {args.prefix_tokens}")
    print(f"gen_tokens_requested: {args.gen_tokens}")
    print(f"gen_tokens_produced: {produced}")
    print(f"prefill_seconds: {prefill_s:.4f}")
    print(f"decode_seconds: {decode_s:.4f}")
    if decode_s > 0:
        print(f"decode_tokens_per_second: {decode_toks_per_s:.2f}")
    print(f"kv_len_after_prefill: {kv_len_after_prefill}")
    print(f"kv_len_after_decode: {kv_len_after_decode}")
    if prefill_peak:
        for s in prefill_peak:
            print(
                f"cuda:{s['device']} prefill_peak_allocated_mb={s['max_allocated_mb']:.1f} "
                f"prefill_peak_reserved_mb={s['max_reserved_mb']:.1f}"
            )
    if decode_peak:
        for s in decode_peak:
            print(
                f"cuda:{s['device']} decode_peak_allocated_mb={s['max_allocated_mb']:.1f} "
                f"decode_peak_reserved_mb={s['max_reserved_mb']:.1f}"
            )

    if args.output_json:
        out = {
            "script": "mit-streaming-llm/examples/benchmark_streaming.py",
            "mode": args.mode,
            "model_name_or_path": args.model_name_or_path,
            "model_type": model_type,
            "device": args.device,
            "start_size": int(args.start_size),
            "recent_size": int(args.recent_size),
            "prefix_tokens": int(args.prefix_tokens),
            "prefill_chunk_size": int(args.prefill_chunk_size),
            "gen_tokens_requested": int(args.gen_tokens),
            "gen_tokens_produced": int(produced),
            "data_json": args.data_json,
            "data_text_key": args.data_text_key,
            "data_take": args.data_take,
            "recompute_window_tokens": int(args.recompute_window_tokens),
            "recompute_keep_start": bool(args.recompute_keep_start),
            "enable_pos_shift": bool(args.enable_pos_shift),
            "metrics": {
                "prefill_sec": float(prefill_s),
                "decode_sec": float(decode_s),
                "decode_tokens_per_second": float(decode_toks_per_s),
                "tpot_ms": float(decode_s / max(int(produced), 1) * 1000.0),
                "peak_memory_mb": float(peak_mem_mb),
                "kv_len_after_prefill": int(kv_len_after_prefill),
                "kv_len_after_decode": int(kv_len_after_decode),
            },
            "cuda_peak": {
                "prefill": prefill_peak,
                "decode": decode_peak,
            },
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote benchmark JSON to: {args.output_json}")


if __name__ == "__main__":
    main()
