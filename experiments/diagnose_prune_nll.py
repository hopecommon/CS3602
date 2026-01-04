#!/usr/bin/env python3
"""
Diagnose prune-induced NLL spikes and latency spikes for StreamingLLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming_llm import StreamingLLMWrapper
from experiments.eval_utils import load_tokenized_dataset, tqdm_wrap


def _maybe_load_dotenv(repo_root: Path, overwrite: bool = False) -> None:
    """
    Load `.env` into os.environ without requiring extra dependencies.
    This keeps dataset sample selection consistent across machines.
    """
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not overwrite and key in os.environ:
            continue
        os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose prune NLL spikes")
    parser.add_argument("--model-name", type=str, default=os.environ.get("MODEL_NAME", "EleutherAI/pythia-2.8b"))
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--dataset-name", type=str, default="pg19")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-eval-tokens", type=int, default=20000)
    parser.add_argument("--n-sink", type=int, default=4)
    # Auto-cap: fix a total budget and derive window size.
    parser.add_argument("--cap-total", type=int, default=0, help="If >0, enforce auto-cap and derive window_size.")
    parser.add_argument("--window-size", type=int, default=2048, help="Used when --cap-total=0.")
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--refresh-budget", type=int, default=0)
    parser.add_argument("--refresh-policy", type=str, default="none", choices=["none", "uniform"])
    parser.add_argument("--compress-every", type=int, default=32)
    parser.add_argument("--cache-slack", type=int, default=0)
    parser.add_argument("--max-drop", type=int, default=0)
    parser.add_argument("--spike-window", type=int, default=32, help="Tokens after prune to measure spikes")
    parser.add_argument("--max-decode-steps", type=int, default=0, help="If >0, only decode this many steps after prefill.")
    parser.add_argument("--load-dotenv", action="store_true", default=True)
    parser.add_argument("--no-dotenv", dest="load_dotenv", action="store_false")
    parser.add_argument("--output-dir", type=Path, default=Path("results/diagnose_prune"))
    return parser.parse_args()


def _dtype_from_arg(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    return float(values[min(max(idx, 0), len(values) - 1)])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.load_dotenv:
        _maybe_load_dotenv(Path(__file__).resolve().parents[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = _dtype_from_arg(args.dtype)
    if device.type == "cpu" and torch_dtype != torch.float32:
        torch_dtype = torch.float32

    if args.cap_total and args.cap_total > 0:
        derived = args.cap_total - args.n_sink - args.cache_slack - args.overlap - args.refresh_budget
        if derived <= 0:
            raise ValueError(
                f"Invalid auto-cap window_size={derived} (cap_total={args.cap_total}, sink={args.n_sink}, "
                f"slack={args.cache_slack}, overlap={args.overlap}, refresh={args.refresh_budget})"
            )
        args.window_size = int(derived)
        print(
            f"Auto-cap enabled: cap_total={args.cap_total} => window_size={args.window_size} "
            f"(sink={args.n_sink}, slack={args.cache_slack}, overlap={args.overlap}, refresh={args.refresh_budget})"
        )

    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载数据集...")
    encoded = load_tokenized_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column=args.text_column,
        max_samples=args.max_samples,
        tokenizer=tokenizer,
        max_eval_tokens=args.max_eval_tokens,
    )

    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()

    wrapper = StreamingLLMWrapper(
        model=model,
        n_sink=args.n_sink,
        window_size=args.window_size,
        overlap=args.overlap,
        refresh_budget=args.refresh_budget,
        refresh_policy=args.refresh_policy,
        compress_every=args.compress_every,
        cache_slack=args.cache_slack,
        max_drop=args.max_drop,
    )

    if encoded.device != device:
        encoded = encoded.to(device)

    seq_len = encoded.size(1)
    prefill_len = min(args.n_sink + args.window_size, seq_len)

    rows: list[dict] = []
    prune_events: list[dict] = []
    kv_lengths: list[int] = []
    step_times: list[float] = []

    with wrapper.enable():
        input_ids = encoded[:, :prefill_len]
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        wrapper.update(past_key_values)

        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        if labels.numel() > 0:
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1),
                reduction="none",
            )
            for idx, nll_val in enumerate(nll.tolist()):
                rows.append({
                    "token_index": idx + 1,
                    "nll": nll_val,
                    "time_ms": 0.0,
                    "kv_len": prefill_len,
                    "pruned": 0,
                    "dropped": 0,
                    "step": -1,
                })

        decode_steps = max(0, seq_len - prefill_len)
        if args.max_decode_steps and args.max_decode_steps > 0:
            decode_steps = min(decode_steps, int(args.max_decode_steps))
        for pos in tqdm_wrap(
            range(prefill_len - 1, prefill_len - 1 + decode_steps),
            total=decode_steps,
            desc="Diagnose decode",
            unit="tok",
        ):
            current_input = encoded[:, pos:pos + 1]
            target = encoded[:, pos + 1]

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                outputs = model(
                    input_ids=current_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            logits = outputs.logits[:, -1, :]
            nll_val = F.cross_entropy(
                logits.float(),
                target,
                reduction="sum",
            ).item()

            past_key_values = outputs.past_key_values
            wrapper.update(past_key_values)
            prune_info = wrapper.pop_last_prune()
            if prune_info is not None:
                prune_events.append(prune_info)

            kv_len = past_key_values.layers[0].keys.shape[2]
            kv_lengths.append(kv_len)
            time_ms = (end - start) * 1000.0
            step_times.append(time_ms)

            rows.append({
                "token_index": pos + 1,
                "nll": nll_val,
                "time_ms": time_ms,
                "kv_len": kv_len,
                "pruned": 1 if prune_info else 0,
                "dropped": prune_info["dropped"] if prune_info else 0,
                "step": prune_info["step"] if prune_info else -1,
            })

    nll_values = [r["nll"] for r in rows if r["time_ms"] > 0]
    baseline_nll = median(nll_values) if nll_values else 0.0

    spike_window = max(1, int(args.spike_window))
    spikes = []
    for event in prune_events:
        start_idx = event["step"]
        if start_idx < 0:
            continue
        end_idx = min(len(rows), start_idx + spike_window)
        window_nll = [rows[i]["nll"] for i in range(start_idx, end_idx)]
        if not window_nll:
            continue
        peak = max(window_nll)
        area = sum(max(0.0, v - baseline_nll) for v in window_nll)
        width = sum(1 for v in window_nll if v > baseline_nll)
        spikes.append({
            "step": event["step"],
            "peak": peak,
            "area": area,
            "width": width,
        })

    def _median(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(median(values))

    def _p90(values: list[float]) -> float:
        if not values:
            return 0.0
        return _percentile(values, 0.9)

    peak_list = [s["peak"] for s in spikes]
    area_list = [s["area"] for s in spikes]
    width_list = [s["width"] for s in spikes]

    summary = {
        "model": args.model_name,
        "dataset": f"{args.dataset_name}:{args.dataset_config}",
        "dtype": args.dtype,
        "cap_total": args.cap_total if args.cap_total else None,
        "window_size": args.window_size,
        "n_sink": args.n_sink,
        "overlap": args.overlap,
        "refresh_budget": args.refresh_budget,
        "refresh_policy": args.refresh_policy,
        "compress_every": args.compress_every,
        "cache_slack": args.cache_slack,
        "max_drop": args.max_drop,
        "total_tokens": seq_len,
        "prefill_len": prefill_len,
        "decode_steps": decode_steps,
        "baseline_nll_median": baseline_nll,
        "kv_len_min": min(kv_lengths) if kv_lengths else prefill_len,
        "kv_len_mean": sum(kv_lengths) / len(kv_lengths) if kv_lengths else prefill_len,
        "kv_len_max": max(kv_lengths) if kv_lengths else prefill_len,
        "drop_sizes": [e["dropped"] for e in prune_events],
        "prune_events": len(prune_events),
        "latency_ms_mean": sum(step_times) / len(step_times) if step_times else 0.0,
        "latency_ms_p50": _percentile(step_times, 0.5),
        "latency_ms_p90": _percentile(step_times, 0.9),
        "spike_window": spike_window,
        "spike_stats": spikes,
        "spike_peak_median": _median(peak_list),
        "spike_peak_p90": _p90(peak_list),
        "spike_area_median": _median(area_list),
        "spike_area_p90": _p90(area_list),
        "spike_width_median": _median(width_list),
        "spike_width_p90": _p90(width_list),
    }

    output_prefix = (
        f"{args.dataset_name}_cap{args.cap_total or (args.n_sink + args.window_size)}"
        f"_S{args.n_sink}_W{args.window_size}"
        f"_K{args.compress_every}_sigma{args.cache_slack}_delta{args.max_drop}"
        f"_o{args.overlap}_r{args.refresh_budget}"
    )
    csv_path = args.output_dir / f"{output_prefix}_per_token.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["token_index", "nll", "time_ms", "kv_len", "pruned", "dropped", "step"],
        )
        writer.writeheader()
        writer.writerows(rows)

    (args.output_dir / f"{output_prefix}_prune_events.json").write_text(
        json.dumps(prune_events, indent=2)
    )
    (args.output_dir / f"{output_prefix}_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(f"已保存 per-token CSV: {csv_path}")
    print(f"已保存 summary: {args.output_dir / f'{output_prefix}_summary.json'}")


if __name__ == "__main__":
    main()
