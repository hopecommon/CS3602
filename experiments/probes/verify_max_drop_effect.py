#!/usr/bin/env python3
"""
Verify whether Max_Drop helps under a controlled setting.

This script runs two StreamingLLM configurations that only differ by `max_drop`:
  - delta=0 (Max_Drop disabled)
  - delta>0 (Max_Drop enabled)

It records detailed debug signals for each run:
  - per-token NLL / time / kv_len
  - prune events (count, drop sizes, overflow)
  - NLL spike statistics aligned to prune events (peak/area/width)

It is intended as a fast, reproducible sanity check to validate that Max_Drop
reduces eviction-induced NLL spikes without regressing TPOT.

Run from repo root:
  python -m experiments.probes.verify_max_drop_effect --dataset-name pg19 --n-sink 32 --cap-total 2048 --compress-every 64 --cache-slack 16 --max-drop 32
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
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repo_root()))

from experiments.eval_utils import load_tokenized_dataset, tqdm_wrap  # noqa: E402
from streaming_llm import StreamingLLMWrapper  # noqa: E402


def _maybe_load_dotenv(repo_root: Path, overwrite: bool = False) -> None:
    """
    Load `.env` into os.environ without requiring external dependencies.
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


def _derive_window(cap_total: int, sink: int, slack: int, overlap: int, refresh: int) -> int:
    window = cap_total - sink - slack - overlap - refresh
    if window <= 0:
        raise ValueError(
            f"Invalid auto-cap window_size={window} (cap_total={cap_total}, sink={sink}, slack={slack}, overlap={overlap}, refresh={refresh})"
        )
    return int(window)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Max_Drop effect under controlled settings.")
    parser.add_argument("--model-name", type=str, default=os.environ.get("MODEL_NAME", "EleutherAI/pythia-2.8b"))
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--dataset-name", type=str, default="pg19")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-eval-tokens", type=int, default=20000)

    # Auto-cap controls
    parser.add_argument("--cap-total", type=int, default=2048)
    parser.add_argument("--n-sink", type=int, default=32)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--refresh-budget", type=int, default=0)
    parser.add_argument("--refresh-policy", type=str, default="none", choices=["none", "uniform", "middle"])
    parser.add_argument("--compress-every", type=int, default=64)
    parser.add_argument("--cache-slack", type=int, default=16)
    parser.add_argument("--max-drop", type=int, default=32)

    # Diagnostics
    parser.add_argument("--spike-window", type=int, default=32, help="Tokens after prune to measure NLL spikes.")
    parser.add_argument("--max-decode-steps", type=int, default=0, help="If >0, only decode this many steps after prefill.")
    parser.add_argument("--fp32-loss", action="store_true", default=True)
    parser.add_argument("--no-fp32-loss", dest="fp32_loss", action="store_false")
    parser.add_argument("--load-dotenv", action="store_true", default=True)
    parser.add_argument("--no-dotenv", dest="load_dotenv", action="store_false")

    parser.add_argument("--output-dir", type=Path, default=Path("results/probes/verify_max_drop"))
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def _run_once(
    *,
    args: argparse.Namespace,
    model,
    encoded: torch.Tensor,
    tokenizer,
    device: torch.device,
    window_size: int,
    max_drop: int,
    output_prefix: str,
) -> dict[str, Any]:
    wrapper = StreamingLLMWrapper(
        model=model,
        n_sink=args.n_sink,
        window_size=window_size,
        overlap=args.overlap,
        refresh_budget=args.refresh_budget,
        refresh_policy=args.refresh_policy,
        compress_every=args.compress_every,
        cache_slack=args.cache_slack,
        max_drop=max_drop,
    )

    if encoded.device != device:
        encoded = encoded.to(device)

    seq_len = encoded.size(1)
    prefill_len = min(args.n_sink + window_size, seq_len)
    if prefill_len < 2:
        raise ValueError(f"prefill_len too small: {prefill_len}")

    rows: list[dict[str, Any]] = []
    prune_events: list[dict[str, Any]] = []
    kv_lengths: list[int] = []
    step_times: list[float] = []
    nll_values: list[float] = []
    total_nll: float = 0.0
    total_pred: int = 0

    with wrapper.enable():
        input_ids = encoded[:, :prefill_len]
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        wrapper.update(past_key_values)

        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        if labels.numel() > 0:
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_labels = labels.reshape(-1)
            if args.fp32_loss:
                flat_logits = flat_logits.float()
            nll = F.cross_entropy(flat_logits, flat_labels, reduction="none")
            for idx, nll_val in enumerate(nll.tolist()):
                rows.append(
                    {
                        "token_index": idx + 1,
                        "nll": float(nll_val),
                        "time_ms": 0.0,
                        "kv_len": prefill_len,
                        "pruned": 0,
                        "dropped": 0,
                        "step": -1,
                    }
                )
            total_nll += float(nll.sum().item())
            total_pred += int(nll.numel())

        decode_total = max(0, seq_len - prefill_len)
        decode_steps = decode_total
        if args.max_decode_steps and args.max_decode_steps > 0:
            decode_steps = min(decode_steps, int(args.max_decode_steps))

        for pos in tqdm_wrap(
            range(prefill_len - 1, prefill_len - 1 + decode_steps),
            total=decode_steps,
            desc=f"Decode (delta={max_drop})",
            unit="tok",
        ):
            current_input = encoded[:, pos : pos + 1]
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
            if args.fp32_loss:
                logits = logits.float()
            nll_val = F.cross_entropy(logits, target, reduction="sum").item()

            past_key_values = outputs.past_key_values
            wrapper.update(past_key_values)
            prune_info = wrapper.pop_last_prune()
            if prune_info is not None:
                prune_events.append(prune_info)

            kv_len = past_key_values.layers[0].keys.shape[2]
            kv_lengths.append(int(kv_len))

            time_ms = (end - start) * 1000.0
            step_times.append(float(time_ms))
            nll_values.append(float(nll_val))
            total_nll += float(nll_val)
            total_pred += 1

            rows.append(
                {
                    "token_index": pos + 1,
                    "nll": float(nll_val),
                    "time_ms": float(time_ms),
                    "kv_len": int(kv_len),
                    "pruned": 1 if prune_info else 0,
                    "dropped": int(prune_info["dropped"]) if prune_info else 0,
                    "step": int(prune_info["step"]) if prune_info else -1,
                }
            )

    baseline_nll = float(median(nll_values)) if nll_values else 0.0
    spike_window = max(1, int(args.spike_window))
    spikes: list[dict[str, Any]] = []
    for event in prune_events:
        start_idx = int(event["step"])
        if start_idx < 0:
            continue
        end_idx = min(len(rows), start_idx + spike_window)
        window_nll = [float(rows[i]["nll"]) for i in range(start_idx, end_idx)]
        if not window_nll:
            continue
        peak = max(window_nll)
        area = sum(max(0.0, v - baseline_nll) for v in window_nll)
        width = sum(1 for v in window_nll if v > baseline_nll)
        spikes.append({"step": start_idx, "peak": float(peak), "area": float(area), "width": int(width)})

    peak_list = [float(s["peak"]) for s in spikes]
    area_list = [float(s["area"]) for s in spikes]
    width_list = [float(s["width"]) for s in spikes]

    ppl = float(torch.exp(torch.tensor(total_nll / max(total_pred, 1))).item())

    summary: dict[str, Any] = {
        "model": args.model_name,
        "dataset": f"{args.dataset_name}:{args.dataset_config}",
        "dtype": args.dtype,
        "fp32_loss": bool(args.fp32_loss),
        "cap_total": int(args.cap_total),
        "n_sink": int(args.n_sink),
        "window_size": int(window_size),
        "cache_slack": int(args.cache_slack),
        "overlap": int(args.overlap),
        "refresh_budget": int(args.refresh_budget),
        "refresh_policy": str(args.refresh_policy),
        "compress_every": int(args.compress_every),
        "max_drop": int(max_drop),
        "total_tokens": int(seq_len),
        "prefill_len": int(prefill_len),
        "decode_steps": int(decode_steps),
        "ppl_estimate": ppl,
        "baseline_nll_median": baseline_nll,
        "kv_len_min": int(min(kv_lengths)) if kv_lengths else int(prefill_len),
        "kv_len_mean": float(sum(kv_lengths) / len(kv_lengths)) if kv_lengths else float(prefill_len),
        "kv_len_max": int(max(kv_lengths)) if kv_lengths else int(prefill_len),
        "drop_sizes": [int(e["dropped"]) for e in prune_events],
        "overflow_sizes": [int(e["overflow"]) for e in prune_events],
        "prune_events": int(len(prune_events)),
        "latency_ms_mean": float(sum(step_times) / len(step_times)) if step_times else 0.0,
        "latency_ms_p50": float(_percentile(step_times, 0.5)),
        "latency_ms_p90": float(_percentile(step_times, 0.9)),
        "spike_window": int(spike_window),
        "spike_peak_median": float(median(peak_list)) if peak_list else 0.0,
        "spike_peak_p90": float(_percentile(peak_list, 0.9)),
        "spike_area_median": float(median(area_list)) if area_list else 0.0,
        "spike_area_p90": float(_percentile(area_list, 0.9)),
        "spike_width_median": float(median(width_list)) if width_list else 0.0,
        "spike_width_p90": float(_percentile(width_list, 0.9)),
        "spike_stats": spikes,
    }
    # Minimal runtime diagnostics for debugging.
    summary["debug"] = {
        "drop_sizes_median": float(median(summary["drop_sizes"])) if summary["drop_sizes"] else 0.0,
        "drop_sizes_p90": float(_percentile([float(x) for x in summary["drop_sizes"]], 0.9)) if summary["drop_sizes"] else 0.0,
        "overflow_sizes_median": float(median(summary["overflow_sizes"])) if summary["overflow_sizes"] else 0.0,
        "overflow_sizes_p90": float(_percentile([float(x) for x in summary["overflow_sizes"]], 0.9)) if summary["overflow_sizes"] else 0.0,
        "tokens_in_eval_nll": int(total_pred),
        "decode_steps_total_possible": int(seq_len - prefill_len),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_token_path = out_dir / f"{output_prefix}_per_token.csv"
    with per_token_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["token_index", "nll", "time_ms", "kv_len", "pruned", "dropped", "step"],
        )
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / f"{output_prefix}_prune_events.json").write_text(json.dumps(prune_events, indent=2))
    (out_dir / f"{output_prefix}_summary.json").write_text(json.dumps(summary, indent=2))

    return summary


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()
    if args.load_dotenv:
        _maybe_load_dotenv(repo_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = _dtype_from_arg(args.dtype)
    if device.type == "cpu" and torch_dtype != torch.float32:
        torch_dtype = torch.float32

    window_size = _derive_window(
        int(args.cap_total),
        int(args.n_sink),
        int(args.cache_slack),
        int(args.overlap),
        int(args.refresh_budget),
    )

    tag = args.tag.strip() or f"{args.dataset_name}_S{args.n_sink}_cap{args.cap_total}_K{args.compress_every}_sigma{args.cache_slack}_delta{args.max_drop}"
    print(f"[verify] tag={tag}")
    print(f"[verify] auto-cap: cap_total={args.cap_total} => window_size={window_size} (sink={args.n_sink} slack={args.cache_slack} overlap={args.overlap} refresh={args.refresh_budget})")
    print(f"[verify] controls: K={args.compress_every} sigma={args.cache_slack} delta={args.max_drop} fp32_loss={args.fp32_loss}")

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
    if encoded.device != device:
        encoded = encoded.to(device)

    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch_dtype).to(device)
    model.eval()

    out_dir = Path(args.output_dir)
    prefix_base = f"{tag}_base_delta0"
    prefix_md = f"{tag}_maxdrop_delta{args.max_drop}"

    base = _run_once(
        args=args,
        model=model,
        encoded=encoded,
        tokenizer=tokenizer,
        device=device,
        window_size=window_size,
        max_drop=0,
        output_prefix=prefix_base,
    )
    md = _run_once(
        args=args,
        model=model,
        encoded=encoded,
        tokenizer=tokenizer,
        device=device,
        window_size=window_size,
        max_drop=int(args.max_drop),
        output_prefix=prefix_md,
    )

    def _get(d: dict[str, Any], k: str, default: float = 0.0) -> float:
        v = d.get(k, default)
        return float(v) if isinstance(v, (int, float)) else default

    compare = {
        "tag": tag,
        "cap_total": int(args.cap_total),
        "window_size": int(window_size),
        "n_sink": int(args.n_sink),
        "compress_every": int(args.compress_every),
        "cache_slack": int(args.cache_slack),
        "delta": int(args.max_drop),
        "fp32_loss": bool(args.fp32_loss),
        "base": {
            "ppl_estimate": _get(base, "ppl_estimate"),
            "latency_ms_mean": _get(base, "latency_ms_mean"),
            "latency_ms_p90": _get(base, "latency_ms_p90"),
            "prune_events": int(base.get("prune_events", 0)),
            "spike_area_median": _get(base, "spike_area_median"),
            "spike_peak_median": _get(base, "spike_peak_median"),
        },
        "max_drop": {
            "ppl_estimate": _get(md, "ppl_estimate"),
            "latency_ms_mean": _get(md, "latency_ms_mean"),
            "latency_ms_p90": _get(md, "latency_ms_p90"),
            "prune_events": int(md.get("prune_events", 0)),
            "spike_area_median": _get(md, "spike_area_median"),
            "spike_peak_median": _get(md, "spike_peak_median"),
        },
    }
    compare["delta_metrics"] = {
        "ppl_estimate_delta": compare["max_drop"]["ppl_estimate"] - compare["base"]["ppl_estimate"],
        "latency_ms_mean_delta": compare["max_drop"]["latency_ms_mean"] - compare["base"]["latency_ms_mean"],
        "latency_ms_p90_delta": compare["max_drop"]["latency_ms_p90"] - compare["base"]["latency_ms_p90"],
        "prune_events_delta": compare["max_drop"]["prune_events"] - compare["base"]["prune_events"],
        "spike_area_median_delta": compare["max_drop"]["spike_area_median"] - compare["base"]["spike_area_median"],
        "spike_peak_median_delta": compare["max_drop"]["spike_peak_median"] - compare["base"]["spike_peak_median"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}_compare.json").write_text(json.dumps(compare, indent=2))

    print("\n[compare]")
    print(json.dumps(compare["delta_metrics"], indent=2))
    print(f"Wrote: {out_dir / f'{tag}_compare.json'}")


if __name__ == "__main__":
    main()
