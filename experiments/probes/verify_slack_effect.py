#!/usr/bin/env python3
"""
Verify whether Slack (cache_slack = sigma) provides measurable benefits, and in what regime.

Key point:
  - With Max_Drop disabled (delta=0), Slack should be a no-op in our implementation
    (it should not affect pruning cadence or target length).
  - With Max_Drop enabled (delta>0), Slack can matter because it defines the hard cap H=C+sigma
    used by staged eviction.

To avoid confounds, for each sigma we run two configs with the same (S, window, K):
  - control: (sigma, delta=0)
  - staged:  (sigma, delta=delta_fix)

We use Auto-cap by default:
  window = CAP_TOTAL - sink - sigma - overlap - refresh_budget

Outputs per sigma:
  - *_control_summary.json / *_staged_summary.json
  - *_control_prune_events.json / *_staged_prune_events.json
  - *_compare.json (delta vs control)

And a single CSV summary for the whole sweep.

Example:
  python -m experiments.probes.verify_slack_effect \\
    --dataset-name pg19 --dataset-config pg19 \\
    --cap-total 2048 --n-sink 32 \\
    --compress-every 64 --delta 32 \\
    --sigma-list 0,8,16,32,64 \\
    --max-eval-tokens 10000 --max-decode-steps 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repo_root()))

from experiments.eval_utils import load_tokenized_dataset, tqdm_wrap  # noqa: E402
from streaming_llm import StreamingLLMWrapper  # noqa: E402


def _maybe_load_dotenv(repo_root: Path, overwrite: bool = False) -> None:
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


def _parse_int_list(s: str) -> list[int]:
    items: list[int] = []
    for part in s.replace(",", " ").split():
        if not part:
            continue
        items.append(int(part))
    return items


def _derive_window(cap_total: int, sink: int, sigma: int, overlap: int, refresh: int) -> int:
    window = cap_total - sink - sigma - overlap - refresh
    if window <= 0:
        raise ValueError(
            f"Invalid derived window_size={window} (cap_total={cap_total}, sink={sink}, sigma={sigma}, overlap={overlap}, refresh={refresh})"
        )
    return int(window)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    return float(values[min(max(idx, 0), len(values) - 1)])


def _run_config(
    *,
    args: argparse.Namespace,
    model,
    encoded: torch.Tensor,
    device: torch.device,
    window_size: int,
    sigma: int,
    delta: int,
    tag: str,
    out_dir: Path,
) -> dict[str, Any]:
    wrapper = StreamingLLMWrapper(
        model=model,
        n_sink=args.n_sink,
        window_size=window_size,
        overlap=args.overlap,
        refresh_budget=args.refresh_budget,
        refresh_policy=args.refresh_policy,
        compress_every=args.compress_every,
        cache_slack=sigma,
        max_drop=delta,
    )

    if encoded.device != device:
        encoded = encoded.to(device)

    seq_len = int(encoded.size(1))
    prefill_len = min(args.n_sink + window_size, seq_len)
    decode_total = max(0, seq_len - prefill_len)
    decode_steps = decode_total if args.max_decode_steps <= 0 else min(decode_total, int(args.max_decode_steps))

    prune_events: list[dict[str, Any]] = []
    nll_values: list[float] = []
    step_times: list[float] = []
    kv_lengths: list[int] = []
    total_nll = 0.0
    total_pred = 0

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
            loss = F.cross_entropy(flat_logits, flat_labels, reduction="sum")
            total_nll += float(loss.item())
            total_pred += int(flat_labels.numel())

        for pos in tqdm_wrap(
            range(prefill_len - 1, prefill_len - 1 + decode_steps),
            total=decode_steps,
            desc=f"Slack sigma={sigma} delta={delta}",
            unit="tok",
        ):
            current_input = encoded[:, pos : pos + 1]
            target = encoded[:, pos + 1]

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model(input_ids=current_input, past_key_values=past_key_values, use_cache=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            logits = outputs.logits[:, -1, :]
            if args.fp32_loss:
                logits = logits.float()
            nll = F.cross_entropy(logits, target, reduction="sum").item()
            total_nll += float(nll)
            total_pred += 1
            nll_values.append(float(nll))

            past_key_values = outputs.past_key_values
            wrapper.update(past_key_values)
            prune_info = wrapper.pop_last_prune()
            if prune_info is not None:
                prune_events.append(prune_info)

            kv_len = int(past_key_values.layers[0].keys.shape[2])
            kv_lengths.append(kv_len)
            step_times.append(float((t1 - t0) * 1000.0))

    baseline_nll = float(median(nll_values)) if nll_values else 0.0
    spike_window = max(1, int(args.spike_window))
    spikes: list[dict[str, Any]] = []
    for event in prune_events:
        start_idx = int(event["step"])
        if start_idx < 0:
            continue
        end_idx = min(len(nll_values), start_idx + spike_window)
        window_nll = nll_values[start_idx:end_idx]
        if not window_nll:
            continue
        peak = max(window_nll)
        area = sum(max(0.0, v - baseline_nll) for v in window_nll)
        width = sum(1 for v in window_nll if v > baseline_nll)
        spikes.append({"step": start_idx, "peak": float(peak), "area": float(area), "width": int(width)})

    drop_sizes = [int(e.get("dropped", 0)) for e in prune_events]
    ppl = float(torch.exp(torch.tensor(total_nll / max(total_pred, 1))).item())

    summary = {
        "tag": tag,
        "dataset": f"{args.dataset_name}:{args.dataset_config}",
        "cap_total": int(args.cap_total),
        "n_sink": int(args.n_sink),
        "window_size": int(window_size),
        "compress_every": int(args.compress_every),
        "cache_slack": int(sigma),
        "max_drop": int(delta),
        "fp32_loss": bool(args.fp32_loss),
        "total_tokens": int(seq_len),
        "prefill_len": int(prefill_len),
        "decode_steps": int(decode_steps),
        "ppl_estimate": ppl,
        "latency_ms_mean": float(sum(step_times) / len(step_times)) if step_times else 0.0,
        "latency_ms_p50": float(_percentile(step_times, 0.5)),
        "latency_ms_p90": float(_percentile(step_times, 0.9)),
        "kv_len_mean": float(sum(kv_lengths) / len(kv_lengths)) if kv_lengths else float(prefill_len),
        "kv_len_min": int(min(kv_lengths)) if kv_lengths else int(prefill_len),
        "kv_len_max": int(max(kv_lengths)) if kv_lengths else int(prefill_len),
        "prune_events": int(len(prune_events)),
        "drop_median": float(median(drop_sizes)) if drop_sizes else 0.0,
        "drop_p90": float(_percentile([float(x) for x in drop_sizes], 0.9)) if drop_sizes else 0.0,
        "spike_area_median": float(median([s["area"] for s in spikes])) if spikes else 0.0,
        "spike_peak_median": float(median([s["peak"] for s in spikes])) if spikes else 0.0,
        "spike_width_median": float(median([s["width"] for s in spikes])) if spikes else 0.0,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / f"{tag}_prune_events.json").write_text(json.dumps(prune_events, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Slack effects with variable controls.")
    parser.add_argument("--model-name", type=str, default=os.environ.get("MODEL_NAME", "EleutherAI/pythia-2.8b"))
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--dataset-name", type=str, default="pg19")
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-eval-tokens", type=int, default=20000)

    parser.add_argument("--cap-total", type=int, default=2048)
    parser.add_argument("--n-sink", type=int, default=32)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--refresh-budget", type=int, default=0)
    parser.add_argument("--refresh-policy", type=str, default="none", choices=["none", "uniform", "middle"])

    parser.add_argument("--compress-every", type=int, default=64)
    parser.add_argument("--delta", type=int, default=32, help="Max_Drop value for staged run (control uses delta=0).")
    parser.add_argument("--sigma-list", type=str, default="0,8,16,32,64")
    parser.add_argument("--spike-window", type=int, default=32)
    parser.add_argument("--max-decode-steps", type=int, default=2000)

    parser.add_argument("--fp32-loss", action="store_true", default=True)
    parser.add_argument("--no-fp32-loss", dest="fp32_loss", action="store_false")
    parser.add_argument("--load-dotenv", action="store_true", default=True)
    parser.add_argument("--no-dotenv", dest="load_dotenv", action="store_false")

    parser.add_argument("--output-dir", type=Path, default=Path("results/probes/verify_slack"))
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()
    if args.load_dotenv:
        _maybe_load_dotenv(repo_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = _dtype_from_arg(args.dtype)
    if device.type == "cpu" and torch_dtype != torch.float32:
        torch_dtype = torch.float32

    sigmas = _parse_int_list(args.sigma_list)
    if not sigmas:
        raise SystemExit("Empty --sigma-list")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag_root = args.tag.strip() or f"{args.dataset_name}_S{args.n_sink}_cap{args.cap_total}_K{args.compress_every}_delta{args.delta}"
    print(f"[verify_slack] tag={tag_root}")
    print(f"[verify_slack] sigmas={sigmas} fp32_loss={args.fp32_loss} max_decode_steps={args.max_decode_steps}")

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

    rows: list[dict[str, Any]] = []
    for sigma in sigmas:
        window = _derive_window(args.cap_total, args.n_sink, sigma, args.overlap, args.refresh_budget)
        base_tag = f"{tag_root}_sigma{sigma}_W{window}_control"
        staged_tag = f"{tag_root}_sigma{sigma}_W{window}_staged"

        base = _run_config(
            args=args,
            model=model,
            encoded=encoded,
            device=device,
            window_size=window,
            sigma=sigma,
            delta=0,
            tag=base_tag,
            out_dir=out_dir,
        )
        staged = _run_config(
            args=args,
            model=model,
            encoded=encoded,
            device=device,
            window_size=window,
            sigma=sigma,
            delta=int(args.delta),
            tag=staged_tag,
            out_dir=out_dir,
        )

        compare = {
            "tag": f"{tag_root}_sigma{sigma}_W{window}",
            "sigma": sigma,
            "window": window,
            "control": base,
            "staged": staged,
            "delta": {
                "ppl_delta": staged["ppl_estimate"] - base["ppl_estimate"],
                "lat_ms_mean_delta": staged["latency_ms_mean"] - base["latency_ms_mean"],
                "lat_ms_p90_delta": staged["latency_ms_p90"] - base["latency_ms_p90"],
                "prune_events_delta": staged["prune_events"] - base["prune_events"],
                "spike_area_med_delta": staged["spike_area_median"] - base["spike_area_median"],
                "spike_peak_med_delta": staged["spike_peak_median"] - base["spike_peak_median"],
                "spike_width_med_delta": staged["spike_width_median"] - base["spike_width_median"],
            },
        }
        (out_dir / f"{tag_root}_sigma{sigma}_W{window}_compare.json").write_text(
            json.dumps(compare, indent=2), encoding="utf-8"
        )
        rows.append(
            {
                "sigma": sigma,
                "window": window,
                "K": int(args.compress_every),
                "delta": int(args.delta),
                "control_ppl": base["ppl_estimate"],
                "staged_ppl": staged["ppl_estimate"],
                "control_lat_ms": base["latency_ms_mean"],
                "staged_lat_ms": staged["latency_ms_mean"],
                "control_prunes": base["prune_events"],
                "staged_prunes": staged["prune_events"],
                "control_area": base["spike_area_median"],
                "staged_area": staged["spike_area_median"],
                "control_peak": base["spike_peak_median"],
                "staged_peak": staged["spike_peak_median"],
                "ppl_delta": compare["delta"]["ppl_delta"],
                "lat_ms_delta": compare["delta"]["lat_ms_mean_delta"],
                "area_delta": compare["delta"]["spike_area_med_delta"],
                "peak_delta": compare["delta"]["spike_peak_med_delta"],
            }
        )

    summary_csv = out_dir / f"{tag_root}_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {summary_csv}")


if __name__ == "__main__":
    main()

