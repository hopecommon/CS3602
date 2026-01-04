#!/usr/bin/env python3
"""
Probe: measure pruning overhead for StreamingLLM under different `compress_every` (K).

Goal
----
Answer whether a soft-ring implementation is worth doing by quantifying:
  - how much time is spent in KV pruning / relocation (wrapper.update)
  - how pruning frequency changes with K
  - whether TPOT saturates for K>=4/16/32 (i.e., pruning is no longer dominant)

We separate time into:
  - forward_ms: model forward (single-token decode)
  - update_ms: StreamingLLMWrapper.update() including pruning + RoPE re-alignment
  - total_ms: forward_ms + update_ms

We also report stats for prune steps vs non-prune steps.

Run from repo root:
  python -m experiments.probes.profile_prune_overhead --dataset-name pg19 --cap-total 2048 --n-sink 32 --cache-slack 16 --k-list 1,4,16,32,64 --max-decode-steps 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
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


def _derive_window(cap_total: int, sink: int, slack: int, overlap: int, refresh: int) -> int:
    window = cap_total - sink - slack - overlap - refresh
    if window <= 0:
        raise ValueError(
            f"Invalid auto-cap window_size={window} (cap_total={cap_total}, sink={sink}, slack={slack}, overlap={overlap}, refresh={refresh})"
        )
    return int(window)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((len(values) - 1) * q))
    return float(values[min(max(idx, 0), len(values) - 1)])


@dataclass(frozen=True)
class RunSummary:
    dataset: str
    cap_total: int
    n_sink: int
    window_size: int
    cache_slack: int
    max_drop: int
    compress_every: int
    fp32_loss: bool
    total_tokens: int
    prefill_len: int
    decode_steps: int
    prune_events: int
    drop_median: float
    drop_p90: float
    total_ms_mean: float
    total_ms_p50: float
    total_ms_p90: float
    forward_ms_mean: float
    update_ms_mean: float
    update_ms_fraction_mean: float
    prune_step_update_ms_mean: float
    nonprune_step_update_ms_mean: float
    ppl_estimate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile pruning overhead vs compress_every (K).")
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
    parser.add_argument("--cache-slack", type=int, default=16)
    parser.add_argument("--max-drop", type=int, default=0)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--refresh-budget", type=int, default=0)
    parser.add_argument("--refresh-policy", type=str, default="none", choices=["none", "uniform", "middle"])

    parser.add_argument("--k-list", type=str, default="1,4,16,32,64", help="Comma/space separated K values.")
    parser.add_argument("--max-decode-steps", type=int, default=2000, help="Decode steps after prefill (0 = all).")

    parser.add_argument("--fp32-loss", action="store_true", default=True)
    parser.add_argument("--no-fp32-loss", dest="fp32_loss", action="store_false")
    parser.add_argument("--load-dotenv", action="store_true", default=True)
    parser.add_argument("--no-dotenv", dest="load_dotenv", action="store_false")

    parser.add_argument("--output-dir", type=Path, default=Path("results/probes/profile_prune_overhead"))
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def _run_for_k(
    *,
    args: argparse.Namespace,
    model,
    encoded: torch.Tensor,
    device: torch.device,
    window_size: int,
    k: int,
    output_dir: Path,
    tag: str,
) -> RunSummary:
    wrapper = StreamingLLMWrapper(
        model=model,
        n_sink=args.n_sink,
        window_size=window_size,
        overlap=args.overlap,
        refresh_budget=args.refresh_budget,
        refresh_policy=args.refresh_policy,
        compress_every=k,
        cache_slack=args.cache_slack,
        max_drop=args.max_drop,
    )

    if encoded.device != device:
        encoded = encoded.to(device)

    seq_len = int(encoded.size(1))
    prefill_len = min(args.n_sink + window_size, seq_len)
    decode_total = max(0, seq_len - prefill_len)
    decode_steps = decode_total if args.max_decode_steps <= 0 else min(decode_total, int(args.max_decode_steps))

    forward_ms: list[float] = []
    update_ms: list[float] = []
    total_ms: list[float] = []
    update_frac: list[float] = []
    prune_update_ms: list[float] = []
    nonprune_update_ms: list[float] = []
    prune_events: list[dict[str, Any]] = []

    total_nll = 0.0
    total_pred = 0

    per_step_path = output_dir / f"{tag}_K{k}_per_step.csv"
    with per_step_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "token_index",
                "forward_ms",
                "update_ms",
                "total_ms",
                "kv_len",
                "pruned",
                "dropped",
                "overflow",
            ],
        )
        writer.writeheader()

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
                nll = F.cross_entropy(flat_logits, flat_labels, reduction="sum")
                total_nll += float(nll.item())
                total_pred += int(flat_labels.numel())

            for step_idx, pos in enumerate(
                tqdm_wrap(
                    range(prefill_len - 1, prefill_len - 1 + decode_steps),
                    total=decode_steps,
                    desc=f"Profile K={k}",
                    unit="tok",
                )
            ):
                current_input = encoded[:, pos : pos + 1]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    outputs = model(input_ids=current_input, past_key_values=past_key_values, use_cache=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                past_key_values = outputs.past_key_values
                t2 = time.perf_counter()
                wrapper.update(past_key_values)
                t3 = time.perf_counter()

                prune_info = wrapper.pop_last_prune()
                if prune_info is not None:
                    prune_events.append(prune_info)

                kv_len = int(past_key_values.layers[0].keys.shape[2])
                f_ms = (t1 - t0) * 1000.0
                u_ms = (t3 - t2) * 1000.0
                tot_ms = f_ms + u_ms

                forward_ms.append(float(f_ms))
                update_ms.append(float(u_ms))
                total_ms.append(float(tot_ms))
                update_frac.append(float(u_ms / max(tot_ms, 1e-9)))

                pruned = 1 if prune_info else 0
                if pruned:
                    prune_update_ms.append(float(u_ms))
                else:
                    nonprune_update_ms.append(float(u_ms))

                writer.writerow(
                    {
                        "step": step_idx,
                        "token_index": pos + 1,
                        "forward_ms": float(f_ms),
                        "update_ms": float(u_ms),
                        "total_ms": float(tot_ms),
                        "kv_len": kv_len,
                        "pruned": pruned,
                        "dropped": int(prune_info["dropped"]) if prune_info else 0,
                        "overflow": int(prune_info["overflow"]) if prune_info else 0,
                    }
                )

    drop_sizes = [int(e.get("dropped", 0)) for e in prune_events]
    ppl = float(torch.exp(torch.tensor(total_nll / max(total_pred, 1))).item())
    summary = RunSummary(
        dataset=f"{args.dataset_name}:{args.dataset_config}",
        cap_total=int(args.cap_total),
        n_sink=int(args.n_sink),
        window_size=int(window_size),
        cache_slack=int(args.cache_slack),
        max_drop=int(args.max_drop),
        compress_every=int(k),
        fp32_loss=bool(args.fp32_loss),
        total_tokens=int(seq_len),
        prefill_len=int(prefill_len),
        decode_steps=int(decode_steps),
        prune_events=int(len(prune_events)),
        drop_median=float(median(drop_sizes)) if drop_sizes else 0.0,
        drop_p90=float(_percentile([float(x) for x in drop_sizes], 0.9)) if drop_sizes else 0.0,
        total_ms_mean=float(mean(total_ms)) if total_ms else 0.0,
        total_ms_p50=float(_percentile(total_ms, 0.5)),
        total_ms_p90=float(_percentile(total_ms, 0.9)),
        forward_ms_mean=float(mean(forward_ms)) if forward_ms else 0.0,
        update_ms_mean=float(mean(update_ms)) if update_ms else 0.0,
        update_ms_fraction_mean=float(mean(update_frac)) if update_frac else 0.0,
        prune_step_update_ms_mean=float(mean(prune_update_ms)) if prune_update_ms else 0.0,
        nonprune_step_update_ms_mean=float(mean(nonprune_update_ms)) if nonprune_update_ms else 0.0,
        ppl_estimate=ppl,
    )

    (output_dir / f"{tag}_K{k}_prune_events.json").write_text(json.dumps(prune_events, indent=2))
    (output_dir / f"{tag}_K{k}_summary.json").write_text(json.dumps(asdict(summary), indent=2))
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

    ks = _parse_int_list(args.k_list)
    if not ks:
        raise ValueError("Empty --k-list")

    window_size = _derive_window(
        int(args.cap_total),
        int(args.n_sink),
        int(args.cache_slack),
        int(args.overlap),
        int(args.refresh_budget),
    )

    tag = args.tag.strip() or f"{args.dataset_name}_S{args.n_sink}_cap{args.cap_total}_sigma{args.cache_slack}_delta{args.max_drop}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[profile] tag={tag}")
    print(f"[profile] auto-cap => window_size={window_size} (cap={args.cap_total} sink={args.n_sink} slack={args.cache_slack} overlap={args.overlap} refresh={args.refresh_budget})")
    print(f"[profile] ks={ks} fp32_loss={args.fp32_loss} max_decode_steps={args.max_decode_steps}")

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

    summaries: list[RunSummary] = []
    for k in ks:
        summaries.append(
            _run_for_k(
                args=args,
                model=model,
                encoded=encoded,
                device=device,
                window_size=window_size,
                k=int(k),
                output_dir=out_dir,
                tag=tag,
            )
        )

    # Aggregate into a single CSV for easy decision-making.
    summary_csv = out_dir / f"{tag}_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for s in summaries:
            writer.writerow(asdict(s))

    print(f"Wrote: {summary_csv}")


if __name__ == "__main__":
    main()

