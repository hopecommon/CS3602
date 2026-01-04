#!/usr/bin/env python3
"""
Targeted probe to verify whether `cache_slack` (Slack) and `max_drop` (Max_Drop)
have measurable effects under parameter regimes where they are expected to matter.

Key idea:
- If `max_drop >= compress_every`, Max_Drop tends to be clipped away (no staged eviction).
- Slack only matters when Max_Drop is active and would otherwise be clipped to soft capacity.

This script runs a small grid (sigma × delta × compress_every) and reports:
- PPL (decode-style, token-level)
- TPOT / runtime
- Prune event statistics (count, drop sizes, overflow)

It avoids recomputing a baseline: this is a relative probe across settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

# Ensure repo root is on sys.path when invoked as a script.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.eval_utils import load_tokenized_dataset
from streaming_llm.model import StreamingLLMWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class RunConfig:
    dataset_name: str
    dataset_config: str
    split: str
    max_eval_tokens: int
    model_name: str
    dtype: str
    device: str
    n_sink: int
    window_size: int
    compress_every: int
    cache_slack: int
    max_drop: int
    loss_modes: tuple[str, ...]

    @property
    def cap(self) -> int:
        return int(self.n_sink + self.window_size)

    def tag(self) -> str:
        return (
            f"{self.dataset_name}_K{self.compress_every}"
            f"_sigma{self.cache_slack}_delta{self.max_drop}"
            f"_S{self.n_sink}_W{self.window_size}"
        )


def _parse_int_list(s: str) -> list[int]:
    items: list[int] = []
    for part in s.replace(",", " ").split():
        if not part:
            continue
        items.append(int(part))
    return items


def _maybe_load_dotenv(repo_root: Path, overwrite: bool = False) -> None:
    """
    Minimal `.env` loader for this repo (no external dependency).

    - Only parses KEY=VALUE lines (ignores comments/blank lines).
    - Performs simple $VAR / ${VAR} expansion using values already in os.environ.
    - By default does NOT overwrite existing environment variables.
    """
    env_path = repo_root / ".env"
    if not env_path.exists():
        return

    import re

    def expand(val: str) -> str:
        def repl(m: re.Match[str]) -> str:
            key = m.group(1) or m.group(2) or ""
            return os.environ.get(key, "")

        # $VAR or ${VAR}
        return re.sub(r"\$(?:{([^}]+)}|([A-Za-z_][A-Za-z0-9_]*))", repl, val)

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        # Strip optional quotes.
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        if (not overwrite) and (key in os.environ and os.environ.get(key) not in (None, "")):
            continue
        os.environ[key] = expand(val)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    mid = len(vs) // 2
    if len(vs) % 2 == 1:
        return float(vs[mid])
    return float((vs[mid - 1] + vs[mid]) / 2.0)


def _summarize_prunes(events: list[dict[str, Any]]) -> dict[str, Any]:
    dropped = [int(e.get("dropped", 0)) for e in events]
    overflow = [int(e.get("overflow", 0)) for e in events]
    kept = [int(e.get("kept", 0)) for e in events]
    seq_len = [int(e.get("seq_len", 0)) for e in events]

    def mean_int(xs: list[int]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    return {
        "num_events": len(events),
        "dropped_mean": mean_int(dropped),
        "dropped_median": _median([float(x) for x in dropped]),
        "dropped_max": max(dropped) if dropped else 0,
        "overflow_mean": mean_int(overflow),
        "overflow_median": _median([float(x) for x in overflow]),
        "overflow_max": max(overflow) if overflow else 0,
        "kept_mean": mean_int(kept),
        "kept_max": max(kept) if kept else 0,
        "seq_len_at_prune_mean": mean_int(seq_len),
        "seq_len_at_prune_max": max(seq_len) if seq_len else 0,
    }


@dataclass(frozen=True)
class DecodeStats:
    perplexity_fp16loss: float
    perplexity_fp32loss: float
    runtime_sec: float
    prefill_sec: float
    first_token_latency_sec: float
    decode_tokens: int
    decode_time_sec: float
    tpot_ms: float
    peak_memory_mb: float


def _compute_streaming_decode_with_events(
    model,
    encoded_dataset: torch.Tensor,
    device: torch.device,
    wrapper: StreamingLLMWrapper,
    loss_modes: tuple[str, ...],
) -> tuple[DecodeStats, list[dict[str, Any]]]:
    if encoded_dataset.device != device:
        encoded_dataset = encoded_dataset.to(device)

    seq_len = encoded_dataset.size(1)
    if seq_len < 2:
        raise ValueError("Dataset is too short (seq_len < 2).")

    # The decode loop should tolerate the wrapper's hard cap (soft + slack).
    max_cache_size = max(2, min(wrapper.n_sink + wrapper.window_size + wrapper.cache_slack, seq_len))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    use_cuda_timing = device.type == "cuda" and torch.cuda.is_available()
    if use_cuda_timing:
        start_evt = torch.cuda.Event(enable_timing=True)
        prefill_end_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        first_start_evt = torch.cuda.Event(enable_timing=True)
        first_end_evt = torch.cuda.Event(enable_timing=True)
    else:
        total_start = time.perf_counter()
        prefill_start = time.perf_counter()

    total_nll_fp16 = 0.0
    total_nll_fp32 = 0.0
    total_tokens = 0
    decode_tokens = 0
    first_token_time = 0.0

    # Manual context management so we can read prune events before reset().
    wrapper.__enter__()
    past_key_values = None
    prefill_len = min(max_cache_size, seq_len)

    input_ids = encoded_dataset[:, :prefill_len]
    if use_cuda_timing:
        start_evt.record()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    if labels.numel() > 0:
        if "fp16" in loss_modes:
            loss_fp16 = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="sum",
            )
            total_nll_fp16 += float(loss_fp16.item())
        if "fp32" in loss_modes:
            loss_fp32 = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="sum",
            )
            total_nll_fp32 += float(loss_fp32.item())
        total_tokens += int(labels.numel())

    past_key_values = outputs.past_key_values
    wrapper.update(past_key_values)
    if use_cuda_timing:
        prefill_end_evt.record()
        first_token_recorded = False
    else:
        prefill_time = time.perf_counter() - prefill_start
        first_token_recorded = False

    decode_steps = max(0, seq_len - prefill_len)
    for pos in range(prefill_len - 1, seq_len - 1):
        current_input = encoded_dataset[:, pos : pos + 1]
        target = encoded_dataset[:, pos + 1]

        if use_cuda_timing and not first_token_recorded:
            first_start_evt.record()
        with torch.no_grad():
            outputs = model(input_ids=current_input, past_key_values=past_key_values, use_cache=True)
        if use_cuda_timing and not first_token_recorded:
            first_end_evt.record()

        logits = outputs.logits[:, -1, :]
        if "fp16" in loss_modes:
            loss_fp16 = F.cross_entropy(logits, target, reduction="sum")
            total_nll_fp16 += float(loss_fp16.item())
        if "fp32" in loss_modes:
            loss_fp32 = F.cross_entropy(logits.float(), target, reduction="sum")
            total_nll_fp32 += float(loss_fp32.item())
        total_tokens += int(target.numel())

        past_key_values = outputs.past_key_values
        wrapper.update(past_key_values)
        if not first_token_recorded:
            first_token_recorded = True
        decode_tokens += 1

    if use_cuda_timing:
        end_evt.record()
        torch.cuda.synchronize()
        total_time = start_evt.elapsed_time(end_evt) / 1000.0
        prefill_time = start_evt.elapsed_time(prefill_end_evt) / 1000.0
        if total_tokens > 0:
            first_token_time = first_start_evt.elapsed_time(first_end_evt) / 1000.0
    else:
        total_time = time.perf_counter() - total_start

    decode_time = max(0.0, float(total_time - prefill_time))
    denom = max(total_tokens, 1)
    ppl_fp16 = float(torch.exp(torch.tensor(total_nll_fp16 / denom)).item()) if "fp16" in loss_modes else 0.0
    ppl_fp32 = float(torch.exp(torch.tensor(total_nll_fp32 / denom)).item()) if "fp32" in loss_modes else 0.0
    peak_mb = 0.0
    if torch.cuda.is_available():
        peak_mb = float(torch.cuda.max_memory_allocated() / 1024 / 1024)

    events = wrapper.get_prune_events()
    wrapper.__exit__(None, None, None)

    stats = DecodeStats(
        perplexity_fp16loss=ppl_fp16,
        perplexity_fp32loss=ppl_fp32,
        runtime_sec=float(total_time),
        prefill_sec=float(prefill_time),
        first_token_latency_sec=float(first_token_time),
        decode_tokens=int(decode_tokens),
        decode_time_sec=float(decode_time),
        tpot_ms=float(decode_time / max(decode_tokens, 1) * 1000.0),
        peak_memory_mb=peak_mb,
    )
    return stats, events


def _iter_grid(
    compress_values: Iterable[int],
    sigma_values: Iterable[int],
    delta_values: Iterable[int],
) -> Iterable[tuple[int, int, int]]:
    for k in compress_values:
        for sigma in sigma_values:
            for delta in delta_values:
                yield int(k), int(sigma), int(delta)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, default="EleutherAI/pythia-2.8b")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dataset-name", type=str, default="pg19")
    ap.add_argument("--dataset-config", type=str, default="pg19")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max-eval-tokens", type=int, default=20000)

    ap.add_argument("--n-sink", type=int, default=32)
    ap.add_argument("--window-size", type=int, default=2016)

    ap.add_argument("--compress-values", type=str, default="64 128")
    ap.add_argument("--sigma-values", type=str, default="0 64")
    ap.add_argument("--delta-values", type=str, default="0 32")

    ap.add_argument(
        "--loss-modes",
        type=str,
        default="fp16 fp32",
        help="Which loss precisions to compute: any subset of 'fp16 fp32'.",
    )

    # Convenience: derive (K, sigma, delta) triplets by dividing a base tuple.
    ap.add_argument("--base-compress", type=int, default=None, help="Base compress_every (K) for divisor sweep.")
    ap.add_argument("--base-sigma", type=int, default=None, help="Base cache_slack (sigma) for divisor sweep.")
    ap.add_argument("--base-delta", type=int, default=None, help="Base max_drop (delta) for divisor sweep.")
    ap.add_argument(
        "--divisors",
        type=str,
        default="",
        help="Optional divisors (e.g., '1 2 4') to derive K/sigma/delta from base-*. Overrides *-values lists.",
    )

    ap.add_argument("--output-dir", type=Path, default=Path("results/probes/slack_maxdrop"))
    ap.add_argument("--skip-existing", action="store_true", help="Skip configs with existing JSON outputs.")
    ap.add_argument(
        "--rewrite-summary",
        action="store_true",
        help="Always (re)generate summary.csv from all JSONs in output-dir.",
    )
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Only (re)generate summary.csv from existing JSONs; do not load model/dataset.",
    )
    ap.add_argument(
        "--load-dotenv",
        action="store_true",
        default=True,
        help="Load repo `.env` (if present) to align dataset selectors across scripts.",
    )
    ap.add_argument(
        "--no-load-dotenv",
        action="store_false",
        dest="load_dotenv",
        help="Do not load `.env`.",
    )
    ap.add_argument(
        "--include-controls",
        action="store_true",
        default=True,
        help="When using --divisors, also include same-K controls (sigma/delta = 0) for controlled comparisons.",
    )
    ap.add_argument(
        "--no-include-controls",
        action="store_false",
        dest="include_controls",
        help="Disable automatic same-K controls when using --divisors.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    if args.load_dotenv:
        _maybe_load_dotenv(repo_root)

    if args.summary_only:
        _write_summary(args.output_dir, rows=[], rewrite=True)
        return 0

    device = torch.device(args.device)
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    loss_modes = tuple([m.strip() for m in args.loss_modes.replace(",", " ").split() if m.strip()])
    for m in loss_modes:
        if m not in {"fp16", "fp32"}:
            raise SystemExit(f"Unknown loss mode: {m} (expected fp16/fp32)")
    if not loss_modes:
        raise SystemExit("Empty --loss-modes")

    # Load dataset (honors PG19_SAMPLE_FILE / WIKITEXT_SAMPLE_FILE if set).
    # Offline-friendly: avoid network calls. (The repo's `run_paper_experiments.sh` sets these too.)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded_dataset = load_tokenized_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column="text",
        max_samples=1,  # typically pinned to a single presampled segment; keep runtime deterministic
        max_eval_tokens=args.max_eval_tokens,
        tokenizer=tokenizer,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        local_files_only=True,
    ).to(device)
    model.eval()

    if args.divisors.strip():
        if args.base_compress is None or args.base_sigma is None or args.base_delta is None:
            raise SystemExit("--divisors requires --base-compress/--base-sigma/--base-delta")
        divisors = _parse_int_list(args.divisors)
        triplets: list[tuple[int, int, int]] = []
        for div in divisors:
            if div <= 0:
                continue
            k = max(1, int(args.base_compress // div))
            sigma = max(0, int(args.base_sigma // div))
            delta = max(0, int(args.base_delta // div))
            triplets.append((k, sigma, delta))
        # De-dup while preserving order.
        seen = set()
        grid = []
        for t in triplets:
            if t in seen:
                continue
            seen.add(t)
            grid.append(t)

        if args.include_controls:
            # Add controlled comparisons at the same K to isolate sigma/delta effects:
            # - (K, 0, 0): no Slack, no Max_Drop
            # - (K, sigma, 0): Slack only
            # - (K, 0, delta): Max_Drop only (may be clipped if sigma=0; that's informative)
            expanded: list[tuple[int, int, int]] = []
            for k, sigma, delta in grid:
                expanded.extend([(k, 0, 0), (k, sigma, 0), (k, 0, delta), (k, sigma, delta)])
            seen = set()
            grid2: list[tuple[int, int, int]] = []
            for t in expanded:
                if t in seen:
                    continue
                seen.add(t)
                grid2.append(t)
            grid = grid2
    else:
        compress_values = _parse_int_list(args.compress_values)
        sigma_values = _parse_int_list(args.sigma_values)
        delta_values = _parse_int_list(args.delta_values)
        grid = list(_iter_grid(compress_values, sigma_values, delta_values))

    rows: list[dict[str, Any]] = []
    for compress_every, cache_slack, max_drop in grid:
        cfg = RunConfig(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            max_eval_tokens=args.max_eval_tokens,
            model_name=args.model_name,
            dtype=args.dtype,
            device=str(device),
            n_sink=args.n_sink,
            window_size=args.window_size,
            compress_every=compress_every,
            cache_slack=cache_slack,
            max_drop=max_drop,
            loss_modes=loss_modes,
        )
        out_path = args.output_dir / f"{cfg.tag()}.json"
        if args.skip_existing and out_path.exists():
            continue

        wrapper = StreamingLLMWrapper(
            model=model,
            n_sink=cfg.n_sink,
            window_size=cfg.window_size,
            compress_every=cfg.compress_every,
            cache_slack=cfg.cache_slack,
            max_drop=cfg.max_drop,
            overlap=0,
            refresh_budget=0,
            refresh_policy="none",
        )
        stats, events = _compute_streaming_decode_with_events(
            model=model,
            encoded_dataset=encoded_dataset,
            device=device,
            wrapper=wrapper,
            loss_modes=cfg.loss_modes,
        )
        prune_summary = _summarize_prunes(events)

        payload: dict[str, Any] = {
            "config": asdict(cfg),
            "decode": asdict(stats),
            "prune_summary": prune_summary,
            "prune_events": events,
            "dataset_selector": {
                "PG19_SAMPLE_FILE": os.environ.get("PG19_SAMPLE_FILE"),
                "PG19_SAMPLE_LENGTH": os.environ.get("PG19_SAMPLE_LENGTH"),
                "WIKITEXT_SAMPLE_FILE": os.environ.get("WIKITEXT_SAMPLE_FILE"),
                "WIKITEXT_SAMPLE_LENGTH": os.environ.get("WIKITEXT_SAMPLE_LENGTH"),
            },
            "notes": {
                "expected_behavior": (
                    "Slack and Max_Drop typically matter when max_drop < compress_every "
                    "and cache_slack > 0, enabling staged evictions instead of hard drops."
                ),
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote: {out_path}")

        rows.append(_row_from_payload(payload))

    _write_summary(args.output_dir, rows, rewrite=args.rewrite_summary)

    return 0


def _row_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = payload.get("config", {})
    decode = payload.get("decode", {})
    prune = payload.get("prune_summary", {})
    tag = payload.get("tag") or (
        f"{cfg.get('dataset_name','ds')}_K{cfg.get('compress_every')}_"
        f"sigma{cfg.get('cache_slack')}_delta{cfg.get('max_drop')}_"
        f"S{cfg.get('n_sink')}_W{cfg.get('window_size')}"
    )
    decode_tokens = float(decode.get("decode_tokens", 0) or 0)
    decode_time = float(decode.get("decode_time_sec", 0.0) or 0.0)
    speed_tok_s = (decode_tokens / decode_time) if decode_time > 0 else 0.0
    cap = int(cfg.get("n_sink", 0) or 0) + int(cfg.get("window_size", 0) or 0)

    # Backward-compatible fallback for older JSONs (before we recorded both fp16/fp32 loss modes).
    ppl_fp16 = decode.get("perplexity_fp16loss")
    ppl_fp32 = decode.get("perplexity_fp32loss")
    if ppl_fp16 is None and ppl_fp32 is None and decode.get("perplexity") is not None:
        # Historically our probes computed PPL using fp32 loss by default.
        ppl_fp32 = decode.get("perplexity")
        ppl_fp16 = 0.0

    row: dict[str, Any] = {
        "tag": tag,
        "K": int(cfg.get("compress_every", 0) or 0),
        "sigma": int(cfg.get("cache_slack", 0) or 0),
        "delta": int(cfg.get("max_drop", 0) or 0),
        "cap": cap,
        "ppl_fp16loss": float(ppl_fp16 or 0.0),
        "ppl_fp32loss": float(ppl_fp32 or 0.0),
        "tpot_ms": float(decode.get("tpot_ms", 0.0) or 0.0),
        "speed_tok_s": float(speed_tok_s),
    }
    for k, v in prune.items():
        row[f"prune_{k}"] = v
    return row


def _write_summary(output_dir: Path, rows: list[dict[str, Any]], rewrite: bool) -> None:
    csv_path = output_dir / "summary.csv"
    if (not rows) or rewrite:
        # If nothing new was run (e.g., --skip-existing), rebuild rows from disk.
        disk_rows: list[dict[str, Any]] = []
        for p in sorted(output_dir.glob("*.json")):
            try:
                payload = json.loads(p.read_text())
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            disk_rows.append(_row_from_payload(payload))
        if disk_rows:
            rows = disk_rows

    if not rows:
        return

    # Stable column order.
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
