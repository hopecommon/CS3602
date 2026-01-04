#!/usr/bin/env python3
"""
Minimal PG19 PPL sanity check.

This script compares perplexity on two *specific* cached PG19 sample JSON files
to verify whether a PPL shift is caused by switching the underlying book/text.

It uses the repo's decode-loop baseline evaluation:
  - sliding-window recomputation (no KV cache)
  - token-by-token cross-entropy
  - (n_sink + window_size) = max_cache_size
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path when invoked as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.eval_utils import compute_perplexity


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare PG19 perplexity across two cached sample files.")
    ap.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "EleutherAI/pythia-2.8b"))
    ap.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype. Loss is still computed in fp32 by default for stability.",
    )
    ap.add_argument("--max-eval-tokens", type=int, default=20000)
    ap.add_argument("--n-sink", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=2044)
    ap.add_argument("--file-a", type=Path, default=Path("data/pg19/long_context_20000.json"))
    ap.add_argument("--file-b", type=Path, default=Path("data/pg19/long_context_50000.json"))
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test: override --max-eval-tokens=2048 (faster, but not your full 20k eval).",
    )
    ap.add_argument("--no-fp32-loss", action="store_true", help="Compute loss in model dtype (not recommended).")
    return ap.parse_args()


def _dtype(dtype: str) -> torch.dtype:
    import torch
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]

def _maybe_load_dotenv(repo_root: Path, overwrite: bool = False) -> None:
    """
    Minimal `.env` loader for this repo (no external dependency).

    - Parses KEY=VALUE lines (ignores comments/blank lines).
    - Performs simple $VAR / ${VAR} expansion using current os.environ.
    - By default does NOT overwrite already-set env vars.
    """
    env_path = repo_root / ".env"
    if not env_path.exists():
        return

    import re

    def expand(val: str) -> str:
        def repl(m: re.Match[str]) -> str:
            key = m.group(1) or m.group(2) or ""
            return os.environ.get(key, "")

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
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        if (not overwrite) and (key in os.environ and os.environ.get(key) not in (None, "")):
            continue
        os.environ[key] = expand(val)


def _load_sample(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if not data:
            raise ValueError(f"Empty JSON list: {path}")
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected JSON type {type(data)} in {path}")
    return data


def _tokenize_truncate(tokenizer, text: str, max_eval_tokens: int, device: torch.device) -> torch.Tensor:
    import torch
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids[:, :max_eval_tokens]
    return input_ids.to(device)


def eval_file(
    model,
    tokenizer,
    device: torch.device,
    path: Path,
    max_eval_tokens: int,
    n_sink: int,
    window_size: int,
    fp32_loss: bool,
) -> float:
    sample = _load_sample(path)
    text = sample.get("text") or ""
    if not text:
        raise ValueError(f"Missing/empty 'text' in {path}")

    input_ids = _tokenize_truncate(tokenizer, text, max_eval_tokens=max_eval_tokens, device=device)
    max_cache_size = int(n_sink + window_size)

    stats = compute_perplexity(
        model=model,
        encoded_dataset=input_ids,
        device=device,
        max_length=max_cache_size,  # unused in decode-loop path
        stride=max(1, window_size // 2),  # unused in decode-loop path
        use_streaming=False,
        max_cache_size=max_cache_size,
        fp32_loss=fp32_loss,
    )
    return float(stats.perplexity)


def main() -> None:
    # Load repo .env first so HF_HOME/HF_DATASETS_CACHE/MODEL_NAME match your usual scripts.
    _maybe_load_dotenv(REPO_ROOT)

    # Avoid common OpenMP SHM failures in restricted environments.
    os.environ.setdefault("KMP_DISABLE_SHM", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Offline-friendly defaults (still allow override by user).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    args = parse_args()
    if args.quick:
        args.max_eval_tokens = 2048

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = _dtype(args.dtype)
    if device.type == "cpu" and torch_dtype != torch.float32:
        torch_dtype = torch.float32

    # Match the repo's usual behavior: rely on HF_*OFFLINE env vars rather than forcing
    # `local_files_only=True` (some transformers versions can mis-handle missing shards).
    local_files_only = os.environ.get("LOCAL_FILES_ONLY", "0") == "1"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()

    fp32_loss = not args.no_fp32_loss

    def describe(path: Path) -> str:
        s = _load_sample(path)
        title = (s.get("short_book_title") or s.get("title") or "").strip()
        target = s.get("target_tokens")
        avail = s.get("available_tokens")
        return f"{path}  title={title!r}  target_tokens={target}  available_tokens={avail}"

    print(f"device={device}  dtype={torch_dtype}  fp32_loss={fp32_loss}")
    print(f"max_eval_tokens={args.max_eval_tokens}  n_sink={args.n_sink}  window_size={args.window_size}  cap={args.n_sink + args.window_size}")
    print("")

    for label, path in (("A", args.file_a), ("B", args.file_b)):
        path = path if path.is_absolute() else (REPO_ROOT / path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        print(f"[{label}] {describe(path)}")
        ppl = eval_file(
            model=model,
            tokenizer=tokenizer,
            device=device,
            path=path,
            max_eval_tokens=args.max_eval_tokens,
            n_sink=args.n_sink,
            window_size=args.window_size,
            fp32_loss=fp32_loss,
        )
        print(f"[{label}] PPL = {ppl:.6f}")
        print("")


if __name__ == "__main__":
    main()
