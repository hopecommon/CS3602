#!/usr/bin/env python3
"""
Check whether an existing baseline JSON is compatible with the current environment
and expected eval configuration.

Exit code:
  0: compatible
  1: incompatible / missing required fields
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Ensure repo root is on sys.path when invoked as a script (e.g., `python experiments/paper/check_baseline_compat.py`).
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.paper.env_info import collect_env_info, env_compatible


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument("--dtype", type=str, required=True)
    ap.add_argument("--max-length", type=int, required=True)
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--max-eval-tokens", type=int, required=True)
    ap.add_argument("--n-sink", type=int, required=True)
    ap.add_argument("--window-size", type=int, required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not args.baseline.exists():
        return 1
    try:
        data = json.loads(args.baseline.read_text(encoding="utf-8"))
    except Exception:
        return 1

    # Env fingerprint
    env_now = collect_env_info(repo_root=str(Path(__file__).resolve().parents[2])).to_dict()
    env_prev = data.get("env")
    if not env_compatible(env_prev, env_now):
        return 1

    # Config fingerprint (minimal but strict on the key fairness knobs)
    if data.get("model") != args.model_name:
        return 1
    if str(data.get("dtype")) != str(args.dtype):
        return 1
    if int(data.get("max_length", -1)) != int(args.max_length):
        return 1
    if int(data.get("stride", -1)) != int(args.stride):
        return 1
    if int(data.get("max_eval_tokens", -1)) != int(args.max_eval_tokens):
        return 1

    sllm = data.get("streaming_llm") or {}
    if int(sllm.get("n_sink", -1)) != int(args.n_sink):
        return 1
    if int(sllm.get("window_size", -1)) != int(args.window_size):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
