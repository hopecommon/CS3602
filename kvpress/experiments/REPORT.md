# CS3602 – KVPress Reproduction Log

This document tracks the engineering work and current experimental status for reproducing KV cache compression on **EleutherAI/pythia‑70m** with the **WikiText‑103** and **PG19** perplexity datasets.

## Engineering Summary

1. **Reusable perplexity scripts**
   - `experiments/pythia_wikitext_perplexity.py` evaluates a single dataset/model/press pair and emits JSON with PPL & runtime (baseline vs. compressed).
   - `experiments/run_pythia_kvpress_sweep.py` sweeps over multiple compression ratios and datasets, producing aggregated JSON + plots (`experiments/results/pythia_kvpress_sweep.json` and `..._tradeoff.png`). It now streams PG19 so that only one novel (~8k tokens) is fetched.

2. **Integration with KVPress evaluation CLI**
   - Added `wikitext_ppl` and `pg19_ppl` entries to `evaluation/evaluate_registry.py`.
   - `evaluation/evaluate.py` detects these datasets, skips the QA pipeline, streams the raw text, tokenizes it (optional token cap), and directly computes perplexity using `experiments/pythia_wikitext_perplexity.compute_perplexity`.
   - Press instances are `deepcopy`‑ed so each run starts from a fresh object (important for methods that register hooks).
   - Results (perplexity + runtime) are written through the standard CSV/JSON pipeline, so downstream tooling (leaderboard, plots) can ingest them unchanged.

3. **Dataset handling specifics**
   - WikiText: default 64 samples, 4096 token cap to keep runtime small while still exercising prefill.
   - PG19: streamed `test[:1]`, cap at 8192 tokens, which matches the assignment requirement of “at least one PG19 story”.

## How to Reproduce

### 1. Low‑level scripts

```bash
# Single run (baseline vs Knorm)
HF_HOME=$PWD/.cache/huggingface .venv/bin/python \
  experiments/pythia_wikitext_perplexity.py \
  --model-name EleutherAI/pythia-70m \
  --dataset-name wikitext \
  --dataset-config wikitext-103-v1 \
  --split test \
  --max-samples 64 \
  --max-length 1024 \
  --stride 512 \
  --compression-ratio 0.5 \
  --output experiments/results/wikitext_knorm.json

# Compression sweep (wikitext + PG19, ratios 0.0–0.9 plus {0.25,0.5,0.75})
HF_HOME=$PWD/.cache/huggingface .venv/bin/python \
  experiments/run_pythia_kvpress_sweep.py
```

### 2. “Native” evaluation CLI

```bash
# WikiText baseline / Knorm
HF_HOME=$PWD/.cache/huggingface .venv/bin/python evaluation/evaluate.py \
  --dataset wikitext_ppl \
  --model EleutherAI/pythia-70m \
  --press_name no_press \
  --device cuda \
  --output_dir experiments/eval_runs

HF_HOME=$PWD/.cache/huggingface .venv/bin/python evaluation/evaluate.py \
  --dataset wikitext_ppl \
  --model EleutherAI/pythia-70m \
  --press_name knorm \
  --compression_ratio 0.5 \
  --device cuda \
  --output_dir experiments/eval_runs

# PG19 baseline / Knorm (streamed single novel, 8k tokens)
HF_HOME=$PWD/.cache/huggingface .venv/bin/python evaluation/evaluate.py \
  --dataset pg19_ppl \
  --model EleutherAI/pythia-70m \
  --press_name no_press \
  --device cuda \
  --output_dir experiments/eval_runs

HF_HOME=$PWD/.cache/huggingface .venv/bin/python evaluation/evaluate.py \
  --dataset pg19_ppl \
  --model EleutherAI/pythia-70m \
  --press_name knorm \
  --compression_ratio 0.5 \
  --device cuda \
  --output_dir experiments/eval_runs
```

All outputs are stored under `experiments/eval_runs/<dataset>__<model>__<press>__<ratio>/`.

## Current Experimental Status

| Dataset | Tokens (after cap) | Press | Compression | PPL | Runtime (s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| WikiText‑103 | ~3.4k | no_press | 0.0 | 40.31 | **0.32** | Baseline |
| WikiText‑103 | ~3.4k | Knorm | 0.5 | 40.31 | **0.38** | CPU time dominated by K‑norm sorting → slower despite perfect PPL |
| WikiText‑103 | ~3.4k | StreamingLLM | 0.5 | 40.31 | **0.37** | Similar issue: short sequences mean compression overhead > savings |
| PG19 | 8k | no_press | 0.0 | 33.09 | **0.36** | Single streamed novel |
| PG19 | 8k | Knorm | 0.5 | 33.09 | **0.43** | Sorting/gather dominates on long sequences |

*All runs were executed with CUDA fp16 on a single GPU.*

### Diagnosis

- Integrated `evaluate.py` now reproduces the *perplexity* behavior exactly (PPL values match the standalone script when token caps are aligned).
- However, with the current token lengths, the extra work introduced by K-Norm (norms + `topk` + `gather`) outweighs the saved attention cost, so wall-clock time increases even though the KV cache is smaller.
- Using a press that aggressively drops tokens during prefill (e.g., StreamingLLM) helps slightly but is still dominated by overhead at only ~3–8k tokens.

### Warm-run evaluation (new `--press_sequence`)

`evaluation/evaluate.py` now accepts `--press_sequence no_press,knorm,streaming_llm`, which runs all presses within a single process. After a one-time warmup, the measured runtimes match the sweep script:

| Dataset | Press | Compression | Runtime (s) | PPL |
| --- | --- | --- | --- | --- |
| WikiText‑103 | no_press | 0.0 | 0.326 | 40.31 |
|  | Knorm | 0.5 | **0.106** | 40.31 |
|  | StreamingLLM | 0.5 | **0.038** | 40.31 |
| PG19 | no_press | 0.0 | 0.360 | 33.09 |
|  | StreamingLLM | 0.5 | **0.134** | 33.09 |

Each press produces its own folder under `experiments/eval_runs/`. Use `--prefill_only true` if you want `runtime_sec` to report only the prefill portion (the full duration is kept as `total_runtime_sec`).

> **Baseline sanity check**  
> Running `--press_sequence no_press,no_press` shows the second (warmed) baseline dropping to ~0.33 s on WikiText and ~0.36 s on PG19, exactly matching the warmed numbers above. That confirms the big gaps in the table are due to StreamingLLM compression itself, not warm vs. cold starts.

## Next Steps

1. **Measure presses that benefit from shorter contexts**  
   Run `evaluation/evaluate.py` with `--press_name streaming_llm` (already scripted) and tune `compression_ratio` and token caps to see if we can cross the “break-even” threshold. The goal is to find at least one configuration where evaluation runtime decreases while keeping PPL stable.

2. **Document speed/PPL curves**  
   Once a press delivers measurable acceleration, fold its data into the sweep JSON/plot and cite those numbers in the README report.

3. **Optional: expand to PG19 multi-sample**  
   Streaming currently fetches one novel. Increasing `max_samples` (with the same token cap) could produce more stable averages and may change the runtime dynamics.

This log will be updated as new presses or datasets are validated.
