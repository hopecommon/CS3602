# Lazy/Periodic KV Pruning Report

## Goal
Reduce StreamingLLM overhead by pruning KV cache **periodically** instead of every step:

- Allow KV to grow temporarily.
- Prune once every `R` tokens (`compress_every=R`).
- Hypothesis: reduce copy-heavy pruning cost and improve TPOT without harming PPL.

This does **not** change attention kernels or RoPE logic; it only changes the pruning schedule.

## Setup
- Model: EleutherAI/pythia-2.8b (A800)
- Streaming config: `window=2048`, `n_sink=4`
- Datasets:
  - Wikitext (4096 tokens)
  - PG19 (20k tokens)
- Sweep: `compress_every ∈ {4, 16, 32}`

## Results (Wikitext)

| compress_every | TPOT (ms) | Speedup | PPL Δ |
|---:|---:|---:|---:|
| 4  | 15.75 | 6.57x | +3.07% |
| 16 | 14.67 | 7.05x | +2.89% |
| 32 | 14.39 | 7.18x | +3.13% |

**Takeaway:** `compress_every=16/32` reduces TPOT by ~7–9% with nearly unchanged PPL.

## Results (PG19)

| compress_every | TPOT (ms) | Speedup | PPL Δ |
|---:|---:|---:|---:|
| 4  | 15.76 | 6.62x | +5.45% |
| 16 | 14.72 | 7.09x | +5.52% |
| 32 | 14.42 | 7.24x | +6.24% |

**Takeaway:** `compress_every=16` gives a clean speedup with minimal PPL impact.
`compress_every=32` is fastest but PPL increases more.

## Conclusion
Lazy/periodic pruning is **effective** and aligns with the observation that
window size alone does not improve speed on short texts. By reducing pruning
frequency, we recover meaningful TPOT gains without destabilizing perplexity.

Recommended default:
- **Short text (wikitext):** `compress_every=32` (fastest, PPL stable)
- **Long text (PG19):** `compress_every=16` (balanced)
