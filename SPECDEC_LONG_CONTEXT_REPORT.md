# Speculative Decoding: Long-Context Exploration Report

This document records the long-context SpecDec exploration, including workflows,
key measurements, and analysis of why SpecDec underperforms in the 20k context
workload. It is intended to be a traceable artifact for the final write-up.

## Goals
- Validate StreamingLLM behavior in long context (PPL, runtime).
- Measure SpecDec speed/accept rates in long context.
- Determine whether low accept rates are due to implementation bugs or
  distribution mismatch.

## Workloads
1. **PPL decode-loop** (long context, PG19 20k)
   - Script: `run_long_context_spec_check.sh`
   - Baseline vs Streaming only (SpecDec not involved in PPL).

2. **Speed / accept rate** (long context, PG19 20k)
   - Script: `run_long_context_spec_check.sh` -> `run_speculative_comparisons.sh`
   - Baseline vs Streaming, Normal vs SpecDec.

3. **Sanity checks**
   - Script: `run_specdec_sanity_checks.sh`
   - draft=target (implementation check)
   - short context (2k) check
   - draft_k sweep (2/4/8) for 20k context

4. **Draft sweep**
   - Script: `run_specdec_draft_sweep.sh`
   - draft=160m/410m, temperature=0.0/0.7 (long context 20k)

## Key Results

### PPL decode-loop (PG19 20k)
From `results/long_context_spec/pg19_baseline.json` and
`results/long_context_spec/pg19_streaming.json`:

| Method | PPL | Runtime (s) |
| --- | --- | --- |
| baseline | 8.8245 | 1475.26 |
| streaming | 8.9947 | 272.44 |

Observation: PPL increases slightly (+0.17) while runtime drops ~5.4x.

### Speed / accept rate (PG19 20k)
From `results/long_context_spec/speed/*.json`:

| Method | Mean latency (ms/token) | Accept rate | Tokens/target forward |
| --- | --- | --- | --- |
| base normal | 37.7 | - | - |
| base spec (70m) | 145.2 | 0.004 | 0.516 |
| stream normal | 13.4 | - | - |
| stream spec (70m) | 30.7 | 0.134 | 1.034 |

Observation: SpecDec is slower than normal in base mode and only marginally
better with streaming; accept rates remain low.

### Sanity checks
From `results/spec_sanity/*`:

1. **draft=target (20k)**:
   - Accept ~0.99, tokens/target ~2.47
   - Confirms SpecDec alignment/accept logic works.

2. **short context (2k)**:
   - Accept ~0.17, tokens/target ~0.84
   - Even at 2k, 70m draft is weak.

3. **draft_k sweep (20k)**:
   - base spec accept stays <2%
   - streaming spec accept improves (13% -> 36%) but still below the level
     needed for speedup.

### Draft sweep (160m/410m, 20k)
From `results/spec_sanity/draft_sweep/*`:

| Draft | Temp | Base accept | Stream accept | Stream tokens/target |
| --- | --- | --- | --- | --- |
| 160m | 0.0 | 0.062 | 0.106 | 0.712 |
| 160m | 0.7 | 0.050 | 0.093 | 0.686 |
| 410m | 0.0 | 0.049 | 0.094 | 0.688 |
| 410m | 0.7 | 0.036 | 0.109 | 0.717 |

Observation: Larger draft does not substantially improve accept rates or
tokens/target; SpecDec remains in the "cost > benefit" regime.

## Analysis: Why SpecDec Underperforms in Long Context
1. **Distribution mismatch at long positions**:
   - 20k context is far beyond common training lengths, causing large
     divergences between draft and target distributions.
2. **Streaming helps but is insufficient**:
   - Streaming increases accept rates by shrinking effective context length,
     but still fails to reach tokens/target > 1.5.
3. **Implementation is likely correct**:
   - draft=target sanity check produces near-perfect acceptance.
4. **Larger draft does not fix the mismatch**:
   - 160m/410m show no clear improvement, suggesting mismatch is not due to
     capacity alone in this workload.

## Conclusions (usable in paper)
- StreamingLLM is effective for long context: major speedup with minimal PPL
  degradation.
- SpecDec on long-context PG19 with small/medium draft fails to accelerate;
  this is a negative result that is explainable by distribution mismatch.

## Artifact Index
- PPL results:
  - `results/long_context_spec/pg19_baseline.json`
  - `results/long_context_spec/pg19_streaming.json`
- Speed / accept:
  - `results/long_context_spec/speed/*.json`
- Sanity checks:
  - `results/spec_sanity/*`
- Draft sweep:
  - `results/spec_sanity/draft_sweep/*`

