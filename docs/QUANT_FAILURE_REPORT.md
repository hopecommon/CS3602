# Quantization Failure Report

This report documents our quantization attempts, debugging steps, and final
conclusion. It is intended to be cited directly in the paper as a negative
result section.

---

## 1) Context and Goal

- Hardware: NVIDIA A800 (Ampere)
- Model: Pythia-2.8B (GPT-NeoX)
- Method: StreamingLLM (N_SINK=4, WINDOW_SIZE≈2048)
- Bottleneck: decode already <20 ms/token; profiling shows MLP + residual/LN +
  framework overhead dominate. Attention/KV is already reduced by StreamingLLM.
- Goal: improve TPOT (Time Per Output Token) while keeping PPL stable.

We expected quantization to accelerate MLP (the bottleneck) without altering
attention/KV, so this was our main candidate after speculative decoding and
flash attention were deemed unsuitable for long-context streaming.

---

## 2) Attempt A: Full-Model INT8 (TorchAO)

### A.1 Settings
- Quantization: TorchAO INT8 weight-only (int8wo)
- Variants: int8wo and int8da (dynamic activation + weight)
- Eval: decode-loop PPL + per-token latency benchmarks on WikiText (4096 tokens)

### A.2 Observations
- int8wo: PPL became NaN (numerical failure).
- int8da: extremely slow; per-token latency jumped to ~130 ms/token.
- Both variants were slower than FP16/BF16 streaming.

Example numbers (from `results/quant_sweep`):
- Baseline/streaming (no quant): ~14.1 ms/token.
- int8wo: ~22.9 ms/token (slower).
- int8da: ~133 ms/token (much slower).

### A.3 Interim conclusion
Full-model INT8 is not suitable for batch=1 streaming decode on A800. The
dynamic activation path (int8da) adds heavy overhead; the weight-only path
(int8wo) is numerically unstable.

---

## 3) Attempt B: MLP-Only INT8 (v1 config)

### B.1 Rationale
Since profiling shows MLP dominates, we limited quantization to:
- `mlp.dense_h_to_4h`
- `mlp.dense_4h_to_h`
All other layers (attention, embeddings, LN, lm_head) were kept in FP16/BF16.

### B.2 Sanity check (v1 path)
We ran a forward sanity check (no PPL loop) and explicitly used fp32 loss:
- `logits finite: False`
- first non-finite layer: `gpt_neox.layers.2`
- `fp32 loss finite: False`

Conclusion: NaN originates in the quantized forward path, not in the loss.

---

## 4) Debugging Steps and Fixes

### D.1 TorchAO v1 vs v2
TorchAO warned that Int8WeightOnlyConfig v1 is deprecated. We tested v2:
- v1: logits non-finite at layer 2.
- v2: logits finite, fp32 loss finite.

Conclusion: the NaN was due to the deprecated v1 quantization path.

### D.2 INT4 path blocked
INT4 weight-only required `fbgemm-gpu-genai`, which is unavailable in our
current Python 3.12 environment (module `fbgemm_gpu.experimental.genai` missing
even after package install). Therefore int4wo could not be validated.

### D.3 torch.compile / CUDAGraph issues
When using `torch.compile(mode="reduce-overhead")`, cudagraphs caused runtime
errors in looped decoding. We disabled cudagraphs to stabilize runs; compile
did not provide noticeable speedups.

---

## 5) Attempt C: MLP-Only INT8 (v2) Full Evaluation

We re-ran the full experiment with IntxWeightOnlyConfig v2 (MLP-only).

Configuration:
- `window_size=1020`, `n_sink=4`
- Dtype: FP16

Results (from `results/quantization/mlp_only_int8.json`):
- Baseline (no quant, no streaming):
  - PPL: 9.53, latency: 14.39 ms/token
- Streaming (no quant):
  - PPL: 9.78, latency: 14.13 ms/token
- Streaming + MLP-only INT8 v2:
  - PPL: 9.79 (stable)
  - latency: 31.92 ms/token (≈2.3x slower)
  - runtime: 103.17 s vs 47.86 s for pure streaming
  - peak memory: ~10.1 GB vs ~6.3 GB for pure streaming

Conclusion: Even when numerically stable, MLP-only INT8 **significantly slows
down** batch=1 decode and increases memory. It provides no TPOT benefit.

---

## 6) Final Conclusion

1) Quantization is not a viable acceleration path for our setting:
   - Full-model INT8 is unstable or very slow.
   - MLP-only INT8 v2 is stable but still much slower and more memory-hungry.

2) In long-context streaming, attention is no longer the bottleneck; the decode
   loop becomes MLP-dominated with batch=1, which does **not** benefit from the
   current INT8 kernels in this software/hardware stack.

3) This is a strong negative result: StreamingLLM provides major gains, while
   additional quantization (even targeted at the bottleneck) does not improve
   TPOT and may degrade stability or memory usage.

---

## 7) Suggested Writeup Snippet (Paper)

"We evaluated INT8 quantization using TorchAO on Pythia-2.8B under long-context
StreamingLLM. Full-model INT8 was unstable (NaN) or extremely slow, and even a
targeted MLP-only INT8 configuration (v2) remained numerically stable but
degraded decode latency by ~2.3x and increased memory usage. This indicates that
batch-1 decode in a MLP-dominated streaming regime does not benefit from current
INT8 kernels on A800, making quantization an ineffective acceleration path for
our setting."
