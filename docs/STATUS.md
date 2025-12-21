# Project Status and Next Steps

This note summarizes where we are and what is still viable given the current
constraints (A800, Pythia-2.8b, StreamingLLM, long-context focus).

---

## 1) Current Findings

- Profiling: decode is already <20 ms/token; bottleneck is MLP + residual/LN + framework overhead.
- Attention/KV: already optimized by StreamingLLM (window + sink), so further attention/KV tweaks give limited gains.
- Long context is where StreamingLLM shines; large speedup mostly comes from long sequences.

---

## 2) What We Tried (and Why It Failed)

- Flash attention / flash decoding:
  - Conflicts with streaming changes (RoPE rotation, cache handling).
  - Integration cost high; benefit limited because attention is no longer the bottleneck.
- Speculative decoding:
  - Unreliable for long-context streaming; quality/robustness not acceptable.
- Quantization:
  - TorchAO INT8 (int8wo/int8da) slowed decode substantially in single-token decode.
  - INT4 path requires fbgemm-gpu-genai (not available in our current Python/venv),
    so int4wo is currently blocked.
  - PPL instability observed in int8wo (NaN).

---

## 3) What Still Looks Feasible (Given Current Constraints)

These are the few options left that do not conflict with StreamingLLM and still
align with the course requirements.

### A) Reduce Framework Overhead (Low Risk)
- Use torch.compile without cudagraphs to reduce Python overhead.
- Keep shapes stable (window + sink fixed) to maximize compile benefit.
- Expect small gains; still worth reporting as an ablation.

### B) Mixed Precision / Partial Quantization (Medium Risk)
- Instead of full-model INT8, try MLP-only quantization (the real bottleneck):
  - Quantize only MLP linear layers, keep attention + embeddings in BF16/FP16.
  - This can be a "paper contribution": targeted quantization on the true bottleneck.
- If int4 remains unavailable, try int8 weight-only on MLP only.

### C) RefreshLite / Smaller Window (Medium Risk)
- If PPL is stable at window=2048, attempt window reduction (1024, 512).
- Use RefreshLite to stabilize PPL and plot a TPOT-PPL trade-off curve.
- The gain may be modest but provides solid analysis and a Pareto curve.

### D) Stronger Evaluation Framing (Low Risk, High Value)
- Keep StreamingLLM as the main result.
- Show that other "obvious accelerations" (flash/speculative/quant) fail or provide
  negative gains in long-context streaming.
- This matches the course emphasis on careful analysis, not just raw speed.

---

## 4) Recommended Minimum Deliverables (Meets Teacher Requirements)

- Main result: StreamingLLM at long contexts (PPL increase <= ~3-4%, large speedup).
- Ablation: window size sweep (or refresh-lite) + stable PPL/TPOT trade-off curve.
- Negative results: flash/speculative/quantization documented with reasons.
- Optional: compile vs no-compile as a small positive gain or null effect.

---

## 5) Practical Next Steps

1) Consolidate stable StreamingLLM results (baseline vs streaming).
2) Run a window-size sweep (2048 -> 1024 -> 512) and track PPL + TPOT.
3) If feasible, add RefreshLite to recover PPL at smaller windows.
4) If we can stabilize int8 MLP-only quantization, include as a targeted attempt.

---

## 6) What We Will NOT Emphasize

- FP8 / KV-cache quantization on A800 (hardware and/or kernel path not suitable).
- Full-model quantization for single-token decode (currently slower).
- Speculative decoding for long contexts (instability and mismatch with streaming).
