# StreamingLLM 复现 - 技术设计与对比说明

本文件解释当前实现中关键设计点、与 MIT/kvpress 的对比路径、以及为何仍保留 rerotation hook 方案。

## 1. 设计概览
- **模型**：`EleutherAI/pythia-70m`（GPTNeoX 架构），使用 kvpress/.venv 提供的 PyTorch + Transformers 环境。
- **目标**：复现 StreamingLLM 的 attention sink + recent window 思路，并与 MIT `StartRecentKVCache`、kvpress `StreamingLLMPress` 进行对比。
- **度量**：以 decode-loop `compute_perplexity` 为最严格 baseline，chunked / kvpress official 为补充；所有比对都记录于 `results/comprehensive/*.json`。

## 2. 核心模块

### 2.1 StreamingKVCache
- 保持 `n_sink + window_size` 固定 KV cache。
- `compress()` 返回 sink + recent tokens 的 concatenated tensor，尽量避免多次 gather。
- 在 `StreamingLLMWrapper.update` 中复用该逻辑，保证 PyTorch hook 在每层都看到新 cache。

### 2.2 MIT-style cache（StartRecentKVCache）
- 新增 `streaming_llm/mit_cache.py`，复刻 MIT `StartRecentKVCache` 的 sink + recent slicing逻辑。
- `StreamingLLMWrapper` 支持通过 `cache` 参数注入该行为，运行时直接 sliced chunk 插入，而不是 `torch.gather` 多次聚合。
- 该模式用于 `--streaming-mode mit`，以便与 MIT 公开形式保持一致。

### 2.3 Rerotation & pos-shift
- `streaming_llm/model.py` 默认通过 `rerotate_keys` 恢复 RoPE；这在我们 wrapping 过程中依然保持精度。
- 曾尝试 `streaming_llm/pos_shift/modify_gpt_neox.py`，但由于 Transformers 版本与 MIT 参考存在接口差异（`rotate_half`/`pos_emb` 参数不同、`num_attention_heads` 无法访问），暂时未启用。README/Design 中记录该尝试作为折衷说明。

### 2.4 kvpress 方案
- `experiments/run_kvpress_streaming_decode.sh` 会使用 `KeyRerotationPress` + `StreamingLLMPress`，将 kvpress 列为 decode-loop 评估（与 ours 共享 `compute_perplexity`）。
- `eval_kvpress.py` 保留，用于生成 kvpress 官方 chunked baseline + streaming 结果（路径在 `results/kvpress/`）。

## 3. 实验架构

### 3.1 评估流程
1. **Chunked**：`eval_streaming_llm.py` 分段 evaluate + sliding window baseline。
2. **Decode loop**：`run_decode_perplexity.py` 使用 `compute_perplexity` 进行逐 token evaluation，所有方法共享 script。
3. **kvpress official**：`eval_kvpress.py`（baseline + kvpress）保留原始对比。

### 3.2 运行脚本组织
- 核心：`run_comprehensive_comparisons.sh` + `run_decode_perplexity.py`。
- kvpress decode-loop：`run_kvpress_streaming_decode.sh`（支持通过环境变量指定数据集）。
- Legacy：`run_everything.sh` / `run_decoding_latency.sh`（保留但标记 legacy）。

## 4. 未来优化方向
1. **Profiler 定位**：若仍需压缩 runtime，可用 `torch.profiler` 跟踪 `StreamingLLMWrapper.update` 中 rerotation 与 gather，继续找出瓶颈。
2. **MIT pos-shift 复刻**：若 transformers 升级/接口回退，可再启用 `streaming_llm/pos_shift` 中的 attention 导出逻辑，实现原生 RoPE 重新计算。
3. **kvpress 再对齐**：通过 decode-loop 量化 kvpress rerotation 与 ours 差别，探索是否可借鉴 `KeyRerotationPress` 中的 rerotation顺序简化我们的 wrapper。
