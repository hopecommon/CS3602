# StreamingLLM 复现 - 技术设计与对比说明

本文件解释当前实现中关键设计点、与 MIT/kvpress 的对比路径、以及为何仍保留 rerotation hook 方案。

## 1. 设计概览
- **模型**：`EleutherAI/pythia-2.8b`（GPTNeoX 架构），使用 kvpress/.venv 提供的 PyTorch + Transformers 环境
- **目标**：复现 StreamingLLM 的 attention sink + recent window 思路，并与 MIT `StartRecentKVCache` 进行对比
- **度量**：以 decode-loop `compute_perplexity` 为主要评估方法，结果保存在 `results/fixed_eval/` 和 `results/comprehensive/`
- **数据集**：WikiText-103 (4k tokens) 和 PG19 (20k tokens)

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

**1. Decode-loop (主评估方法)**:
- **脚本**：`run_decode_perplexity.py` 调用 `eval_utils.compute_perplexity`
- **Baseline实现**：Sliding window + 重算
  * 每个decode步骤重新forward最近1028个token
  * 使用`use_cache=False`，不复用KV cache
  * 复杂度：O(window_size) per token
  * 对齐StreamingLLM论文的baseline定义
- **StreamingLLM实现**：
  * 使用KV cache，每步只forward 1个新token
  * 通过`StreamingLLMWrapper.update()`压缩cache到1028 tokens
  * 复杂度：O(1) per token (cache大小固定)
- **公平性保证**：两种方法解码相同的token序列，只有cache策略不同
- **指标**：JSON结果包含`runtime_sec`、`prefill_sec`、`perplexity`、`first_token_latency_sec`、`peak_memory_mb`，方便全面对比

**2. Chunked (Legacy方法)**:
- **脚本**：`eval_streaming_llm.py`
- 分段evaluate，每段独立计算PPL
- 已被decode-loop取代，仅保留用于历史对比

**3. KVPress官方 (对照方法)**:
- **脚本**：`eval_kvpress.py`
- 使用kvpress的chunked evaluation pipeline
- 评估方法与decode-loop不同，结果不可直接比较
- 仅作为kvpress官方实现的参考数据

### 3.2 运行脚本组织
- **主入口**：`run_fixed_evaluation.sh` —— 快速复现主要结果（decode-loop评估）
- **全量对比**：`run_comprehensive_comparisons.sh` —— 完整的chunked + decode-loop评估
- **单独评估**：`run_decode_perplexity.py` —— 灵活的单方法评估工具
- **可视化**：`plot_fixed_eval_results.py` / `plot_comprehensive_results.py` —— 图表生成

## 4. 未来优化方向
1. **Profiler 定位**：若仍需压缩 runtime，可用 `torch.profiler` 跟踪 `StreamingLLMWrapper.update` 中 rerotation 与 gather，继续找出瓶颈。
2. **MIT pos-shift 复刻**：若 transformers 升级/接口回退，可再启用 `streaming_llm/pos_shift` 中的 attention 导出逻辑，实现原生 RoPE 重新计算。
3. **kvpress 再对齐**：通过 decode-loop 量化 kvpress rerotation 与 ours 差别，探索是否可借鉴 `KeyRerotationPress` 中的 rerotation顺序简化我们的 wrapper。
