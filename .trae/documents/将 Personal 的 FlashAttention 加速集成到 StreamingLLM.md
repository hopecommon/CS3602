## 目标
- 将 Personal 的 FlashAttention 适配器无侵入整合到当前 StreamingLLM，实现 `flash/on` 与 `math/off` 两种内核模式。
- 把新模式纳入现有评测与可视化管线，生成与固定评测一致的 JSON 与图表，便于直观比较速度与 PPL。

## 集成 FlashAttention（不改数学逻辑）
- 适配器模式：在模型加载处为注意力层包裹 `NeoXFlashAttentionAdapter`，仅通过 `sdp_kernel` 切换 SDPA 内核（参考 Personal/src/models/pythia_flash_model.py:11, 80）。
- 开关与 dtype：
  - 运行时开关：`FLASH_SDPA=1` 打开 flash；或通过代码参数 `attn_backend=(auto|flash|math)`。
  - 自动选择 dtype（bf16 优先，fallback fp16/fp32），复用 `check_sdpa_flash_available`（Personal/src/utils.py:22）。
- 保持 StreamingLLM 的 KV 裁剪与 RoPE纠偏不变（streaming_llm/model.py 与 rope_utils.py）。

## 纳入统一性能衡量
### 1) 评测 Runner 增强
- 在 `experiments/run_final_experiments.py` 的 `ExperimentRunner`：
  - 增加参数 `attn_backend_list`（如 `["math","flash"]`）。
  - 主实验、消融实验均循环 `backend in attn_backend_list`：
    - 设置环境或传参启用对应内核。
    - 以同一配置跑 baseline 与 streaming；产出包含 `backend` 字段的结果。
  - 输出文件命名：`*_main_backend-math.json`、`*_main_backend-flash.json`；消融同理。
  - 结果 JSON 增加：`{"backend": "flash" | "math"}`，其余结构保持现有键（`baseline`/`streaming`、`perplexity`、`runtime_sec`、`prefill_sec`、`speedup` 等）。

### 2) 评测函数对齐
- 复用现有 `compute_perplexity` 与统计字段：
  - `perplexity`（PPL），`runtime_sec`（总解码时间），`prefill_sec`（首 token前），可增加可选 `decode_toks_per_sec`（平均吞吐）。
- 通过统一的入口（`eval_streaming_llm.py` 或 Runner 内部）控制 `attn_backend`，避免分散脚本。

### 3) 可视化脚本支持新后端
- `experiments/plot_fixed_eval_results.py` 与综合图：
  - 扩展数据加载逻辑，识别 `backend` 字段或文件名后缀，将 `math` 与 `flash` 两套曲线并列绘制。
  - 统一图例：`StreamingLLM (math)`、`StreamingLLM (flash)`；维持原有 `baseline` 与 `kvpress` 款式。
- `experiments/plot_comprehensive_results.py` 与 decode-loop 对比：同样按后端拆分子系列或上色。

## 指标与比较维度
- 度量保持一致：`PPL`、`runtime_sec`、`speedup`、`prefill_sec`；可选 `decode_toks_per_sec`。
- 维度：数据集（WikiText-103 / PG19）、后端（math/flash）、方法（baseline / StreamingLLM / KVPress）。

## 验证步骤
- 正确性：对比 `math` 与 `flash` 的 logits（允许 1e-3 量级浮动），PPL 一致或轻微差异。
- 性能：确认 flash 在 GPU/bf16 条件下显著缩短 `runtime_sec`，同时 `prefill_sec` 与 TTFT 统计正常。
- 可视化：固定评测与综合图均能展示 `math`/`flash` 两套结果，图例清晰、文件命名一致。

## 风险与兼容
- Transformers 版本统一到 ≥4.56，规避已知 FA 问题；kvpress 也建议该版本。
- 不同模型族的注意力模块识别：先支持 Pythia/GPT‑NeoX；如需 Llama/Mistral，再补充适配器识别表。
- dtype/连续性：保证裁剪后 `.contiguous()` 与 bf16/fp16，避免隐式类型转换影响性能。

## 产出与可交付
- 新的评测结果文件：每个数据集/方法×后端各一份 JSON（带 `backend` 字段）。
- 图表：在现有图表基础上，新增并列系列展示 `math` 与 `flash` 的差异。
- 脚本：Runner 与绘图脚本对 `attn_backend` 的统一支持，不改变现有使用方式（默认 `math`，可开关 `flash`）。