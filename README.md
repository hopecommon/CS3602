# CS3602 NLP 大作业 - StreamingLLM 实现与 Lazy Pruning

本仓库实现了基于 HuggingFace（Transformers Cache API）的 StreamingLLM（Start+Recent）推理加速评测，并在其上提出并验证了一个主要改进：**Lazy Pruning**（延迟触发 KV compaction 与 RoPE re-alignment，以摊销 cache-management 开销）。仓库同时保留了大量“失败/不稳定尝试”的证据链与分析，用于解释为什么在 batch-1、bounded-attention 的长上下文流式解码设定下，很多常见加速并不直接适用。

## TL;DR（当前推荐入口）
- 跑“论文/主结果”实验：`run_paper_experiments.sh`（输出 `results/paper_experiments/` 并生成 `NeurIPS/generated/*.tex`）
- 构建 PDF：`build_paper_pdf.sh`（输出 `NeurIPS/neurips_2025_compressed.pdf`）
- 关键日志：`docs/探索日志.md`

## 1. 项目说明

### 1.1 个人项目部分
本仓库由 Jifan Lin 创建并维护。个人项目阶段实现了一套基于 HuggingFace Transformers Cache API 的 StreamingLLM 工程化评测版本（不修改 attention forward），与 MIT 官方实现采用不同的技术路线，并在此基础上进行了系统性实验与误差分析。

### 1.2 团队项目部分

#### Jifan Lin 的主要贡献
- **Lazy Pruning 算法改进**：在原始 StreamingLLM（Start+Recent）基础上提出并实现了 Lazy Pruning 优化方法，通过延迟触发 KV compaction 与 RoPE re-alignment 来摊销 cache-management 开销。实验结果表明该方法显著提升了推理速度，同时 PPL 未出现显著恶化
- **系统性探索与消融实验**：
  - 尝试并评估了多种优化方向：量化、投机解码、算子融合、overlap、refresh、slack、max_drop、bridge 等
  - 虽然上述方法在当前场景（batch-1、bounded-attention 长上下文流式解码）下未能带来正收益，但通过实验分析识别了不可行的原因、系统瓶颈以及未来改进方向
  - 产出了详细的探索日志与证据链（见 `docs/探索日志.md`）
- **实验框架与结果整理**：负责构建完整的实验脚本体系（`run_paper_experiments.sh` 等）、大量消融实验执行，以及结果输出流程（`results/paper_experiments/`、`NeurIPS/` 生成物等）

#### 其他成员贡献
- **[成员姓名]**：负责 Flash Attention 相关方向的探索与其他工作（待补充详细内容）

## 2. 快速开始
详见 `QUICKSTART.md`（包含离线环境、`.env`、数据样本与推荐脚本）。

最小步骤：
```bash
source kvpress/.venv/bin/activate
cp .env.example .env
chmod +x run_paper_experiments.sh build_paper_pdf.sh
./run_paper_experiments.sh
./build_paper_pdf.sh
```

## 3. 复现实验（论文主结果）
### 3.1 Auto-cap 与数据固定（避免 PPL 波动）
PG19 的 PPL 对“选了哪一本书/哪个段落”非常敏感；因此我们默认固定到同一个长文本文件，并通过 `--max-eval-tokens` 控制评测长度。

推荐在 `.env` 固定（与当前默认脚本一致）：
- `PG19_SAMPLE_FILE=data/pg19/long_context_50000.json`
- `PG19_SAMPLE_LENGTH=50000`
- `WIKITEXT_SAMPLE_FILE=data/wikitext/long_context_4096.json`
- `WIKITEXT_SAMPLE_LENGTH=4096`

`run_paper_experiments.sh` 使用 **Auto-cap**：给定总预算 `CAP_TOTAL=2048`，根据 `sink/slack/overlap/refresh` 自动推导 `window_size`（避免“手动设置 window 导致 budget 不一致”的混杂）。

### 3.2 一键运行 / 跳过已有 / 强制重跑
```bash
chmod +x run_paper_experiments.sh
./run_paper_experiments.sh        # 默认跳过已有结果
./run_paper_experiments.sh -n     # 只打印命令（dry-run）
./run_paper_experiments.sh -f     # 强制重跑
```

产物：
- 结果 JSON：`results/paper_experiments/*.json`
- 更细粒度结果：`results/paper_experiments/*_runs/`
- 论文表格/图的生成物：`NeurIPS/generated/*.tex`

### 3.3 构建论文 PDF
```bash
chmod +x build_paper_pdf.sh
./build_paper_pdf.sh
```
输出：`NeurIPS/neurips_2025_compressed.pdf`

## 4. 结果与证据链入口
- 论文（主结论与附录证据）：`NeurIPS/neurips_2025_compressed.pdf`
- 探索日志（大量失败尝试与原因）：`docs/探索日志.md`
- 论文实验 JSON：`results/paper_experiments/`
- Probes（profiling/诊断）：`results/probes/`

## 5. 目录结构（核心部分）
```text
CS3602/
├── run_paper_experiments.sh          # 论文主实验脚本（Auto-cap + 生成 LaTeX）
├── build_paper_pdf.sh                # 构建 NeurIPS PDF（会生成 NeurIPS/generated）
├── NeurIPS/                          # 论文源码与生成 PDF
├── streaming_llm/                    # 我们的 StreamingLLM 实现（HF Cache API + RoPE re-align）
├── experiments/                      # 评测脚本与 paper 生成脚本
├── results/                          # paper_experiments / probes / baselines / legacy 等
└── docs/                             # 探索日志与额外报告
```

## 6. Legacy（历史对照）
仓库仍保留早期脚本与结果（例如 `run_fixed_evaluation.sh`、`results/fixed_eval/`、`run_comprehensive_comparisons.sh`），用于对比/回溯；但当前主线请以 `run_paper_experiments.sh` + `NeurIPS/` 为准。

