# NeurIPS 论文实验复现指南

本文档说明如何 **一键跑完论文所需实验**，并自动生成论文使用的 LaTeX 表格（不手填数字，避免出错）。

核心原则：
- `NeurIPS/neurips_2025_compressed.tex` **不写死实验数字**；默认显示占位符，避免误把旧结果写进 PDF。
- 所有数字来自 `results/**.json`，由脚本自动生成表格。
- Baseline 默认 **只跑一次并固定复用**（避免每次 sweep 重跑 baseline；PG19 很慢）。
- 默认会复用已有实验点结果（避免误触发长时间重跑）；每个结果文件都带 `config_hash` 用于识别配置。

---

## 1. 文件结构

```
CS3602/
├── NeurIPS/
│   ├── neurips_2025_compressed.tex     # 4页正文版（表格自动生成）
│   ├── generated/tables.tex            # 自动生成：主结果表
│   ├── generated/ablations.tex         # 自动生成：Lazy Pruning 消融表（其余为可选探索）
│   ├── generated/sweeps.tex            # 自动生成：R/σ/δ 扫描表（可作补充材料）
│   ├── generated/negative_results.tex  # 自动生成：定性负结果表
│   └── references.bib
├── experiments/
│   ├── eval_streaming_llm.py
│   ├── run_fixed_baseline.py           # 生成固定 baseline（只需一次）
│   └── paper/
│       ├── generate_tables_tex.py
│       ├── generate_ablations_tex.py
│       ├── generate_sweeps_tex.py
│       └── generate_negative_results_tex.py
├── run_paper_experiments.sh            # 一键：主结果+消融+扫描+生成tex
└── (no quick script)                   # 只维护一个入口，避免口径漂移
```

---

## 2. 环境准备（推荐）

可选 `.env`（脚本会自动读取）：

```bash
PYTHON_BIN=kvpress/.venv/bin/python
MODEL_NAME=EleutherAI/pythia-2.8b

HF_HOME=$PWD/.cache/huggingface
HF_DATASETS_CACHE=$HF_HOME/datasets
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

# (Recommended) Pin presampled evaluation segments for reproducibility
# (these are consumed by experiments/eval_utils.py)
WIKITEXT_SAMPLE_LENGTH=4096
PG19_SAMPLE_LENGTH=50000
PG19_SAMPLE_FILE=data/pg19/long_context_50000.json
```

---

## 3. 一次性生成固定 Baseline（只需做一次）

`run_paper_experiments.sh` 默认 **不重跑 baseline**，而是复用：
- `results/baselines/wikitext_baseline_avg.json`
- `results/baselines/pg19_baseline_avg.json`

生成方式（默认跑 1 次；如需更稳可手动改成 3/5 次）：

```bash
kvpress/.venv/bin/python experiments/run_fixed_baseline.py --runs 1
```

产物会写到：
- `results/baselines/wikitext_baseline_avg.json`
- `results/baselines/pg19_baseline_avg.json`

说明：
- `run_paper_experiments.sh` 会优先复用上述 baseline；若缺失且 `AUTO_BASELINE=1`（默认），会自动触发生成。
- `run_paper_experiments.sh -f` 会强制重新生成 baseline（并重跑所有实验点）。
- Baseline 本质是 **sliding-window recomputation (no KV cache)**，只依赖总 cap（固定 2048）；脚本中保留 `sink/window` 仅用于统一记录/算出 cap。
  为避免误解，baseline 默认固定使用 `BASELINE_SINK=4, BASELINE_WINDOW=2044`（cap 仍是 2048），即使主实验使用 `S=32`。

---

## 4. 一键跑完论文全部实验并生成 TeX

```bash
# 推荐在仓库根目录执行（脚本也会自动切到仓库根目录）
./run_paper_experiments.sh
```

默认会跑：
- 主结果：Baseline(复用) + Start+Recent(严格裁剪) + Ours(Lazy Pruning)（PG19 + WikiText）
- 消融：以 Lazy Pruning 为主（PG19）；Slack/Max\_Drop 默认视为可选探索（Appendix/负结果）
- 扫描：默认只扫 `R`（PG19）；`σ/δ` 仅在开启探索项时跑
- 生成 `NeurIPS/generated/*.tex`（主表、消融表、扫描表、负结果表）

---

## 5. Auto-cap（非常重要）

本仓库的最终评测统一使用 **auto-cap** 设计：固定总 KV 预算（默认 `CAP_TOTAL=2048`），并由脚本自动计算
`window_size = CAP_TOTAL - n_sink - cache_slack - overlap - refresh_budget`。

这样做的目的：
- 避免手动调 window 导致口径漂移；
- 在总预算恒定的前提下，让 `cache_slack/max_drop` 的作用变得可测量（否则很容易被 window 的主效应掩盖）。

## 6. 控制开关（避免太慢）

重复运行（统计稳定性）：
- 默认是“先跑通 pipeline”：`WARMUP_RUNS=0`、`REPEAT_RUNS=1`（最快）
- 需要最终表格数字时再开启重复跑（均值/方差），例如：

```bash
WARMUP_RUNS=1 REPEAT_RUNS=3 ./run_paper_experiments.sh
```

只跑主结果（跳过扫参和消融）：

```bash
RUN_SWEEPS=0 RUN_ABLATIONS=0 ./run_paper_experiments.sh
```

自定义 sweep 网格（默认值见脚本）：

```bash
SWEEP_R_VALUES="1 16 32 64" \
SWEEP_SIGMA_VALUES="0 16 32" \
SWEEP_DELTA_VALUES="0 16 32" \
./run_paper_experiments.sh
```

若要开启质量相关探索项（会更慢；会启用 Slack/Max\_Drop 相关消融与 `σ/δ` 扫描）：

```bash
RUN_QUALITY_HEURISTICS=1 ./run_paper_experiments.sh
```

强制重跑（忽略已有结果，覆盖生成的 JSON/TeX）：

```bash
./run_paper_experiments.sh -f
```

默认行为（不加 `-f`）会尽量复用已有 JSON，避免误触发长时间重跑。
`run_paper_experiments.sh` 默认开启 `ALLOW_STALE_RESULTS=1`：即使参数变了也会复用旧的聚合结果（避免意外重跑）。
如需严格模式（参数变了就自动重跑），可设置：

```bash
ALLOW_STALE_RESULTS=0 ./run_paper_experiments.sh
```

Baseline 指纹检查（跨机器/环境复用）：
- 默认 `STRICT_BASELINE_CHECK=0`：发现 torch/cuda/GPU/配置不一致会提示警告，但不会中断运行（避免在他人服务器上卡住）。
- 如需严格模式：`STRICT_BASELINE_CHECK=1 ./run_paper_experiments.sh`，不一致会自动重跑 baseline 以避免 speedup 漂移。

PG19 采样说明（避免“PPL 看起来不对”的困惑）：
- 默认建议固定使用 `data/pg19/long_context_50000.json`（由 `.env`/环境变量 `PG19_SAMPLE_FILE` 指定），评测长度再用 `--max-eval-tokens=20000` 截断到 20k；因此 `--max-samples` 对 PG19 基本无效（始终是单段长文本）。
- 这保证跨机器可复现，但 PPL 数值不一定与“全 PG19 测试集平均”一致；论文中以相对变化与趋势为主。
- 默认 PPL 使用 `--fp32-loss`（fp32 计算 cross-entropy），避免 fp16 下 softmax/logsumexp 的数值误差导致 PPL 偏离（甚至“异常偏低”）。

MIT 官方代码路径对照（可选）：
- `run_paper_experiments.sh` 支持额外跑一组 “MIT reference” 指标（直接调用 `mit-streaming-llm/examples/eval_long_ppl.py` 并用其 `StartRecentKVCache` 裁剪逻辑），输出到：
  - `results/mit_bench/pg19_mit_reference.json`
  - `results/mit_bench/wikitext_mit_reference.json`
  默认关闭：`RUN_MIT_REFERENCE=0`。注意：MIT repo 的 pos-shift patch 与我们默认 Transformers 版本对 GPTNeoX/Pythia 不兼容，可能导致崩溃或 PPL 异常；除非你确认在当前环境可用，否则不要把这些数字写进主表。

（已移除）本仓库自实现的 `--streaming-mode mit` 对照：
- 该路径会引入“实现差异/缓存格式差异/计时口径差异”等混杂，无法作为 MIT 官方结果的可靠对照。
- 论文主表/消融默认只使用我们统一实现的 Start+Recent vs Lazy Pruning，MIT official 仅作为可选 sanity/benchmark（如果环境能跑通）。

MIT reference 的 window-matched 对照（可选）：
- 为了隔离 “recent window 变小” 这个混杂因素，脚本会额外跑一条 MIT reference：保持 `start_size` 相同，但把 `recent_size` 强制设为 Ours 的 recent window（即 slack 从 window 中挪走后的大小）。
- 输出到：
  - `results/mit_bench/pg19_mit_reference_windowmatch.json`
  - `results/mit_bench/wikitext_mit_reference_windowmatch.json`

MIT benchmark（速度/显存，对照用）：
- `run_paper_experiments.sh` 也会调用 `mit-streaming-llm/examples/benchmark_streaming.py` 生成“更适合计时”的速度/显存对照（PPL 不在该脚本中计算）：
  - `results/mit_bench/pg19_benchmark_streaming.json`
  - `results/mit_bench/pg19_benchmark_recompute.json`
  - `results/mit_bench/wikitext_benchmark_streaming.json`
  - `results/mit_bench/wikitext_benchmark_recompute.json`
- 默认关闭：`RUN_MIT_BENCH=0`。若开启，默认 `gen_tokens=512`（可用环境变量 `MIT_BENCH_GEN_TOKENS` 调整）。该 benchmark 主要用于比较 tokens/s、TPOT、peak mem，以及 streaming vs recompute 的加速趋势。

---

## 7. 编译论文（表格自动填充）

默认 `NeurIPS/neurips_2025_compressed.tex` 不会自动 `\input` 生成表格，以避免把旧/不匹配的数字写入 PDF。
当你确认 `NeurIPS/generated/*.tex` 已更新且可信时，在 `NeurIPS/neurips_2025_compressed.tex` 顶部将 `\usegeneratedfalse` 改为 `\usegeneratedtrue`，再编译：

同理，正文页数要求严格时（4 页），默认关闭 Appendix；如需附录材料，可将 `\useappendixfalse` 改为 `\useappendixtrue`。

```bash
./build_paper_pdf.sh
```

---

## 8. 更新数字的推荐流程

1) 重新运行 `./run_paper_experiments.sh`（刷新 JSON + 重新生成 `NeurIPS/generated/*.tex`）  
2) 重新编译 `NeurIPS/neurips_2025_compressed.tex`

如果某些表格仍显示 `[INSERT DATA]`，说明对应 JSON 结果缺失/失败，需要检查 `results/paper_experiments/` 下是否生成了相应文件。

---

## 9. 负结果说明（目前以定性为主）

负结果相关方法涉及其他分支/环境差异（例如 FlashAttention、CUDA fusion、部分 compile/graph 行为），在课程项目资源约束下我们优先以 **定性总结表**输出：
- `NeurIPS/generated/negative_results.tex`

详细证据与过程记录在：
- `docs/探索日志.md`
- `docs/QUANT_FAILURE_REPORT.md`
- `docs/CUDA_KERNEL_FULL_REPORT.md`（如存在）
