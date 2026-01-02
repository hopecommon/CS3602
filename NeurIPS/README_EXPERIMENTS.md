# NeurIPS 论文实验复现指南

本文档说明如何 **一键跑完论文所需实验**，并自动生成论文使用的 LaTeX 表格（不手填数字，避免出错）。

核心原则：
- `NeurIPS/neurips_2025_compressed.tex` **不写死实验数字**，仅 `\input{NeurIPS/generated/*.tex}`。
- 所有数字来自 `results/**.json`，由脚本自动生成表格。
- Baseline 默认 **只跑一次并固定复用**（避免每次 sweep 重跑 baseline；PG19 很慢）。
- 默认会复用已有实验点结果（避免误触发长时间重跑）；每个结果文件都带 `config_hash`，参数变了会自动重跑。

---

## 1. 文件结构

```
CS3602/
├── NeurIPS/
│   ├── neurips_2025_compressed.tex     # 4页正文版（表格自动生成）
│   ├── generated/tables.tex            # 自动生成：主结果表
│   ├── generated/ablations.tex         # 自动生成：Slack/Max_Drop 消融表
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
PG19_SAMPLE_LENGTH=20000
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

---

## 4. 一键跑完论文全部实验并生成 TeX

```bash
# 推荐在仓库根目录执行（脚本也会自动切到仓库根目录）
./run_paper_experiments.sh
```

默认会跑：
- 主结果：Baseline(复用) + MIT + Ours（PG19 + WikiText）
- 消融：ladder（MIT → +Lazy → +Slack → +Max\_Drop + 以及 w/o Lazy 对照）（PG19）
- 扫描：PG19 上的 `R/σ/δ`（可配置取值）
- 生成 `NeurIPS/generated/*.tex`（主表、消融表、扫描表、负结果表）

---

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

---

## 7. 编译论文（表格自动填充）

```bash
cd NeurIPS
pdflatex neurips_2025_compressed.tex
```

若你还需要 bibtex：

```bash
bibtex neurips_2025_compressed
pdflatex neurips_2025_compressed.tex
pdflatex neurips_2025_compressed.tex
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
