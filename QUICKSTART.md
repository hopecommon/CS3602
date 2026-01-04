# QUICKSTART

这份文档给出当前仓库的“最短可复现路径”：跑出论文主实验结果并构建 PDF。更详细的失败尝试与分析见 `docs/探索日志.md`。

## 1) 环境准备

推荐复用 `kvpress/.venv`：
```bash
cd /data2/jflin/CS3602
source kvpress/.venv/bin/activate
```

离线模式（建议在无网络环境下固定开启）：
```bash
export HF_HOME=$PWD/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"
```

## 2) 配置 `.env`（强烈建议）

复制模板：
```bash
cp .env.example .env
```

为保证 PG19 的 PPL 跨机器/脚本可比，建议固定到同一个长文本文件，并用评测脚本控制评测 token 数：
```bash
# PG19 固定样本（推荐）
export PG19_SAMPLE_FILE=data/pg19/long_context_50000.json
export PG19_SAMPLE_LENGTH=50000

# WikiText 固定样本（推荐）
export WIKITEXT_SAMPLE_FILE=data/wikitext/long_context_4096.json
export WIKITEXT_SAMPLE_LENGTH=4096
```

说明：
- PG19 的 PPL 对“选了哪一本书/哪一段”敏感；如果换了采样文件，PPL 可能从 ~20 变成 ~8（反之亦然），这不代表算法本身变强/变弱。
- `run_paper_experiments.sh` 默认会 pin 到 `data/pg19/long_context_50000.json`，你也可以在 `.env` 显式写死以避免误用。

## 3) 运行论文主实验（Auto-cap）

`run_paper_experiments.sh` 使用 Auto-cap：给定 `CAP_TOTAL=2048`，自动根据 `sink/slack/overlap/refresh` 推导 `window_size`，避免手动设置 window 导致总预算不一致。

```bash
chmod +x run_paper_experiments.sh

./run_paper_experiments.sh        # 默认跳过已有结果
./run_paper_experiments.sh -n     # dry-run，只打印命令
./run_paper_experiments.sh -f     # 强制重跑
```

主要输出：
- 结果 JSON：`results/paper_experiments/*.json`
- 更细粒度结果：`results/paper_experiments/*_runs/`
- 论文表格/图生成物：`NeurIPS/generated/*.tex`

## 4) 构建论文 PDF

```bash
chmod +x build_paper_pdf.sh
./build_paper_pdf.sh
```

输出：`NeurIPS/neurips_2025_compressed.pdf`

## 5) Legacy（可选）

仓库保留了早期脚本与结果用于对比/回溯（例如 `run_fixed_evaluation.sh`、`results/fixed_eval/`、`run_comprehensive_comparisons.sh`），但当前主线以 `run_paper_experiments.sh` + `NeurIPS/` 为准。

