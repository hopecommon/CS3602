# 快速启动与运行指南

这个文档记录当前版本实验的标准运行流程、依赖说明和 legacy 脚本说明。请优先阅读并执行“推荐流程”;老脚本仅当需要和历史结果对比时才使用。

## 1. 环境准备（推荐复用 kvpress 虚拟环境）

```bash
cd ./CS3602
source kvpress/.venv/bin/activate

# 离线缓存（推荐）
export HF_HOME=$PWD/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"
```

> kvpress/.venv 已包含 torch/transformers/datasets 等依赖，无需另行 pip 安装。确保在每次运行前设置 HF cache 以避免网络请求。

### 如果 .venv 不存在

首次使用需要创建 kvpress 环境:

```bash
cd kvpress
UV_CACHE_DIR=$PWD/.cache/uv uv sync --all-groups
UV_CACHE_DIR=$PWD/.cache/uv uv sync --extra eval
source .venv/bin/activate
cd ..
```

**说明**: 
- `uv sync --all-groups` 安装所有依赖组
- `uv sync --extra eval` 安装评估相关依赖

## 1.1 可选 `.env`（共享配置）

在根目录复制 `.env.example` 为 `.env`，修改其中的 `HF_HOME`、`DATASET_NAME`、`PYTHON_BIN`、`N_SINK` 等字段即可，脚本会自动加载：

```
cp .env.example .env
# 编辑 .env 以适应当前环境
```

## 2. 推荐流程（高标准实验）

1. **核心脚本**：`run_comprehensive_comparisons.sh` 会在 WikiText-103 和 PG19 上运行 4 × 3 = 12 条配置（chunked / decode-loop / kvpress official）。它会在 `results/comprehensive/` 产出 JSON。
2. **执行命令**：
   ```bash
   chmod +x run_comprehensive_comparisons.sh
   ./run_comprehensive_comparisons.sh
   ```
3. **后处理**：查看 `results/comprehensive/*.json`，提取 runtime/PPL/speedup，或用 `results/figures/` 中已有图表；本 README 已列出关键数据供报告撰写。

## 3. 辅助工具（按需调用）

- `python experiments/run_decode_perplexity.py --method ours --dataset-name wikitext --dataset-config wikitext-103-v1`：单独跑 decode-loop。
- `bash experiments/run_kvpress_streaming_decode.sh`：让 kvpress 也走 decode-loop；对于 PG19，可设置 `DATASET_NAME=pg19 DATASET_CONFIG=\"\" bash ...`。
- `python experiments/plot_decode_loop_comparison.py`：根据 `results/comprehensive/*_decode_loop.json` 生成 runtime 与 PPL 图，输出到 `results/figures/`。
- `python experiments/test_streaming_llm.py`：快速 smoke test（可选）。

## 4. Legacy 脚本（仅限对比 / 参考）

1. `run_everything.sh`：旧版一键实验（baseline + streaming + kvpress + ablations），依然可用但结果生成方式与当前推荐流程不同，现标注为 **Legacy**。
2. `run_decoding_latency.sh`：旧版 decoding latency 评估。（仍可运行）
3. `experiments/run_all_experiments.py`：已停止维护，建议改用 shell 脚本。

## 5. 结果复现要求

- 12 条核心 JSON 保存在 `results/comprehensive/`，包含 runtime/PPL/speedup/compression ratio。
- 图表在 `results/figures/`，如需再现可用 Python/Matplotlib 读取 JSON。
- 报告写在 README 中，提交时将 repo push 到 GitHub 即可。
