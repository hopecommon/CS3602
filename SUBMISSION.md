## 提交前自检清单（建议按顺序）

### 1) 仓库状态与敏感文件

- 确认工作区干净：`git status -sb`
- 确认不会提交本地配置/缓存：`.env`、`.cache/`、`kvpress/.venv/` 不应出现在 `git status` 中
- 如果你用“打包目录”的方式提交：务必不要把 `.git/` 一起打包（本仓库 `.git/lfs` 可能非常大）

### 2) 可运行性（不跑模型也能做的检查）

- 语法检查：`python -m compileall -q experiments streaming_llm`
- 文档入口是否能对应到结果：`results/fixed_eval/summary.md`、`results/figures/fixed_eval_*.png`

### 3) 复现主结果（需要已缓存模型/数据）

- 主入口：`chmod +x run_fixed_evaluation.sh && ./run_fixed_evaluation.sh`
- 生成汇总表：`python experiments/summarize_fixed_eval_results.py`
- 生成图表：`python experiments/plot_fixed_eval_results.py`

### 4) 推荐的“干净打包”方式（只包含 Git 跟踪文件）

- 生成提交压缩包：`git archive -o CS3602_submission.zip HEAD`

