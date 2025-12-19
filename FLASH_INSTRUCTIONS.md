# StreamingLLM Flash 加速集成 - 运行验证指南

本文档指导如何在具有 NVIDIA GPU 的环境中运行集成了 FlashAttention 的 StreamingLLM 评测，并验证其加速效果。

## 1. 环境准备

请确保运行环境满足以下要求：

- **GPU**: NVIDIA GPU（建议 Ampere 架构如 A100/3090/4090，或 Turing T4/2080 等），显存 >= 16GB。
- **驱动与 CUDA**: CUDA 11.8 或 12.x。
- **Python**: 3.10+。
- **依赖**: 已安装项目依赖，特别是 `torch` 和 `transformers`。

### 安装依赖

如果尚未安装，请运行：

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

> **注意**: `flash-attn` 是可选的，但在支持的 GPU 上安装它能获得最佳性能。如果没有安装，代码会自动回退到 PyTorch 内置的 math 后端。

## 2. 快速验证 (Smoke Test)

首先运行一个极简的验证脚本，确认 FlashAttention 适配器是否能被正确加载且不会报错。

请在项目根目录下运行以下命令：

```bash
python -c "from experiments.eval_utils import check_sdpa_flash_available; import json; print(json.dumps(check_sdpa_flash_available(), indent=2))"
```

**预期输出示例 (GPU 环境)**:
```json
{
  "flash_enabled": true,
  "smoke_test": true
}
```

如果输出 `flash_enabled: false`，说明当前环境不支持 FlashAttention（可能是驱动问题、PyTorch 版本过旧或无 GPU）。代码仍可运行，但将回退到 math 模式。

## 3. 运行完整评测

运行主实验脚本，它会自动对比 `math` (默认) 和 `flash` (加速) 两种后端的性能。

```bash
# 运行所有实验 (WikiText-103 和 PG19)
python experiments/run_final_experiments.py --model-name EleutherAI/pythia-2.8b
```

脚本会自动：
1. 检测 FlashAttention 可用性。
2. 分别以 `math` 和 `flash` 后端运行 Baseline 和 StreamingLLM 评测。
3. 将结果保存到 `results/final/`，文件名格式如 `wikitext_main_backend-flash.json`。

**预期日志片段**:
```text
[INFO] 运行实验: WikiText-103 (Backend: flash)
[INFO]   评估基线...
[INFO]   基线 PPL: 5.42, Runtime: 12.5s
[INFO]   评估 StreamingLLM...
[INFO]   StreamingLLM PPL: 5.45, Runtime: 8.2s
[INFO] ✓ 实验完成: WikiText-103 (加速比: 1.52x)
```

## 4. 生成对比图表

实验完成后，运行绘图脚本生成包含 math/flash 对比的图表：

```bash
python experiments/plot_fixed_eval_results.py
```

图表将保存在 `results/figures/` 目录下。请重点查看：
- `fixed_eval_speedup.png`: 观察 Flash 后端是否带来了显著的加速比提升。
- `fixed_eval_ppl_comparison.png`: 确认 Flash 后端的 PPL 与 Math 后端一致（或差异极小）。

## 5. 提交反馈

请将以下内容打包或截图反馈给我：
1. **第 2 步的输出** (Smoke Test JSON)。
2. **第 3 步的运行日志** (只需包含 WikiText 部分即可，确认是否成功跑通了 flash 后端)。
3. **`results/final/` 目录下生成的 JSON 文件列表**。
4. (可选) `results/figures/` 下生成的图片。

如果运行过程中报错，请直接粘贴完整的 Traceback。
