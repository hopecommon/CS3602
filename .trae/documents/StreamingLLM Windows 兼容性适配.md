## Windows 环境适配方案

虽然 FlashAttention 核心库不支持 Windows，但我们可以通过以下修改让**整个 StreamingLLM 评测流程在 Windows 上正常跑通**（自动回退到 Math 内核，功能完整但速度稍慢）。

### 1. 代码兼容性修复
- **修复 `PYTHONPATH` 分隔符**：`experiments/run_mit_official_benchmark.py` 中硬编码了 Linux 的冒号 `:` 分隔符，会导致 Windows (使用分号 `;`) 路径解析失败。
  - **修改**: 将 `":"` 替换为 `os.pathsep` (自动识别系统分隔符)。

### 2. 创建 Windows 启动脚本
- 原项目依赖 `.sh` 脚本设置环境变量和参数，在 Windows 上无法直接运行。
- **新增 `run_final_experiments.ps1`**：创建一个 PowerShell 脚本，对齐 Linux 版的逻辑：
  - 自动加载 `.env` (如果存在)。
  - 设置 `HF_HOME` 等环境变量。
  - 调用 `python experiments/run_final_experiments.py`。

### 3. 更新依赖与文档
- **更新文档**: 在 `FLASH_INSTRUCTIONS.md` 中增加 "Windows 运行指南" 章节，说明如何使用 PowerShell 脚本以及预期的 "Flash not available" 警告。
- **依赖说明**: 明确指出在 Windows 上只需安装 `requirements.txt`，跳过 `flash-attn` 的安装步骤。

## 预期效果
- **可以运行**: 你可以在 Windows 终端（PowerShell）中直接执行评测，无需 WSL。
- **功能完整**: PPL 计算、KV Cache 压缩、RoPE 修正等核心逻辑完全一致。
- **性能差异**: 由于无法加载 FlashAttention，会自动使用 PyTorch 的 Math Backend。日志会显示 `[INFO] FlashAttention not available, falling back to math`，这是正常现象。

准备好后，我将执行上述修改。