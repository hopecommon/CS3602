# Windows PowerShell 启动脚本
# 功能: 加载 .env 环境变量并运行 Python 评测脚本

# 设置错误行为
$ErrorActionPreference = "Stop"

# 1. 加载 .env 文件
$EnvFile = ".env"
if (Test-Path $EnvFile) {
    Write-Host "正在加载 .env 环境变量..." -ForegroundColor Cyan
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#")) {
            $parts = $line.Split("=", 2)
            if ($parts.Count -eq 2) {
                $name = $parts[0].Trim()
                $value = $parts[1].Trim()
                # 移除可能的引号
                $value = $value -replace '^"|"$', ""
                [System.Environment]::SetEnvironmentVariable($name, $value, [System.EnvironmentVariableTarget]::Process)
                # Write-Host "  Set $name = $value" -ForegroundColor Gray
            }
        }
    }
} else {
    Write-Host "未找到 .env 文件，使用默认配置" -ForegroundColor Yellow
}

# 2. 设置必要的默认环境变量
if (-not $env:HF_HOME) {
    $env:HF_HOME = "$PWD\.cache\huggingface"
    Write-Host "设置 HF_HOME = $env:HF_HOME" -ForegroundColor Gray
}

# 3. 运行 Python 脚本
Write-Host "`n==========================================" -ForegroundColor Green
Write-Host "开始运行 StreamingLLM 评测 (Windows)" -ForegroundColor Green
Write-Host "==========================================`n"

# 检查 Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python 版本: $pythonVersion" -ForegroundColor Gray
} catch {
    Write-Error "未找到 python 命令，请确保 Python 已安装并添加到 PATH。"
    exit 1
}

# 构造参数
$ScriptPath = "experiments/run_final_experiments.py"
$ModelName = if ($env:MODEL_NAME) { $env:MODEL_NAME } else { "EleutherAI/pythia-2.8b" }

Write-Host "运行模型: $ModelName" -ForegroundColor Cyan
Write-Host "脚本路径: $ScriptPath" -ForegroundColor Cyan

# 执行
python $ScriptPath --model-name $ModelName

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n==========================================" -ForegroundColor Green
    Write-Host "评测成功完成!" -ForegroundColor Green
    Write-Host "结果保存在 results/final/" -ForegroundColor Green
    Write-Host "=========================================="
} else {
    Write-Host "`n==========================================" -ForegroundColor Red
    Write-Host "评测失败 (Exit Code: $LASTEXITCODE)" -ForegroundColor Red
    Write-Host "=========================================="
    exit $LASTEXITCODE
}
