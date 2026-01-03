# Windows PowerShell script for running fixed evaluation (decode-loop)
# Equivalent to run_fixed_evaluation.sh

$ErrorActionPreference = "Stop"

# 1. Load .env file if it exists
$EnvFile = ".env"
if (Test-Path $EnvFile) {
    Write-Host "Loading environment variables from $EnvFile..."
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -match '^([^#=]+)=(.*)$') {
            $key = $matches[1]
            $value = $matches[2]
            # Remove quotes if present
            $value = $value -replace '^["'']|["'']$', ''
            [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

# 2. Set default environment variables
if (-not $env:HF_HOME) { $env:HF_HOME = "$PWD\.cache\huggingface" }
if (-not $env:HF_DATASETS_CACHE) { $env:HF_DATASETS_CACHE = "$env:HF_HOME\datasets" }
if (-not $env:HF_HUB_OFFLINE) { $env:HF_HUB_OFFLINE = "1" }
if (-not $env:TRANSFORMERS_OFFLINE) { $env:TRANSFORMERS_OFFLINE = "1" }

# 3. Determine Python interpreter
$PYTHON = "python"
if ($env:PYTHON_BIN) {
    $PYTHON = $env:PYTHON_BIN
} elseif (Get-Command "conda" -ErrorAction SilentlyContinue) {
    # Try to find python in the specified conda environment
    $condaEnv = "lora_env"
    $condaInfo = conda info --envs | Select-String $condaEnv
    if ($condaInfo) {
        $envPath = $condaInfo.ToString().Split()[-1].Trim()
        $PYTHON = Join-Path $envPath "python.exe"
    }
} elseif (Test-Path "kvpress\.venv\Scripts\python.exe") {
    $PYTHON = "kvpress\.venv\Scripts\python.exe"
}

# 4. Experiment parameters
if (-not $env:MODEL_NAME) { $env:MODEL_NAME = "EleutherAI/pythia-2.8b" }
if (-not $env:RESULT_ROOT) { $env:RESULT_ROOT = "results\fixed_eval" }
if (-not $env:N_SINK) { $env:N_SINK = "4" }
if (-not $env:WINDOW_SIZE) { $env:WINDOW_SIZE = "2048" }
if (-not $env:COMPRESS_EVERY) { $env:COMPRESS_EVERY = "4" }
if (-not $env:WIKITEXT_MAX_TOKENS) { $env:WIKITEXT_MAX_TOKENS = "4096" }
if (-not $env:PG19_20K_MAX_TOKENS) { $env:PG19_20K_MAX_TOKENS = "20000" }

if (-not (Test-Path $env:RESULT_ROOT)) {
    New-Item -ItemType Directory -Force -Path $env:RESULT_ROOT | Out-Null
}

# 5. Define datasets and configs
$Datasets = @("wikitext", "pg19_20k")
$Configs = @{
    "wikitext" = "wikitext-103-v1"
    "pg19_20k" = ""
}
$MaxTokens = @{
    "wikitext" = $env:WIKITEXT_MAX_TOKENS
    "pg19_20k" = $env:PG19_20K_MAX_TOKENS
}
$Tags = @{
    "wikitext" = "wikitext"
    "pg19_20k" = "pg19_20k"
}

# 6. Define methods
if ($env:FIXED_EVAL_METHODS) {
    $Methods = $env:FIXED_EVAL_METHODS -split " "
} else {
    $Methods = @("baseline", "ours", "kvpress")
}

# 7. Run experiments
foreach ($dataset in $Datasets) {
    $config = $Configs[$dataset]
    $max_tok = $MaxTokens[$dataset]
    $dataset_tag = $Tags[$dataset]

    # Determine actual dataset name
    $actual_dataset = $dataset
    if ($dataset -like "pg19_*") {
        $actual_dataset = "pg19"
    }

    Write-Host "=========================================="
    Write-Host "Dataset: $dataset_tag"
    Write-Host "  Actual dataset: $actual_dataset"
    Write-Host "  Config: $config"
    Write-Host "  Max tokens: $max_tok"
    Write-Host "  N_sink: $env:N_SINK"
    Write-Host "  Window size: $env:WINDOW_SIZE"
    Write-Host "=========================================="

    foreach ($method in $Methods) {
        $output_file = Join-Path $env:RESULT_ROOT "${dataset_tag}_${method}.json"
        
        if (Test-Path $output_file) {
            # Check for legacy format (kvpress specific check from bash script)
            if ($method -eq "kvpress") {
                $content = Get-Content $output_file -Raw
                if ($content -notmatch '"total_time"') {
                    $legacy_file = $output_file.Replace(".json", "_legacy.json")
                    Write-Host "Found legacy KVPress result. Renaming to $legacy_file and re-running..."
                    Move-Item -Path $output_file -Destination $legacy_file -Force
                } else {
                    Write-Host "Skipping $output_file (already exists)"
                    continue
                }
            } else {
                Write-Host "Skipping $output_file (already exists)"
                continue
            }
        }

        Write-Host "=== $dataset_tag $method ==="
        
        $argsList = @(
            "experiments/run_decode_perplexity.py",
            "--model-name", $env:MODEL_NAME,
            "--method", $method,
            "--dataset-name", $actual_dataset,
            "--dataset-config", $config,
            "--max-eval-tokens", $max_tok,
            "--n-sink", $env:N_SINK,
            "--window-size", $env:WINDOW_SIZE,
            "--compress-every", $env:COMPRESS_EVERY,
            "--output", $output_file
        )

        & $PYTHON $argsList
        Write-Host ""
    }
}

# 8. Generate summary and plots
Write-Host "=========================================="
Write-Host "Generating summary table..."
& $PYTHON experiments/summarize_fixed_eval_results.py --results-dir $env:RESULT_ROOT --output "$env:RESULT_ROOT\summary.md"

Write-Host "Generating plots..."
& $PYTHON experiments/plot_fixed_eval_results.py

# 9. Optional MIT Benchmark
if ($env:RUN_MIT_OFFICIAL_BENCHMARK -eq "1") {
    Write-Host "=========================================="
    Write-Host "Running MIT official throughput/VRAM benchmark - non-PPL..."
    
    if ($env:MIT_BENCH_PYTHON) { $MIT_PYTHON = $env:MIT_BENCH_PYTHON } else { $MIT_PYTHON = $PYTHON }
    if ($env:MIT_BENCH_MODEL_PATH) { $MIT_MODEL = $env:MIT_BENCH_MODEL_PATH } else { $MIT_MODEL = $env:MODEL_NAME }
    if ($env:PG19_SAMPLE_FILE) { $MIT_DATA = $env:PG19_SAMPLE_FILE } else { $MIT_DATA = "data/pg19/long_context_20000.json" }
    if ($env:MIT_BENCH_DATA_JSON) { $MIT_DATA = $env:MIT_BENCH_DATA_JSON }

    if (-not (Test-Path "results\mit_official")) {
        New-Item -ItemType Directory -Force -Path "results\mit_official" | Out-Null
    }

    & $PYTHON experiments/run_mit_official_benchmark.py `
        --python $MIT_PYTHON `
        --model-name-or-path $MIT_MODEL `
        --data-json $MIT_DATA `
        --prefix-tokens 20000 `
        --gen-tokens 512 `
        --start-size $env:N_SINK `
        --recent-size $env:WINDOW_SIZE `
        --output "results/mit_official/pg19_20k_benchmark.json"
}

Write-Host "=========================================="
Write-Host "All evaluations complete!"
Write-Host "Results saved to: $env:RESULT_ROOT"
Write-Host "=========================================="
