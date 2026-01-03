<#
.SYNOPSIS
Paper experiments (one-click) + auto-generated LaTeX tables.
Windows PowerShell version of run_paper_experiments.sh

.DESCRIPTION
- Runs a minimal, paper-oriented set of evaluations for:
  * Baseline (sliding window, no KV cache; same context cap)
  * MIT StreamingLLM (Start+Recent)
  * Ours (StreamingLLM + Lazy Prune + Slack + Max_Drop)
- Reuses a fixed baseline by default (to avoid repeatedly running baseline).
- Generates: `NeurIPS/generated/tables.tex`.

.EXAMPLE
.\run_paper_experiments.ps1
.\run_paper_experiments.ps1 -Force
#>

[CmdletBinding()]
param (
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# ------------------------------------------------------------------------------
# Load .env if present
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Environment defaults (offline-friendly)
# ------------------------------------------------------------------------------
if (-not $env:HF_HOME) { $env:HF_HOME = "$PWD\.cache\huggingface" }
if (-not $env:HF_DATASETS_CACHE) { $env:HF_DATASETS_CACHE = "$env:HF_HOME\datasets" }
if (-not $env:HF_HUB_OFFLINE) { $env:HF_HUB_OFFLINE = "1" }
if (-not $env:TRANSFORMERS_OFFLINE) { $env:TRANSFORMERS_OFFLINE = "1" }

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

if (-not $env:MODEL_NAME) { $env:MODEL_NAME = "EleutherAI/pythia-2.8b" }

if ($env:RESULTS_DIR) { $RESULTS_DIR = $env:RESULTS_DIR } else { $RESULTS_DIR = "results\paper_experiments" }
if ($env:BASELINE_DIR) { $BASELINE_DIR = $env:BASELINE_DIR } else { $BASELINE_DIR = "results\baselines" }

if (-not (Test-Path $RESULTS_DIR)) { New-Item -ItemType Directory -Force -Path $RESULTS_DIR | Out-Null }
if (-not (Test-Path $BASELINE_DIR)) { New-Item -ItemType Directory -Force -Path $BASELINE_DIR | Out-Null }

if (-not $env:WIKITEXT_TOKENS) { $env:WIKITEXT_TOKENS = "4096" }
if (-not $env:PG19_TOKENS) { $env:PG19_TOKENS = "20000" }

# How many independent samples for dataset evaluation (mean across samples).
if (-not $env:MAX_SAMPLES_WIKI) { $env:MAX_SAMPLES_WIKI = "64" }
if (-not $env:MAX_SAMPLES_PG19) { $env:MAX_SAMPLES_PG19 = "10" }

# ---------------------------------------------------------------------------
# Main-result fairness: align (S, W) between MIT and Ours.
# ---------------------------------------------------------------------------
if (-not $env:MAIN_SINK) { $env:MAIN_SINK = "4" }
if (-not $env:MAIN_WINDOW) { $env:MAIN_WINDOW = "2044" }

# Ours default hyperparameters
if (-not $env:OURS_SINK) { $env:OURS_SINK = $env:MAIN_SINK }
if (-not $env:OURS_WINDOW) { $env:OURS_WINDOW = $env:MAIN_WINDOW }
if (-not $env:OURS_COMPRESS_EVERY) { $env:OURS_COMPRESS_EVERY = "64" }
if (-not $env:OURS_CACHE_SLACK) { $env:OURS_CACHE_SLACK = "16" }
if (-not $env:OURS_MAX_DROP) { $env:OURS_MAX_DROP = "32" }

# MIT baseline config
if (-not $env:MIT_SINK) { $env:MIT_SINK = $env:MAIN_SINK }
if (-not $env:MIT_WINDOW) { $env:MIT_WINDOW = $env:MAIN_WINDOW }

# Optional additional (S, W) pair
if (-not $env:EXTRA_SINK) { $env:EXTRA_SINK = "32" }
if (-not $env:EXTRA_WINDOW) { $env:EXTRA_WINDOW = "2016" }
if (-not $env:RUN_SINK_CONFOUND) { $env:RUN_SINK_CONFOUND = "1" }

# Repeatability protocol
if (-not $env:WARMUP_RUNS) { $env:WARMUP_RUNS = "1" }
if (-not $env:REPEAT_RUNS) { $env:REPEAT_RUNS = "3" }

$SKIP_EXISTING = "1"
if ($Force) {
    $SKIP_EXISTING = "0"
} elseif ($env:SKIP_EXISTING) {
    $SKIP_EXISTING = $env:SKIP_EXISTING
}

# Run knobs (1=run, 0=skip)
if (-not $env:RUN_MAIN) { $env:RUN_MAIN = "1" }
if (-not $env:RUN_ABLATIONS) { $env:RUN_ABLATIONS = "1" }
if (-not $env:RUN_SWEEPS) { $env:RUN_SWEEPS = "1" }

if (-not $env:SWEEP_R_VALUES) { $env:SWEEP_R_VALUES = "1 4 16 32 64 128" }
if (-not $env:SWEEP_SIGMA_VALUES) { $env:SWEEP_SIGMA_VALUES = "0 8 16 32 64" }
if (-not $env:SWEEP_DELTA_VALUES) { $env:SWEEP_DELTA_VALUES = "0 8 16 32 64" }

# Baseline generation
if (-not $env:AUTO_BASELINE) { $env:AUTO_BASELINE = "1" }
if (-not $env:BASELINE_RUNS) { $env:BASELINE_RUNS = "1" }
if (-not $env:BASELINE_SINK) { $env:BASELINE_SINK = "4" }
if (-not $env:BASELINE_WINDOW) { $env:BASELINE_WINDOW = "2044" }
if (-not $env:BASELINE_WIKITEXT) { $env:BASELINE_WIKITEXT = Join-Path $BASELINE_DIR "wikitext_baseline_avg.json" }
if (-not $env:BASELINE_PG19) { $env:BASELINE_PG19 = Join-Path $BASELINE_DIR "pg19_baseline_avg.json" }

if (-not $env:STRICT_BASELINE_CHECK) { $env:STRICT_BASELINE_CHECK = "0" }

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

function Run-Command {
    param(
        [string]$Name,
        [string[]]$CmdArgs
    )
    Write-Host ""
    Write-Host "================================================================"
    Write-Host "$Name"
    Write-Host "================================================================"
    Write-Host "cmd: $PYTHON $CmdArgs"
    & $PYTHON $CmdArgs
}

function Run-Repeated {
    param(
        [string]$Name,
        [string]$OutPath,
        [string[]]$ExtraArgs
    )
    
    $argsList = @("experiments/paper/run_repeated_eval.py", "--warmup", $env:WARMUP_RUNS, "--runs", $env:REPEAT_RUNS)
    if ($SKIP_EXISTING -eq "1") {
        $argsList += "--skip-existing"
    }
    $argsList += "--out"
    $argsList += $OutPath
    $argsList += $ExtraArgs

    Run-Command -Name $Name -CmdArgs $argsList
}

function Ensure-Fixed-Baselines {
    if ($env:AUTO_BASELINE -ne "1") { return }
    
    if ((-not $Force) -and (Test-Path $env:BASELINE_WIKITEXT) -and (Test-Path $env:BASELINE_PG19)) {
        return
    }

    Write-Host "Fixed baselines missing; generating them now (runs=$env:BASELINE_RUNS)..."
    
    $argsList = @(
        "experiments/run_fixed_baseline.py",
        "--model-name", $env:MODEL_NAME,
        "--dtype", "float16",
        "--n-sink", $env:BASELINE_SINK,
        "--window-size", $env:BASELINE_WINDOW,
        "--runs", $env:BASELINE_RUNS,
        "--output-dir", $BASELINE_DIR
    )
    if ($Force) {
        $argsList += "--no-skip-existing"
    }

    Run-Command -Name "Generate fixed baselines" -CmdArgs $argsList
}

function Ensure-Baseline-Link {
    param(
        [string]$Dataset,
        [string]$Src,
        [string]$Dst
    )
    
    $expected_max_length = [int]$env:BASELINE_SINK + [int]$env:BASELINE_WINDOW
    $expected_stride = [int]([int]$env:BASELINE_WINDOW / 2)
    
    if ($Dataset -eq "pg19") { $tokens = $env:PG19_TOKENS } else { $tokens = $env:WIKITEXT_TOKENS }

    # Check existing destination
    if ((-not $Force) -and (Test-Path $Dst)) {
        $checkArgs = @(
            "experiments/paper/check_baseline_compat.py",
            "--baseline", $Dst,
            "--model-name", $env:MODEL_NAME,
            "--dtype", "float16",
            "--max-length", $expected_max_length,
            "--stride", $expected_stride,
            "--max-eval-tokens", $tokens,
            "--n-sink", $env:BASELINE_SINK,
            "--window-size", $env:BASELINE_WINDOW
        )
        
        # We need to capture exit code, try/catch around command execution
        try {
            & $PYTHON $checkArgs | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✓ baseline exists (compatible): $Dst"
                return
            }
        } catch {}
        
        Write-Host "⚠ baseline exists but fingerprint/config mismatch: $Dst"
        if ($env:STRICT_BASELINE_CHECK -ne "1") {
            Write-Host "  Continuing with the existing baseline (STRICT_BASELINE_CHECK=0)."
            return
        }
    }

    # Generate source if missing
    if (-not (Test-Path $Src)) {
        Ensure-Fixed-Baselines
    }

    # Check source compatibility
    if (Test-Path $Src) {
        $checkArgs = @(
            "experiments/paper/check_baseline_compat.py",
            "--baseline", $Src,
            "--model-name", $env:MODEL_NAME,
            "--dtype", "float16",
            "--max-length", $expected_max_length,
            "--stride", $expected_stride,
            "--max-eval-tokens", $tokens,
            "--n-sink", $env:BASELINE_SINK,
            "--window-size", $env:BASELINE_WINDOW
        )

        try {
            & $PYTHON $checkArgs | Out-Null
            $compatible = ($LASTEXITCODE -eq 0)
        } catch { $compatible = $false }

        if (-not $compatible) {
            Write-Host "⚠ fixed baseline fingerprint/config mismatch: $Src"
            if ($env:STRICT_BASELINE_CHECK -eq "1") {
                Write-Host "  STRICT_BASELINE_CHECK=1 -> regenerating baselines (overwrite, runs=$env:BASELINE_RUNS)..."
                
                $regenArgs = @(
                    "experiments/run_fixed_baseline.py",
                    "--model-name", $env:MODEL_NAME,
                    "--dtype", "float16",
                    "--n-sink", $env:BASELINE_SINK,
                    "--window-size", $env:BASELINE_WINDOW,
                    "--runs", $env:BASELINE_RUNS,
                    "--no-skip-existing",
                    "--output-dir", $BASELINE_DIR
                )
                Run-Command -Name "Regenerate fixed baselines" -CmdArgs $regenArgs
            } else {
                Write-Host "  Continuing with the existing baseline (STRICT_BASELINE_CHECK=0)."
            }
        }
    }

    if (Test-Path $Src) {
        Copy-Item -Path $Src -Destination $Dst -Force
        Write-Host "✓ baseline copied: $Dst <- $Src"
        return
    }

    Write-Error "✗ baseline missing for $Dataset: expected $Src"
}

# ------------------------------------------------------------------------------
# 0) Reuse fixed baselines
# ------------------------------------------------------------------------------
Ensure-Fixed-Baselines
Ensure-Baseline-Link -Dataset "wikitext" -Src $env:BASELINE_WIKITEXT -Dst (Join-Path $RESULTS_DIR "wikitext_baseline.json")
Ensure-Baseline-Link -Dataset "pg19" -Src $env:BASELINE_PG19 -Dst (Join-Path $RESULTS_DIR "pg19_baseline.json")

if ($env:RUN_MAIN -eq "1") {
    # ------------------------------------------------------------------------------
    # 1) MIT StreamingLLM (Start+Recent) - MAIN (S,W) aligned
    # ------------------------------------------------------------------------------
    Run-Repeated -Name "MIT StreamingLLM (PG19)" -OutPath (Join-Path $RESULTS_DIR "pg19_mit.json") -ExtraArgs @(
        "--model-name", $env:MODEL_NAME, "--dataset-name", "pg19", "--dataset-config", "pg19", "--split", "test",
        "--max-samples", $env:MAX_SAMPLES_PG19, "--max-eval-tokens", $env:PG19_TOKENS,
        "--max-length", "2048", "--stride", "1022",
        "--n-sink", $env:MAIN_SINK, "--window-size", $env:MAIN_WINDOW,
        "--compress-every", "1", "--streaming-mode", "mit", "--cache-backend", "mit", "--mode", "streaming",
        "--baseline-results", (Join-Path $RESULTS_DIR "pg19_baseline.json")
    )

    Run-Repeated -Name "MIT StreamingLLM (WikiText)" -OutPath (Join-Path $RESULTS_DIR "wikitext_mit.json") -ExtraArgs @(
        "--model-name", $env:MODEL_NAME, "--dataset-name", "wikitext", "--dataset-config", "wikitext-103-v1", "--split", "test",
        "--max-samples", $env:MAX_SAMPLES_WIKI, "--max-eval-tokens", $env:WIKITEXT_TOKENS,
        "--max-length", "2048", "--stride", "1022",
        "--n-sink", $env:MAIN_SINK, "--window-size", $env:MAIN_WINDOW,
        "--compress-every", "1", "--streaming-mode", "mit", "--cache-backend", "mit", "--mode", "streaming",
        "--baseline-results", (Join-Path $RESULTS_DIR "wikitext_baseline.json")
    )

    # ------------------------------------------------------------------------------
    # 1.1) Fairness check: same semantics, different implementation
    # ------------------------------------------------------------------------------
    Run-Repeated -Name "Ours-framework-only (PG19; Start+Recent semantics)" -OutPath (Join-Path $RESULTS_DIR "pg19_ours_framework_only.json") -ExtraArgs @(
        "--model-name", $env:MODEL_NAME, "--dataset-name", "pg19", "--dataset-config", "pg19", "--split", "test",
        "--max-samples", $env:MAX_SAMPLES_PG19, "--max-eval-tokens", $env:PG19_TOKENS,
        "--max-length", "2048", "--stride", "1022",
        "--n-sink", $env:MAIN_SINK, "--window-size", $env:MAIN_WINDOW,
        "--compress-every", "1", "--cache-slack", "0", "--max-drop", "0",
        "--streaming-mode", "ours", "--cache-backend", "mit", "--mode", "streaming",
        "--baseline-results", (Join-Path $RESULTS_DIR "pg19_baseline.json")
    )

    Run-Repeated -Name "Ours-framework-only (WikiText; Start+Recent semantics)" -OutPath (Join-Path $RESULTS_DIR "wikitext_ours_framework_only.json") -ExtraArgs @(
        "--model-name", $env:MODEL_NAME, "--dataset-name", "wikitext", "--dataset-config", "wikitext-103-v1", "--split", "test",
        "--max-samples", $env:MAX_SAMPLES_WIKI, "--max-eval-tokens", $env:WIKITEXT_TOKENS,
        "--max-length", "2048", "--stride", "1022",
        "--n-sink", $env:MAIN_SINK, "--window-size", $env:MAIN_WINDOW,
        "--compress-every", "1", "--cache-slack", "0", "--max-drop", "0",
        "--streaming-mode", "ours", "--cache-backend", "mit", "--mode", "streaming",
        "--baseline-results", (Join-Path $RESULTS_DIR "wikitext_baseline.json")
    )

    # ------------------------------------------------------------------------------
    # 2) Ours (Lazy + Slack + Max_Drop) - MAIN (S,W) aligned
    # ------------------------------------------------------------------------------
    Run-Repeated -Name "Ours (PG19)" -OutPath (Join-Path $RESULTS_DIR "pg19_ours.json") -ExtraArgs @(
        "--model-name", $env:MODEL_NAME, "--dataset-name", "pg19", "--dataset-config", "pg19", "--split", "test",
        "--max-samples", $env:MAX_SAMPLES_PG19, "--max-eval-tokens", $env:PG19_TOKENS,
        "--max-length", "2048", "--stride", "1022",
        "--n-sink", $env:MAIN_SINK, "--window-size", $env:MAIN_WINDOW,
        "--compress-every", $env:OURS_COMPRESS_EVERY, "--cache-slack", $env:OURS_CACHE_SLACK, "--max-drop", $env:OURS_MAX_DROP,
        "--streaming-mode", "ours", "--cache-backend", "ours", "--mode", "streaming",
        "--baseline-results", (Join-Path $RESULTS_DIR "pg19_baseline.json")
    )

    Run-Repeated -Name "Ours (WikiText)" -OutPath (Join-Path $RESULTS_DIR "wikitext_ours.json") -ExtraArgs @(
        "--model-name", $env:MODEL_NAME, "--dataset-name", "wikitext", "--dataset-config", "wikitext-103-v1", "--split", "test",
        "--max-samples", $env:MAX_SAMPLES_WIKI, "--max-eval-tokens", $env:WIKITEXT_TOKENS,
        "--max-length", "2048", "--stride", "1022",
        "--n-sink", $env:MAIN_SINK, "--window-size", $env:MAIN_WINDOW,
        "--compress-every", $env:OURS_COMPRESS_EVERY, "--cache-slack", $env:OURS_CACHE_SLACK, "--max-drop", $env:OURS_MAX_DROP,
        "--streaming-mode", "ours", "--cache-backend", "ours", "--mode", "streaming",
        "--baseline-results", (Join-Path $RESULTS_DIR "wikitext_baseline.json")
    )

    # ------------------------------------------------------------------------------
    # 2.1) Sink-size confound check (optional)
    # ------------------------------------------------------------------------------
    if ($env:RUN_SINK_CONFOUND -eq "1") {
        Run-Repeated -Name "MIT StreamingLLM (PG19, sink confound S=$env:EXTRA_SINK)" -OutPath (Join-Path $RESULTS_DIR "pg19_mit_s${env:EXTRA_SINK}.json") -ExtraArgs @(
            "--model-name", $env:MODEL_NAME, "--dataset-name", "pg19", "--dataset-config", "pg19", "--split", "test",
            "--max-samples", $env:MAX_SAMPLES_PG19, "--max-eval-tokens", $env:PG19_TOKENS,
            "--max-length", "2048", "--stride", "1022",
            "--n-sink", $env:EXTRA_SINK, "--window-size", $env:EXTRA_WINDOW,
            "--compress-every", "1", "--streaming-mode", "mit", "--mode", "streaming",
            "--baseline-results", (Join-Path $RESULTS_DIR "pg19_baseline.json")
        )

        Run-Repeated -Name "Ours (PG19, sink confound S=$env:EXTRA_SINK)" -OutPath (Join-Path $RESULTS_DIR "pg19_ours_s${env:EXTRA_SINK}.json") -ExtraArgs @(
            "--model-name", $env:MODEL_NAME, "--dataset-name", "pg19", "--dataset-config", "pg19", "--split", "test",
            "--max-samples", $env:MAX_SAMPLES_PG19, "--max-eval-tokens", $env:PG19_TOKENS,
            "--max-length", "2048", "--stride", "1022",
            "--n-sink", $env:EXTRA_SINK, "--window-size", $env:EXTRA_WINDOW,
            "--compress-every", $env:OURS_COMPRESS_EVERY, "--cache-slack", $env:OURS_CACHE_SLACK, "--max-drop", $env:OURS_MAX_DROP,
            "--streaming-mode", "ours", "--mode", "streaming",
            "--baseline-results", (Join-Path $RESULTS_DIR "pg19_baseline.json")
        )
    }
}

# ------------------------------------------------------------------------------
# 2.5) Ablations (ladder): MIT -> +Lazy -> +Slack -> +Max_Drop
# ------------------------------------------------------------------------------
if ($env:RUN_ABLATIONS -eq "1") {
    $ABL_DIR = Join-Path $RESULTS_DIR "ablations"
    if (-not (Test-Path $ABL_DIR)) { New-Item -ItemType Directory -Force -Path $ABL_DIR | Out-Null }

    # A0: MIT
    Run-Repeated -Name "Ablation A0 (PG19) MIT (aligned S,W)" -OutPath (Join-Path $ABL_DIR "pg19_A0_mit.json") -ExtraArgs @(
        "--model-name", $env:MODEL_NAME, "--dataset-name", "pg19", "--dataset-config", "pg19", "--split", "test",
        "--max-samples", $env:MAX_SAMPLES_PG19, "--max-eval-tokens", $env:PG19_TOKENS,
        "--max-length", "2048", "--stride", "1022",
        "--n-sink", $env:MAIN_SINK, "--window-size", $env:MAIN_WINDOW,
        "--compress-every", "1", "--streaming-mode", "mit", "--mode", "streaming",
        "--baseline-results", (Join-Path $RESULTS_DIR "pg19_baseline.json")
    )

    # A1: +Lazy
    Run-Repeated -Name "Ablation A1 (PG19) +Lazy (ours, no slack/max_drop)" -OutPath (Join-Path $ABL_DIR "pg19_A1_lazy.json") -ExtraArgs @(
        "--model-name", $env:MODEL_NAME, "--dataset-name", "pg19", "--dataset-config", "pg19", "--split", "test",
        "--max-samples", $env:MAX_SAMPLES_PG19, "--max-eval-tokens", $env:PG19_TOKENS,
        "--max-length", "2048", "--stride", "1022",
        "--n-sink", $env:MAIN_SINK, "--window-size", $env:MAIN_WINDOW,
        "--compress-every", $env:OURS_COMPRESS_EVERY, "--cache-slack", "0", "--max-drop", "0",
        "--streaming-mode", "ours", "--mode", "streaming",
        "--baseline-results", (Join-Path $RESULTS_DIR "pg19_baseline.json")
    )

    # A2: +Lazy +Slack
    Run-Repeated -Name "Ablation A2 (PG19) +Lazy+Slack (ours, no max_drop)" -OutPath (Join-Path $ABL_DIR "pg19_A2_lazy_slack.json") -ExtraArgs @(
        "--model-name", $env:MODEL_NAME, "--dataset-name", "pg19", "--dataset-config", "pg19", "--split", "test",
        "--max-samples", $env:MAX_SAMPLES_PG19, "--max-eval-tokens", $env:PG19_TOKENS,
        "--max-length", "2048", "--stride", "1022",
        "--n-sink", $env:MAIN_SINK, "--window-size", $env:MAIN_WINDOW,
        "--compress-every", $env:OURS_COMPRESS_EVERY, "--cache-slack", $env:OURS_CACHE_SLACK, "--max-drop", "0",
        "--streaming-mode", "ours", "--mode", "streaming",
        "--baseline-results", (Join-Path $RESULTS_DIR "pg19_baseline.json")
    )

    # A3: Full
    Run-Repeated -Name "Ablation A3 (PG19) +Lazy+Slack+Max_Drop (full)" -OutPath (Join-Path $ABL_DIR "pg19_A3_full.json") -ExtraArgs @(
        "--model-name", $env:MODEL_NAME, "--dataset-name", "pg19", "--dataset-config", "pg19", "--split", "test",
        "--max-samples", $env:MAX_SAMPLES_PG19, "--max-eval-tokens", $env:PG19_TOKENS,
        "--max-length", "2048", "--stride", "1022",
        "--n-sink", $env:MAIN_SINK, "--window-size", $env:MAIN_WINDOW,
        "--compress-every", $env:OURS_COMPRESS_EVERY, "--cache-slack", $env:OURS_CACHE_SLACK, "--max-drop", $env:OURS_MAX_DROP,
        "--streaming-mode", "ours", "--mode", "streaming",
        "--baseline-results", (Join-Path $RESULTS_DIR "pg19_baseline.json")
    )

    # A(-Lazy): strict pruning
    Run-Repeated -Name "Ablation A(-Lazy) (PG19) w/o Lazy (ours: R=1, no slack/max_drop)" -OutPath (Join-Path $ABL_DIR "pg19_Aneg_lazy_strict.json") -ExtraArgs @(
        "--model-name", $env:MODEL_NAME, "--dataset-name", "pg19", "--dataset-config", "pg19", "--split", "test",
        "--max-samples", $env:MAX_SAMPLES_PG19, "--max-eval-tokens", $env:PG19_TOKENS,
        "--max-length", "2048", "--stride", "1022",
        "--n-sink", $env:MAIN_SINK, "--window-size", $env:MAIN_WINDOW,
        "--compress-every", "1", "--cache-slack", "0", "--max-drop", "0",
        "--streaming-mode", "ours", "--mode", "streaming",
        "--baseline-results", (Join-Path $RESULTS_DIR "pg19_baseline.json")
    )
}

# ------------------------------------------------------------------------------
# 2.6) Sweeps (R / sigma / delta) for PG19
# ------------------------------------------------------------------------------
if ($env:RUN_SWEEPS -eq "1") {
    $SWEEP_DIR = Join-Path $RESULTS_DIR "sweeps"
    if (-not (Test-Path "$SWEEP_DIR\R")) { New-Item -ItemType Directory -Force -Path "$SWEEP_DIR\R" | Out-Null }
    if (-not (Test-Path "$SWEEP_DIR\sigma")) { New-Item -ItemType Directory -Force -Path "$SWEEP_DIR\sigma" | Out-Null }
    if (-not (Test-Path "$SWEEP_DIR\delta")) { New-Item -ItemType Directory -Force -Path "$SWEEP_DIR\delta" | Out-Null }

    $R_VALUES = $env:SWEEP_R_VALUES -split " "
    foreach ($R in $R_VALUES) {
        Run-Repeated -Name "Sweep R=$R (PG19)" -OutPath (Join-Path "$SWEEP_DIR\R" "pg19_R${R}.json") -ExtraArgs @(
            "--model-name", $env:MODEL_NAME, "--dataset-name", "pg19", "--dataset-config", "pg19", "--split", "test",
            "--max-samples", $env:MAX_SAMPLES_PG19, "--max-eval-tokens", $env:PG19_TOKENS,
            "--max-length", "2048", "--stride", "1022",
            "--n-sink", $env:OURS_SINK, "--window-size", $env:OURS_WINDOW,
            "--compress-every", $R, "--cache-slack", $env:OURS_CACHE_SLACK, "--max-drop", $env:OURS_MAX_DROP,
            "--streaming-mode", "ours", "--mode", "streaming",
            "--baseline-results", (Join-Path $RESULTS_DIR "pg19_baseline.json")
        )
    }

    $SIGMA_VALUES = $env:SWEEP_SIGMA_VALUES -split " "
    foreach ($SIGMA in $SIGMA_VALUES) {
        Run-Repeated -Name "Sweep sigma=$SIGMA (PG19)" -OutPath (Join-Path "$SWEEP_DIR\sigma" "pg19_sigma${SIGMA}.json") -ExtraArgs @(
            "--model-name", $env:MODEL_NAME, "--dataset-name", "pg19", "--dataset-config", "pg19", "--split", "test",
            "--max-samples", $env:MAX_SAMPLES_PG19, "--max-eval-tokens", $env:PG19_TOKENS,
            "--max-length", "2048", "--stride", "1022",
            "--n-sink", $env:OURS_SINK, "--window-size", $env:OURS_WINDOW,
            "--compress-every", $env:OURS_COMPRESS_EVERY, "--cache-slack", $SIGMA, "--max-drop", $env:OURS_MAX_DROP,
            "--streaming-mode", "ours", "--mode", "streaming",
            "--baseline-results", (Join-Path $RESULTS_DIR "pg19_baseline.json")
        )
    }

    $DELTA_VALUES = $env:SWEEP_DELTA_VALUES -split " "
    foreach ($DELTA in $DELTA_VALUES) {
        Run-Repeated -Name "Sweep delta=$DELTA (PG19)" -OutPath (Join-Path "$SWEEP_DIR\delta" "pg19_delta${DELTA}.json") -ExtraArgs @(
            "--model-name", $env:MODEL_NAME, "--dataset-name", "pg19", "--dataset-config", "pg19", "--split", "test",
            "--max-samples", $env:MAX_SAMPLES_PG19, "--max-eval-tokens", $env:PG19_TOKENS,
            "--max-length", "2048", "--stride", "1022",
            "--n-sink", $env:OURS_SINK, "--window-size", $env:OURS_WINDOW,
            "--compress-every", $env:OURS_COMPRESS_EVERY, "--cache-slack", $env:OURS_CACHE_SLACK, "--max-drop", $DELTA,
            "--streaming-mode", "ours", "--mode", "streaming",
            "--baseline-results", (Join-Path $RESULTS_DIR "pg19_baseline.json")
        )
    }
}

# ------------------------------------------------------------------------------
# 3) Generate LaTeX tables for the paper
# ------------------------------------------------------------------------------
Run-Command -Name "Generate LaTeX tables" -CmdArgs @(
    "experiments/paper/generate_tables_tex.py",
    "--results-dir", $RESULTS_DIR,
    "--baseline-dir", $BASELINE_DIR,
    "--out", "NeurIPS/generated/tables.tex"
)

Run-Command -Name "Generate ablation tables" -CmdArgs @(
    "experiments/paper/generate_ablations_tex.py",
    "--results-dir", $RESULTS_DIR,
    "--out", "NeurIPS/generated/ablations.tex"
)

Run-Command -Name "Generate sweep tables (supplementary)" -CmdArgs @(
    "experiments/paper/generate_sweeps_tex.py",
    "--results-dir", $RESULTS_DIR,
    "--out", "NeurIPS/generated/sweeps.tex"
)

Run-Command -Name "Generate negative-results table (qualitative)" -CmdArgs @(
    "experiments/paper/generate_negative_results_tex.py",
    "--out", "NeurIPS/generated/negative_results.tex"
)

Write-Host ""
Write-Host "Done."
Write-Host "- Results: $RESULTS_DIR\"
Write-Host "- Paper tables: NeurIPS/generated/tables.tex"
Write-Host "- Compile paper: (cd NeurIPS && pdflatex neurips_2025_compressed.tex)"
