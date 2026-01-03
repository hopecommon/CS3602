import subprocess
import sys
import os

def run_task(name, dataset_path, eval_tokens, prefix_tokens):
    python_exe = sys.executable
    # 获取当前脚本所在目录的绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建目标脚本的绝对路径
    script_ppl = os.path.join(base_dir, "examples", "eval_long_ppl.py")
    script_bench = os.path.join(base_dir, "examples", "benchmark_streaming.py")
    
    # 转换 dataset_path 为绝对路径
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.abspath(os.path.join(base_dir, dataset_path))

    print(f"\n{'#'*60}")
    print(f"Running Experiment: {name}")
    print(f"Dataset: {dataset_path}")
    print(f"{'#'*60}\n")

    # --- Phase 1: PPL Evaluation ---
    print(f">>> Phase 1: Running PPL Evaluation ({name})...")
    if not os.path.exists(script_ppl):
        print(f"Error: Script not found at {script_ppl}")
        return

    # 设置环境变量以防止递归，并传递 PYTHONPATH 确保能找到 streaming_llm 包
    env = os.environ.copy()
    env["RUNNING_EXPERIMENT_CHILD"] = "1"
    env["PYTHONPATH"] = base_dir + os.pathsep + env.get("PYTHONPATH", "")

    cmd_ppl = [
        python_exe, script_ppl,
        "--model_name_or_path", "EleutherAI/pythia-2.8b",
        "--dataset_path", dataset_path, # 优先使用本地文件
        "--enable_start_recent_kv_cache",
        "--start_size", "32",
        "--recent_size", "2016",
        "--num_eval_tokens", str(eval_tokens),
        "--output_dir", os.path.join(base_dir, "results", f"{name}_local_test")
    ]
    
    try:
        subprocess.run(cmd_ppl, check=True, env=env, cwd=base_dir)
    except subprocess.CalledProcessError as e:
        print(f"PPL Evaluation failed for {name} with exit code {e.returncode}")

    # --- Phase 2: Speed Benchmark ---
    print(f"\n>>> Phase 2: Running Speed Benchmark ({name})...")
    if not os.path.exists(script_bench):
        print(f"Error: Script not found at {script_bench}")
        return

    # 确保 prefix_tokens 合理
    bench_prefix = min(prefix_tokens, eval_tokens - 1000) if eval_tokens > 1000 else 100
    bench_prefix = max(bench_prefix, 100)

    cmd_bench = [
        python_exe, script_bench,
        "--model_name_or_path", "EleutherAI/pythia-2.8b",
        "--mode", "streaming",
        "--start_size", "32",
        "--recent_size", "2016",
        "--gen_tokens", "1000",       
        "--prefix_tokens", str(bench_prefix),   
        "--data_json", dataset_path,
        "--data_text_key", "text"
    ]
    
    try:
        subprocess.run(cmd_bench, check=True, env=env, cwd=base_dir)
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed for {name} with exit code {e.returncode}")


def main():
    # 防止递归调用的安全检查
    if os.environ.get("RUNNING_EXPERIMENT_CHILD") == "1":
        print("Error: Detected recursive call! The script is calling itself instead of the target scripts.")
        sys.exit(1)

    # 定义实验列表
    experiments = [
        {
            "name": "pg19_20k",
            "path": "../data/pg19/long_context_20000.json",
            "eval_tokens": 20000,
            "prefix_tokens": 19000 
        },
        {
            "name": "wikitext_4k",
            "path": "../data/wikitext/long_context_4096.json",
            "eval_tokens": 4096,
            "prefix_tokens": 3000
        },
        {
            "name": "wikitext_8k",
            "path": "../data/wikitext/long_context_8192.json",
            "eval_tokens": 8192,
            "prefix_tokens": 7000
        }
    ]

    print("Starting Batch Experiments...")
    
    for exp in experiments:
        # 简单检查文件是否存在
        # 注意：这里我们还没有转换成绝对路径，但在 run_task 中会转换
        # 为了避免报错，这里先尝试用 dataset_path 判断
        run_task(
            name=exp["name"],
            dataset_path=exp["path"],
            eval_tokens=exp["eval_tokens"],
            prefix_tokens=exp["prefix_tokens"]
        )

    print("\n" + "="*60)
    print("All Batch Experiments Completed!")
    print("Results saved in 'results/' directory.")
    print("="*60)

if __name__ == "__main__":
    main()
