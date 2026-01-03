import subprocess
import sys
import os

def main():
    # 定义基础参数
    python_exe = sys.executable
    script_path = os.path.join("examples", "eval_long_ppl.py")
    
    # 检查脚本是否存在
    if not os.path.exists(script_path):
        print(f"Error: Could not find script at {script_path}")
        return

    # 构建命令
    # 注意：这里调用的是 examples/eval_long_ppl.py
    # 该文件此前已修改，支持 --dataset_path 并能显示实时速度和显存
    cmd = [
        python_exe, script_path,
        "--model_name_or_path", "EleutherAI/pythia-2.8b",
        "--dataset_path", "../data/pg19/long_context_20000.json",
        "--enable_start_recent_kv_cache",
        "--start_size", "32",
        "--recent_size", "2016",
        "--num_eval_tokens", "50000",
        "--output_dir", "results/pg19_local_test"
    ]

    print("=" * 50)
    print("Running StreamingLLM Evaluation (PPL & Speed)")
    print("Command:", " ".join(cmd))
    print("=" * 50)

    try:
        # 调用子进程执行官方示例脚本
        subprocess.run(cmd, check=True)
        print("\nExperiment finished successfully!")
        print("Results saved in 'results/pg19_local_test/'")
    except subprocess.CalledProcessError as e:
        print(f"\nExperiment failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")

if __name__ == "__main__":
    main()
