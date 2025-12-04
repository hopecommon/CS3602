#!/usr/bin/env python3
"""
运行所有实验的主脚本

包括:
1. WikiText-103 评估
2. PG19 评估  
3. Window size 消融实验
4. N_sink 消融实验
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """运行命令并打印输出"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"命令: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n错误: {description} 失败")
        return False
    
    print(f"\n✓ {description} 完成")
    return True


def main():
    # 设置环境变量
    import os
    hf_home = Path.cwd() / ".cache" / "huggingface"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    
    python_exe = sys.executable
    
    experiments = [
        # 1. WikiText-103 评估
        {
            "cmd": [
                python_exe,
                "experiments/eval_streaming_llm.py",
                "--dataset-name", "wikitext",
                "--dataset-config", "wikitext-103-v1",
                "--max-samples", "64",
                "--max-eval-tokens", "4096",
                "--n-sink", "4",
                "--window-size", "1024",
                "--output", "results/streaming_llm/wikitext_result.json"
            ],
            "description": "WikiText-103 评估"
        },
        
        # 2. PG19 评估
        {
            "cmd": [
                python_exe,
                "experiments/eval_streaming_llm.py",
                "--dataset-name", "pg19",
                "--dataset-config", "",
                "--max-samples", "1",
                "--max-eval-tokens", "8192",
                "--n-sink", "4",
                "--window-size", "1024",
                "--trust-remote-code",
                "--output", "results/streaming_llm/pg19_result.json"
            ],
            "description": "PG19 评估"
        },
        
        # 3. Window size 消融实验
        {
            "cmd": [
                python_exe,
                "experiments/ablation_study.py",
                "--ablation-type", "window_size",
                "--max-samples", "64",
                "--max-eval-tokens", "4096",
                "--output", "results/streaming_llm/ablation_window_size.json"
            ],
            "description": "Window Size 消融实验"
        },
        
        # 4. N_sink 消融实验
        {
            "cmd": [
                python_exe,
                "experiments/ablation_study.py",
                "--ablation-type", "n_sink",
                "--max-samples", "64",
                "--max-eval-tokens", "4096",
                "--output", "results/streaming_llm/ablation_n_sink.json"
            ],
            "description": "N_sink 消融实验"
        },
    ]
    
    print(f"\n{'='*60}")
    print(f"开始运行所有实验")
    print(f"{'='*60}")
    print(f"总共 {len(experiments)} 个实验")
    print(f"HF_HOME: {hf_home}")
    print(f"{'='*60}\n")
    
    success_count = 0
    failed_experiments = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] {exp['description']}")
        
        if run_command(exp['cmd'], exp['description']):
            success_count += 1
        else:
            failed_experiments.append(exp['description'])
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"实验总结")
    print(f"{'='*60}")
    print(f"成功: {success_count}/{len(experiments)}")
    
    if failed_experiments:
        print(f"\n失败的实验:")
        for exp in failed_experiments:
            print(f"  - {exp}")
    else:
        print(f"\n✓ 所有实验成功完成!")
    
    print(f"{'='*60}\n")
    
    # 生成图表
    if len(failed_experiments) == 0:
        print(f"\n{'='*60}")
        print(f"生成可视化图表")
        print(f"{'='*60}\n")
        
        plot_cmd = [python_exe, "experiments/plot_results.py"]
        plot_result = subprocess.run(plot_cmd, capture_output=False, text=True)
        
        if plot_result.returncode == 0:
            print(f"\n✓ 图表生成成功")
        else:
            print(f"\n⚠ 图表生成失败,但实验数据已保存")
    
    return len(failed_experiments) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)