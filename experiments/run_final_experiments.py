#!/usr/bin/env python3
"""
统一实验运行脚本

自动运行所有缺失的实验,包括:
1. WikiText-103 主实验 (Baseline + StreamingLLM)
2. PG19 主实验 (Baseline + StreamingLLM)
3. Window size 消融实验 (128, 256, 512, 1024, 2048, 4096)
4. N_sink 消融实验 (0, 1, 2, 4, 8, 16)

特性:
- 自动检测已完成的实验并跳过
- 详细的进度显示和日志输出
- 自动保存结果到 results/final/
- 错误处理和恢复
- 估算总运行时间
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming_llm import StreamingLLMWrapper
from eval_utils import (
    load_tokenized_dataset,
    compute_perplexity,
)

DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "EleutherAI/pythia-2.8b")


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        output_dir: Path = Path("results/final"),
        skip_existing: bool = True,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.skip_existing = skip_existing
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 实验统计
        self.total_experiments = 0
        self.completed_experiments = 0
        self.skipped_experiments = 0
        self.failed_experiments = 0
        self.start_time = None
        
        # 模型和数据缓存
        self.model = None
        self.tokenizer = None
        self.datasets = {}
    
    def log(self, message: str, level: str = "INFO"):
        """打印日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def load_model(self):
        """加载模型和tokenizer"""
        if self.model is None:
            self.log(f"加载模型: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
            ).to(self.device)
            self.model.eval()
            self.log("模型加载完成")
    
    def load_dataset(
        self,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        max_samples: int = 64,
        max_eval_tokens: int = 4096,
    ):
        """加载数据集(带缓存)"""
        cache_key = f"{dataset_name}:{dataset_config}:{max_samples}:{max_eval_tokens}"
        
        if cache_key not in self.datasets:
            self.log(f"加载数据集: {dataset_name}:{dataset_config}")
            encoded_dataset = load_tokenized_dataset(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                split="test",
                text_column="text",
                max_samples=max_samples,
                tokenizer=self.tokenizer,
                max_eval_tokens=max_eval_tokens,
            )
            self.datasets[cache_key] = encoded_dataset
            self.log(f"数据集加载完成: {encoded_dataset.shape[1]} tokens")
        
        return self.datasets[cache_key]
    
    def experiment_exists(self, output_path: Path) -> bool:
        """检查实验结果是否已存在"""
        if not self.skip_existing:
            return False
        
        if output_path.exists():
            try:
                data = json.loads(output_path.read_text())
                # 检查是否有有效的结果
                if "results" in data or "baseline" in data:
                    return True
            except:
                pass
        
        return False
    
    def save_result(self, result: Dict, output_path: Path):
        """保存实验结果"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
        self.log(f"结果已保存: {output_path}")
    
    def run_baseline_experiment(
        self,
        dataset_name: str,
        dataset_config: Optional[str],
        max_samples: int,
        max_eval_tokens: int,
        max_length: int = 1024,
        stride: int = 512,
        max_cache_size: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """运行基线实验"""
        encoded_dataset = self.load_dataset(
            dataset_name, dataset_config, max_samples, max_eval_tokens
        )
        
        stats = compute_perplexity(
            model=self.model,
            encoded_dataset=encoded_dataset,
            device=self.device,
            max_length=max_length,
            stride=stride,
            use_streaming=False,
            max_cache_size=max_cache_size,
        )
        
        return stats.perplexity, stats.runtime_sec, stats.prefill_sec
    
    def run_streaming_experiment(
        self,
        dataset_name: str,
        dataset_config: Optional[str],
        max_samples: int,
        max_eval_tokens: int,
        n_sink: int,
        window_size: int,
        max_length: int = 1024,
        stride: int = 512,
    ) -> Tuple[float, float, float, float]:
        """运行StreamingLLM实验"""
        encoded_dataset = self.load_dataset(
            dataset_name, dataset_config, max_samples, max_eval_tokens
        )
        
        wrapper = StreamingLLMWrapper(
            model=self.model,
            n_sink=n_sink,
            window_size=window_size
        )
        
        stats = compute_perplexity(
            model=self.model,
            encoded_dataset=encoded_dataset,
            device=self.device,
            max_length=max_length,
            stride=stride,
            use_streaming=True,
            streaming_wrapper=wrapper,
            max_cache_size=wrapper.cache.max_size,
        )
        compression_ratio = wrapper.get_compression_ratio(encoded_dataset.shape[1])
        
        return stats.perplexity, stats.runtime_sec, stats.prefill_sec, compression_ratio
    
    def run_main_experiments(self):
        """运行主实验 (WikiText-103 和 PG19)"""
        self.log("="*60)
        self.log("开始主实验")
        self.log("="*60)
        
        experiments = [
            {
                "name": "WikiText-103",
                "dataset_name": "wikitext",
                "dataset_config": "wikitext-103-v1",
                "max_samples": 64,
                "max_eval_tokens": 4096,
                "output": self.output_dir / "wikitext_main.json",
            },
            {
                "name": "PG19",
                "dataset_name": "pg19",
                "dataset_config": None,
                "max_samples": 1,
                "max_eval_tokens": 8192,
                "output": self.output_dir / "pg19_main.json",
            },
        ]
        
        for exp in experiments:
            self.total_experiments += 1
            
            if self.experiment_exists(exp["output"]):
                self.log(f"跳过已完成的实验: {exp['name']}")
                self.skipped_experiments += 1
                continue
            
            try:
                self.log(f"运行实验: {exp['name']}")
                
                # 运行基线
                self.log(f"  评估基线...")
                baseline_ppl, baseline_time, baseline_prefill = self.run_baseline_experiment(
                    dataset_name=exp["dataset_name"],
                    dataset_config=exp["dataset_config"],
                    max_samples=exp["max_samples"],
                    max_eval_tokens=exp["max_eval_tokens"],
                    max_cache_size=4 + 1024,
                )
                self.log(f"  基线 PPL: {baseline_ppl:.2f}, Runtime: {baseline_time:.3f}s")
                
                # 运行StreamingLLM
                self.log(f"  评估 StreamingLLM...")
                streaming_ppl, streaming_time, streaming_prefill, compression = \
                    self.run_streaming_experiment(
                        dataset_name=exp["dataset_name"],
                        dataset_config=exp["dataset_config"],
                        max_samples=exp["max_samples"],
                        max_eval_tokens=exp["max_eval_tokens"],
                        n_sink=4,
                        window_size=1024,
                    )
                self.log(f"  StreamingLLM PPL: {streaming_ppl:.2f}, Runtime: {streaming_time:.3f}s")
                
                # 计算指标
                speedup = baseline_time / streaming_time if streaming_time > 0 else 0
                ppl_increase = ((streaming_ppl - baseline_ppl) / baseline_ppl) * 100
                
                # 保存结果
                result = {
                    "experiment": exp["name"],
                    "model": self.model_name,
                    "dataset": f"{exp['dataset_name']}:{exp['dataset_config']}",
                    "max_samples": exp["max_samples"],
                    "max_eval_tokens": exp["max_eval_tokens"],
                    "streaming_llm": {
                        "n_sink": 4,
                        "window_size": 1024,
                        "max_cache_size": 4 + 1024,
                    },
                    "baseline": {
                        "perplexity": baseline_ppl,
                        "runtime_sec": baseline_time,
                        "prefill_sec": baseline_prefill,
                    },
                    "streaming": {
                        "perplexity": streaming_ppl,
                        "runtime_sec": streaming_time,
                        "prefill_sec": streaming_prefill,
                    },
                    "metrics": {
                        "speedup": speedup,
                        "compression_ratio": compression,
                        "ppl_increase_percent": ppl_increase,
                    },
                    "timestamp": datetime.now().isoformat(),
                }
                
                self.save_result(result, exp["output"])
                self.completed_experiments += 1
                self.log(f"✓ 实验完成: {exp['name']} (加速比: {speedup:.2f}x)")
                
            except Exception as e:
                self.log(f"✗ 实验失败: {exp['name']} - {str(e)}", "ERROR")
                self.failed_experiments += 1
    
    def run_ablation_window_size(self):
        """运行 window_size 消融实验"""
        self.log("="*60)
        self.log("开始 Window Size 消融实验")
        self.log("="*60)
        
        output_path = self.output_dir / "ablation_window_size.json"
        
        if self.experiment_exists(output_path):
            self.log("跳过已完成的 Window Size 消融实验")
            self.skipped_experiments += 1
            return
        
        self.total_experiments += 1
        
        try:
            window_sizes = [128, 256, 512, 1024, 2048, 4096]
            n_sink = 4
            results = []
            
            for window_size in window_sizes:
                self.log(f"测试 window_size={window_size}")
                
                ppl, runtime, prefill, compression = self.run_streaming_experiment(
                    dataset_name="wikitext",
                    dataset_config="wikitext-103-v1",
                    max_samples=64,
                    max_eval_tokens=4096,
                    n_sink=n_sink,
                    window_size=window_size,
                )
                
                result = {
                    "window_size": window_size,
                    "n_sink": n_sink,
                    "max_cache_size": n_sink + window_size,
                    "perplexity": ppl,
                    "runtime_sec": runtime,
                    "prefill_sec": prefill,
                    "compression_ratio": compression,
                }
                results.append(result)
                
                self.log(f"  PPL: {ppl:.2f}, Runtime: {runtime:.3f}s, "
                        f"Compression: {compression:.2%}")
            
            # 保存结果
            output_data = {
                "experiment": "Window Size Ablation",
                "model": self.model_name,
                "dataset": "wikitext:wikitext-103-v1",
                "ablation_type": "window_size",
                "fixed_n_sink": n_sink,
                "results": results,
                "timestamp": datetime.now().isoformat(),
            }
            
            self.save_result(output_data, output_path)
            self.completed_experiments += 1
            self.log("✓ Window Size 消融实验完成")
            
        except Exception as e:
            self.log(f"✗ Window Size 消融实验失败: {str(e)}", "ERROR")
            self.failed_experiments += 1
    
    def run_ablation_n_sink(self):
        """运行 n_sink 消融实验"""
        self.log("="*60)
        self.log("开始 N_sink 消融实验")
        self.log("="*60)
        
        output_path = self.output_dir / "ablation_n_sink.json"
        
        if self.experiment_exists(output_path):
            self.log("跳过已完成的 N_sink 消融实验")
            self.skipped_experiments += 1
            return
        
        self.total_experiments += 1
        
        try:
            n_sinks = [0, 1, 2, 4, 8, 16]
            window_size = 1024
            results = []
            
            for n_sink in n_sinks:
                self.log(f"测试 n_sink={n_sink}")
                
                ppl, runtime, prefill, compression = self.run_streaming_experiment(
                    dataset_name="wikitext",
                    dataset_config="wikitext-103-v1",
                    max_samples=64,
                    max_eval_tokens=4096,
                    n_sink=n_sink,
                    window_size=window_size,
                )
                
                result = {
                    "n_sink": n_sink,
                    "window_size": window_size,
                    "max_cache_size": n_sink + window_size,
                    "perplexity": ppl,
                    "runtime_sec": runtime,
                    "prefill_sec": prefill,
                    "compression_ratio": compression,
                }
                results.append(result)
                
                self.log(f"  PPL: {ppl:.2f}, Runtime: {runtime:.3f}s, "
                        f"Compression: {compression:.2%}")
            
            # 保存结果
            output_data = {
                "experiment": "N_sink Ablation",
                "model": self.model_name,
                "dataset": "wikitext:wikitext-103-v1",
                "ablation_type": "n_sink",
                "fixed_window_size": window_size,
                "results": results,
                "timestamp": datetime.now().isoformat(),
            }
            
            self.save_result(output_data, output_path)
            self.completed_experiments += 1
            self.log("✓ N_sink 消融实验完成")
            
        except Exception as e:
            self.log(f"✗ N_sink 消融实验失败: {str(e)}", "ERROR")
            self.failed_experiments += 1
    
    def estimate_time(self) -> str:
        """估算剩余时间"""
        if self.completed_experiments == 0:
            return "未知"
        
        elapsed = time.time() - self.start_time
        avg_time_per_exp = elapsed / self.completed_experiments
        remaining_exps = self.total_experiments - self.completed_experiments - self.skipped_experiments
        estimated_remaining = avg_time_per_exp * remaining_exps
        
        return str(timedelta(seconds=int(estimated_remaining)))
    
    def print_summary(self):
        """打印实验总结"""
        elapsed = time.time() - self.start_time
        
        self.log("="*60)
        self.log("实验总结")
        self.log("="*60)
        self.log(f"总实验数: {self.total_experiments}")
        self.log(f"已完成: {self.completed_experiments}")
        self.log(f"已跳过: {self.skipped_experiments}")
        self.log(f"失败: {self.failed_experiments}")
        self.log(f"总耗时: {timedelta(seconds=int(elapsed))}")
        self.log(f"结果目录: {self.output_dir}")
        self.log("="*60)
    
    def run_all(self):
        """运行所有实验"""
        self.start_time = time.time()
        
        self.log("="*60)
        self.log("StreamingLLM 统一实验运行器")
        self.log("="*60)
        self.log(f"模型: {self.model_name}")
        self.log(f"设备: {self.device}")
        self.log(f"数据类型: {self.torch_dtype}")
        self.log(f"输出目录: {self.output_dir}")
        self.log(f"跳过已完成: {self.skip_existing}")
        self.log("="*60)
        
        # 加载模型
        self.load_model()
        
        # 运行所有实验
        self.run_main_experiments()
        self.run_ablation_window_size()
        self.run_ablation_n_sink()
        
        # 打印总结
        self.print_summary()


def parse_args():
    parser = argparse.ArgumentParser(
        description="统一实验运行脚本 - 运行所有 StreamingLLM 实验"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="模型名称"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/final"),
        help="输出目录"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="不跳过已完成的实验(重新运行所有)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备 (cuda/cpu, 默认自动检测)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    runner = ExperimentRunner(
        model_name=args.model_name,
        output_dir=args.output_dir,
        skip_existing=not args.no_skip_existing,
        device=args.device,
    )
    
    runner.run_all()


if __name__ == "__main__":
    main()
