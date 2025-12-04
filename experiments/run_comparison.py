#!/usr/bin/env python3
"""
Run Comparison Experiments

Run both our implementation and kvpress implementation with the same parameters
for direct comparison.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print its output"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run comparison experiments between our implementation and kvpress"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        choices=["wikitext", "pg19"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--n-sink",
        type=int,
        default=4,
        help="Number of sink tokens"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1024,
        help="Window size"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=64,
        help="Maximum number of samples"
    )
    parser.add_argument(
        "--max-eval-tokens",
        type=int,
        default=4096,
        help="Maximum evaluation tokens"
    )
    
    args = parser.parse_args()
    
    # Setup dataset parameters
    if args.dataset == "wikitext":
        dataset_name = "wikitext"
        dataset_config = "wikitext-103-v1"
        text_column = "text"
        trust_remote_code = False
    else:  # pg19
        dataset_name = "pg19"
        dataset_config = None
        text_column = "text"
        trust_remote_code = True
    
    # Common parameters
    common_params = [
        "--dataset-name", dataset_name,
        "--split", "test",
        "--text-column", text_column,
        "--max-samples", str(args.max_samples),
        "--max-eval-tokens", str(args.max_eval_tokens),
        "--max-length", "1024",
        "--stride", "512",
        "--n-sink", str(args.n_sink),
        "--window-size", str(args.window_size),
    ]
    
    if dataset_config:
        common_params.extend(["--dataset-config", dataset_config])
    
    if trust_remote_code:
        common_params.append("--trust-remote-code")
    
    # Output paths
    our_output = f"results/streaming_llm/{args.dataset}_comparison.json"
    kvpress_output = f"results/kvpress/{args.dataset}_comparison.json"
    
    print(f"\n{'='*60}")
    print(f"Running Comparison Experiments")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"n_sink: {args.n_sink}")
    print(f"window_size: {args.window_size}")
    print(f"max_samples: {args.max_samples}")
    print(f"max_eval_tokens: {args.max_eval_tokens}")
    print(f"{'='*60}\n")
    
    # Run our implementation
    our_cmd = [
        sys.executable,
        "experiments/eval_streaming_llm.py",
        *common_params,
        "--output", our_output,
    ]
    
    if not run_command(our_cmd, "Running Our StreamingLLM Implementation"):
        print("Failed to run our implementation")
        return 1
    
    # Run kvpress implementation
    kvpress_cmd = [
        sys.executable,
        "experiments/eval_kvpress.py",
        *common_params,
        "--output", kvpress_output,
    ]
    
    if not run_command(kvpress_cmd, "Running kvpress StreamingLLM Implementation"):
        print("Failed to run kvpress implementation")
        return 1
    
    print(f"\n{'='*60}")
    print(f"Comparison Experiments Completed!")
    print(f"{'='*60}")
    print(f"Our results: {our_output}")
    print(f"kvpress results: {kvpress_output}")
    print(f"\nYou can now generate comparison plots with:")
    print(f"  python experiments/plot_comparison.py")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())