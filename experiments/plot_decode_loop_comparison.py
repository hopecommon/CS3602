#!/usr/bin/env python3
"""
Generate comparison plots for decode-loop evaluations.

Reads JSON files from results/comprehensive/{dataset}_{method}_decode_loop.json
and outputs runtime / PPL bar charts.
"""

from __future__ import annotations

import json
from math import ceil
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt

ROOT = Path("results/comprehensive")
OUTPUT_DIR = Path("results/figures")
DATASETS = ["wikitext", "pg19_20k"]
METHODS = ["baseline", "ours", "mit"]


def load_decode_loop(dataset: str, method: str) -> Tuple[float, float]:
    path = ROOT / f"{dataset}_{method}_decode_loop.json"
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    return float(data["total_time"]), float(data["perplexity"])


def plot_grouped_bars(values: Dict[str, Dict[str, float]], title: str, ylabel: str, filename: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    num_groups = len(DATASETS)
    num_methods = len(METHODS)
    width = 0.15
    x = range(num_groups)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, method in enumerate(METHODS):
        method_vals = [values[dataset][method] for dataset in DATASETS]
        offset = (i - (num_methods - 1) / 2) * width
        ax.bar([xi + offset for xi in x], method_vals, width, label=method.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close(fig)


def main():
    runtime_values: Dict[str, Dict[str, float]] = {}
    ppl_values: Dict[str, Dict[str, float]] = {}
    for dataset in DATASETS:
        runtime_values[dataset] = {}
        ppl_values[dataset] = {}
        for method in METHODS:
            runtime, ppl = load_decode_loop(dataset, method)
            runtime_values[dataset][method] = runtime
            ppl_values[dataset][method] = ppl

    plot_grouped_bars(runtime_values, "Decode-loop Runtime Comparison", "Runtime (s)", "decode_loop_runtime_comparison.png")
    plot_grouped_bars(ppl_values, "Decode-loop Perplexity Comparison", "PPL", "decode_loop_ppl_comparison.png")


if __name__ == "__main__":
    main()
