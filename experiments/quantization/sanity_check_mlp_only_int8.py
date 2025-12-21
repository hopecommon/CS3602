#!/usr/bin/env python3
"""
Sanity checks for MLP-only INT8:
1) Locate first layer where NaN/Inf appears.
2) Check whether loss stays finite when computed in fp32.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "experiments"))

from eval_utils import load_tokenized_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity checks for MLP-only INT8 quantization"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.environ.get("MODEL_NAME", "EleutherAI/pythia-2.8b"),
        help="Model name"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="wikitext",
        help="Dataset name"
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="wikitext-103-v1",
        help="Dataset config"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Text column"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help="Max samples"
    )
    parser.add_argument(
        "--max-eval-tokens",
        type=int,
        default=1024,
        help="Max eval tokens"
    )
    parser.add_argument(
        "--prefill-len",
        type=int,
        default=512,
        help="Prefill length for the sanity check"
    )
    parser.add_argument(
        "--print-quantized",
        action="store_true",
        help="Print which modules are quantized"
    )
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Disable quantization (baseline sanity check)"
    )
    parser.add_argument(
        "--int8-config",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="INT8 weight-only config version (v2 uses IntxWeightOnlyConfig)"
    )
    return parser.parse_args()


def _mlp_filter(module: torch.nn.Module, fqn: str) -> bool:
    if not isinstance(module, torch.nn.Linear):
        return False
    return ("mlp.dense_h_to_4h" in fqn) or ("mlp.dense_4h_to_h" in fqn)


def _quantize_mlp_only(
    model: torch.nn.Module,
    *,
    config_version: str,
    print_modules: bool = False,
) -> int:
    try:
        from torchao.quantization import Int8WeightOnlyConfig, IntxWeightOnlyConfig, quantize_
    except Exception as exc:
        raise RuntimeError("torchao is required for MLP-only INT8 quantization.") from exc

    if print_modules:
        for name, module in model.named_modules():
            if _mlp_filter(module, name):
                print(f"quantize: {name}")

    if config_version == "v2":
        config = IntxWeightOnlyConfig(weight_dtype=torch.int8, version=2)
    else:
        config = Int8WeightOnlyConfig()
    quantize_(model, config, filter_fn=_mlp_filter)
    return sum(1 for name, module in model.named_modules() if _mlp_filter(module, name))


def _first_nonfinite_layer(model, input_ids: torch.Tensor) -> Optional[str]:
    if not hasattr(model, "gpt_neox"):
        return None

    bad_layer: Optional[str] = None
    handles = []

    def _hook(name):
        def _inner(module, inputs, output):
            nonlocal bad_layer
            if bad_layer is not None:
                return
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(tensor):
                if not torch.isfinite(tensor).all():
                    bad_layer = name
        return _inner

    for idx, layer in enumerate(model.gpt_neox.layers):
        handles.append(layer.register_forward_hook(_hook(f"gpt_neox.layers.{idx}")))

    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=False)

    for handle in handles:
        handle.remove()

    return bad_layer


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_mapping[args.dtype]
    if device.type == "cpu" and torch_dtype != torch.float32:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded_dataset = load_tokenized_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column=args.text_column,
        max_samples=args.max_samples,
        tokenizer=tokenizer,
        max_eval_tokens=args.max_eval_tokens,
        trust_remote_code=args.trust_remote_code,
    )
    input_ids = encoded_dataset[:, : args.prefill_len].to(device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()

    if args.no_quant:
        quantized_count = 0
        print("quantization disabled")
    else:
        quantized_count = _quantize_mlp_only(
            model,
            config_version=args.int8_config,
            print_modules=args.print_quantized,
        )
        print(f"quantized modules: {quantized_count}")

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits
    logits_finite = torch.isfinite(logits).all().item()
    print(f"logits finite: {logits_finite}")

    if not logits_finite:
        bad_layer = _first_nonfinite_layer(model, input_ids)
        print(f"first non-finite layer: {bad_layer}")
    else:
        bad_layer = None

    # Sanity check: fp32 loss
    if input_ids.size(1) >= 2:
        labels = input_ids[:, 1:]
        pred = logits[:, :-1, :]
        loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)).float(), labels.reshape(-1))
        print(f"fp32 loss finite: {torch.isfinite(loss).item()}, loss={loss.item():.6f}")
    else:
        print("prefill_len < 2, skip fp32 loss check")


if __name__ == "__main__":
    main()
