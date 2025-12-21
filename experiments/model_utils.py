"""
Model utilities for quantization and compilation.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM

try:
    from transformers import TorchAoConfig
except Exception:  # pragma: no cover - optional dependency
    TorchAoConfig = None


def build_torchao_config(quantization: str):
    if quantization in {None, "", "none"}:
        return None
    if TorchAoConfig is None:
        raise RuntimeError("TorchAoConfig is unavailable; upgrade transformers.")
    try:
        from torchao.quantization import (
            Int8WeightOnlyConfig,
            Int8DynamicActivationInt8WeightConfig,
            Int4WeightOnlyConfig,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torchao is required for quantization configs.") from exc

    quantization = quantization.lower()
    if quantization == "int8wo":
        quant_config = Int8WeightOnlyConfig()
    elif quantization == "int8da":
        quant_config = Int8DynamicActivationInt8WeightConfig()
    elif quantization == "int4wo":
        quant_config = Int4WeightOnlyConfig(group_size=128)
    else:
        raise ValueError(f"Unknown quantization option: {quantization}")

    return TorchAoConfig(quant_type=quant_config)


def load_model_with_options(
    model_name: str,
    torch_dtype: torch.dtype,
    device: torch.device,
    quantization: str = "none",
    trust_remote_code: bool = False,
):
    quant_config = build_torchao_config(quantization)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
        trust_remote_code=trust_remote_code,
    ).to(device)
    model.eval()
    return model


def maybe_compile_model(
    model,
    use_compile: bool,
    compile_mode: str = "reduce-overhead",
    allow_cudagraphs: bool = False,
) -> Tuple[object, Optional[str]]:
    if not use_compile:
        return model, None
    if not hasattr(torch, "compile"):
        return model, "torch.compile is unavailable in this PyTorch build"
    options = None
    if not allow_cudagraphs:
        os.environ.setdefault("TORCHINDUCTOR_DISABLE_CUDAGRAPHS", "1")
        try:
            import torch._dynamo.config as dynamo_config  # type: ignore
            dynamo_config.cudagraphs = False
            import torch._inductor.config as inductor_config  # type: ignore
            inductor_config.triton.cudagraphs = False
            inductor_config.triton.cudagraph_trees = False
            inductor_config.cuda_graphs = False
        except Exception:
            pass
        options = {
            "triton.cudagraphs": False,
            "triton.cudagraph_trees": False,
        }
    try:
        if options is not None:
            try:
                compiled = torch.compile(model, mode=compile_mode, options=options)
            except TypeError:
                if compile_mode == "reduce-overhead":
                    compiled = torch.compile(model, mode="default")
                    return compiled, (
                        "torch.compile options unsupported; fell back to mode=default "
                        "to avoid cudagraphs"
                    )
                compiled = torch.compile(model, mode=compile_mode)
        else:
            compiled = torch.compile(model, mode=compile_mode)
    except Exception as exc:  # pragma: no cover - best effort
        return model, f"torch.compile failed: {exc}"
    return compiled, None
