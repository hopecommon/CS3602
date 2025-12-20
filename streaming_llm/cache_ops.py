"""
Cache helpers for slicing and committing past_key_values.

These utilities keep speculative decoding and StreamingLLM pruning
aligned without directly depending on wrapper internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple, List

import torch

try:
    from transformers.cache_utils import Cache
except Exception:  # pragma: no cover - transformers may not be available
    Cache = None  # type: ignore


@dataclass
class _LayerView:
    keys: torch.Tensor
    values: torch.Tensor


def _normalize_past(
    past_key_values: Any,
) -> Tuple[List[_LayerView], Callable[[], Any]]:
    if past_key_values is None:
        return [], lambda: None

    if Cache is not None and isinstance(past_key_values, Cache):
        return past_key_values.layers, lambda: past_key_values

    if hasattr(past_key_values, "layers"):
        return past_key_values.layers, lambda: past_key_values

    if isinstance(past_key_values, tuple):
        layers = [_LayerView(k, v) for k, v in past_key_values]

        def to_tuple() -> tuple:
            return tuple((l.keys, l.values) for l in layers)

        return layers, to_tuple

    raise TypeError(
        f"Unsupported past_key_values type: {type(past_key_values)}"
    )


def _infer_seq_dim(tensor: torch.Tensor) -> int:
    if tensor.dim() == 4:
        return 2
    if tensor.dim() == 3:
        return 1
    raise ValueError(f"Unsupported cache tensor rank: {tensor.dim()}")


def get_past_seq_len(past_key_values: Any) -> int:
    layers, _ = _normalize_past(past_key_values)
    if not layers:
        return 0
    seq_dim = _infer_seq_dim(layers[0].keys)
    return int(layers[0].keys.shape[seq_dim])


def slice_past(past_key_values: Any, keep_len: int) -> Any:
    layers, to_output = _normalize_past(past_key_values)
    if not layers:
        return to_output()

    if keep_len <= 0:
        keep_len = 0

    for layer in layers:
        seq_dim = _infer_seq_dim(layer.keys)
        if layer.keys.shape[seq_dim] <= keep_len:
            continue
        if seq_dim == 2:
            layer.keys = layer.keys[:, :, :keep_len, :].contiguous()
            layer.values = layer.values[:, :, :keep_len, :].contiguous()
        elif seq_dim == 1:
            layer.keys = layer.keys[:, :keep_len, :].contiguous()
            layer.values = layer.values[:, :keep_len, :].contiguous()

    return to_output()


def commit_and_prune(
    past_key_values: Any,
    accepted_len: int,
    proposed_len: int,
    streaming_wrapper: Any = None,
) -> Any:
    if past_key_values is None:
        return None

    seq_len = get_past_seq_len(past_key_values)
    trim = max(0, proposed_len - accepted_len)
    keep_len = max(0, seq_len - trim)

    if keep_len < seq_len:
        past_key_values = slice_past(past_key_values, keep_len)

    if streaming_wrapper is not None:
        past_key_values = streaming_wrapper.update(past_key_values)

    return past_key_values
