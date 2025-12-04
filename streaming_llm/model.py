"""
StreamingLLMWrapper: KV cache 管理器

负责在 HuggingFace 模型上实现 StreamingLLM 的压缩逻辑,并在压缩后重新
计算 RoPE 位置编码。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from torch import nn
from transformers.cache_utils import Cache

from .kv_cache import StreamingKVCache
from .rope_utils import rerotate_keys


@dataclass
class LayerInfo:
    """记录 attention 层的 RoPE 配置"""

    rotary_ndims: Optional[int]
    inv_freq: Optional[torch.Tensor]


class StreamingLLMWrapper:
    """
    包装 HuggingFace 模型,提供 StreamingLLM 所需的 KV cache 压缩功能

    使用方式:
        >>> wrapper = StreamingLLMWrapper(model, n_sink=4, window_size=1024)
        >>> with wrapper.enable():
        ...     outputs = model(...)
        ...     wrapper.update(outputs.past_key_values)
    """

    def __init__(
        self,
        model: nn.Module,
        n_sink: int = 4,
        window_size: int = 1024,
        cache: Optional[Any] = None,
    ):
        self.model = model
        self.n_sink = n_sink
        self.window_size = window_size
        self.cache = cache if cache is not None else StreamingKVCache(
            n_sink=n_sink,
            window_size=window_size,
        )
        self.cache_name = type(self.cache).__name__

        self._enabled = False
        self._positions: Optional[torch.Tensor] = None  # [batch, seq_len]
        self._layer_infos = self._collect_layer_infos()

    # ---------------------------------------------------------------------
    # Context manager helpers
    # ---------------------------------------------------------------------
    def enable(self):
        """启用 wrapper (context manager)"""
        return self

    def __enter__(self):
        self.reset()
        self._enabled = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset()
        self._enabled = False
        return False

    def reset(self):
        """重置内部状态"""
        self._positions = None

    # ---------------------------------------------------------------------
    # Core logic
    # ---------------------------------------------------------------------
    def update(self, past_key_values: Cache | None):
        """
        对当前的 past_key_values 执行 StreamingLLM 压缩,并重新计算 RoPE。
        """
        if not self._enabled or past_key_values is None:
            return

        if not isinstance(past_key_values, Cache):
            raise TypeError(
                "StreamingLLMWrapper 仅支持新的 Cache API, "
                "请使用 transformers>=4.36 的模型"
            )

        if len(past_key_values.layers) == 0:
            return

        first_layer = past_key_values.layers[0]
        if first_layer.keys.numel() == 0:
            return

        device = first_layer.keys.device
        batch_size = first_layer.keys.shape[0]
        seq_len = first_layer.keys.shape[2]

        self._ensure_position_buffer(batch_size, seq_len, device)

        indices = self.cache.get_keep_indices(seq_len, device)
        if indices is None:
            return

        old_positions = torch.index_select(self._positions, 1, indices)
        new_positions = torch.arange(
            indices.shape[0], device=device, dtype=old_positions.dtype
        ).unsqueeze(0)
        new_positions = new_positions.expand(batch_size, -1)

        for layer_idx, layer in enumerate(past_key_values.layers):
            key_states = layer.keys
            value_states = layer.values

            compressed_key = torch.index_select(key_states, 2, indices).contiguous()
            compressed_value = torch.index_select(value_states, 2, indices).contiguous()

            layer_info = self._layer_infos[min(layer_idx, len(self._layer_infos) - 1)]
            compressed_key = rerotate_keys(
                compressed_key,
                old_positions,
                new_positions,
                inv_freq=layer_info.inv_freq,
                rotary_dim=layer_info.rotary_ndims,
            )

            layer.keys = compressed_key
            layer.values = compressed_value

        self._positions = new_positions

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _ensure_position_buffer(self, batch_size: int, seq_len: int, device: torch.device):
        """
        初始化/扩展 position buffer,追踪 cache 内每个 token 的相对位置
        """
        if self._positions is None or self._positions.shape[0] != batch_size:
            self._positions = torch.arange(
                seq_len, device=device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1).clone()
            return

        current_len = self._positions.shape[1]
        if seq_len > current_len:
            extra = torch.arange(
                current_len,
                seq_len,
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)
            extra = extra.expand(batch_size, -1).clone()
            self._positions = torch.cat([self._positions, extra], dim=1)
        elif seq_len < current_len:
            self._positions = self._positions[:, :seq_len]

    def _collect_layer_infos(self) -> List[LayerInfo]:
        """
        收集每一层 attention 的 RoPE 配置
        当前主要支持 GPTNeoX (Pythia) 和 LLaMA 系列
        """
        infos: List[LayerInfo] = []

        if hasattr(self.model, "gpt_neox"):
            layers = self.model.gpt_neox.layers
            inv_freq = getattr(self.model.gpt_neox, "rotary_emb", None)
            inv_freq = getattr(inv_freq, "inv_freq", None)
            for layer in layers:
                attn = layer.attention
                infos.append(
                    LayerInfo(
                        rotary_ndims=getattr(attn, "rotary_ndims", None),
                        inv_freq=inv_freq,
                    )
                )
            return infos

        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
            for layer in layers:
                attn = getattr(layer, "self_attn", None)
                rotary_emb = getattr(attn, "rotary_emb", None) if attn else None
                infos.append(
                    LayerInfo(
                        rotary_ndims=getattr(attn, "rotary_ndims", None),
                        inv_freq=getattr(rotary_emb, "inv_freq", None),
                    )
                )
            return infos

        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            # GPT-2 / GPT-J 等无 RoPE 的模型,仅执行裁剪
            for _ in self.model.transformer.h:
                infos.append(LayerInfo(rotary_ndims=None, inv_freq=None))
            return infos

        raise ValueError(f"Unsupported model architecture: {type(self.model).__name__}")

    # ---------------------------------------------------------------------
    # Misc
    # ---------------------------------------------------------------------
    def get_compression_ratio(self, seq_len: int) -> float:
        return self.cache.get_compression_ratio(seq_len)

    def __repr__(self) -> str:
        return (
            f"StreamingLLMWrapper(\n"
            f"  model={type(self.model).__name__},\n"
            f"  n_sink={self.n_sink},\n"
            f"  window_size={self.window_size},\n"
            f"  enabled={self._enabled}\n"
            f")"
        )
