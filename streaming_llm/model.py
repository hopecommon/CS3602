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
from .rope_utils import rerotate_keys, build_rotation_cache


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
        compress_every: int = 4,
        reuse_rotation: bool = True,
    ):
        self.model = model
        self.n_sink = n_sink
        self.window_size = window_size
        self.cache = cache if cache is not None else StreamingKVCache(
            n_sink=n_sink,
            window_size=window_size,
        )
        self.cache_name = type(self.cache).__name__
        self._supports_slice_api = hasattr(self.cache, "get_slice_info")
        self.compress_every = max(1, int(compress_every))
        self.reuse_rotation = reuse_rotation

        self._enabled = False
        self._positions: Optional[torch.Tensor] = None  # [batch, seq_len]
        self._layer_infos = self._collect_layer_infos()
        self._cache_capacity = self._infer_cache_capacity()
        self._layer_key_buffers: List[Optional[torch.Tensor]] = []
        self._layer_value_buffers: List[Optional[torch.Tensor]] = []

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

        cache_limit = self._cache_capacity
        overflow = seq_len - cache_limit
        if overflow <= 0:
            self._ensure_position_buffer(batch_size, seq_len, device)
            return
        if overflow < self.compress_every:
            self._ensure_position_buffer(batch_size, seq_len, device)
            return

        self._ensure_position_buffer(batch_size, seq_len, device)

        indices = None
        slice_info = None
        if self._supports_slice_api:
            slice_info = self.cache.get_slice_info(seq_len)
            sink_count, recent_start, _ = slice_info
            old_positions = self._gather_start_recent_positions(
                sink_count, recent_start, seq_len
            )
            kept = old_positions.shape[1]
        else:
            indices = self.cache.get_keep_indices(seq_len, device)
            if indices is None:
                return
            old_positions = torch.index_select(self._positions, 1, indices)
            kept = indices.shape[0]

        new_positions = torch.arange(
            kept, device=device, dtype=old_positions.dtype
        ).unsqueeze(0)
        new_positions = new_positions.expand(batch_size, -1)

        rotation_cache = {}
        for layer_idx, layer in enumerate(past_key_values.layers):
            key_states = layer.keys
            value_states = layer.values

            if self._supports_slice_api and slice_info is not None:
                sink_count, recent_start, total_len = slice_info
                buffer_key = self._get_or_init_buffer(
                    self._layer_key_buffers, layer_idx, key_states
                )
                compressed_key = self._slice_cache_tensor(
                    key_states, sink_count, recent_start, total_len, kept, buffer_key
                )
                buffer_value = self._get_or_init_buffer(
                    self._layer_value_buffers, layer_idx, value_states
                )
                compressed_value = self._slice_cache_tensor(
                    value_states, sink_count, recent_start, total_len, kept, buffer_value
                )
            else:
                compressed_key = torch.index_select(key_states, 2, indices).contiguous()
                compressed_value = torch.index_select(value_states, 2, indices).contiguous()

            layer_info = self._layer_infos[min(layer_idx, len(self._layer_infos) - 1)]
            rotation = None
            if (
                self.reuse_rotation
                and layer_info.inv_freq is not None
                and layer_info.rotary_ndims
            ):
                cache_key = (
                    id(layer_info.inv_freq),
                    layer_info.rotary_ndims,
                    compressed_key.dtype,
                )
                rotation = rotation_cache.get(cache_key)
                if rotation is None:
                    rotation = build_rotation_cache(
                        old_positions=old_positions,
                        new_positions=new_positions,
                        inv_freq=layer_info.inv_freq.to(device=device),
                        rotary_dim=layer_info.rotary_ndims,
                        dtype=compressed_key.dtype,
                    )
                    rotation_cache[cache_key] = rotation
            compressed_key = rerotate_keys(
                compressed_key,
                old_positions,
                new_positions,
                inv_freq=layer_info.inv_freq,
                rotary_dim=layer_info.rotary_ndims,
                precomputed=rotation,
            )

            layer.keys = compressed_key
            layer.values = compressed_value

        self._positions = new_positions

    def _slice_cache_tensor(
        self,
        tensor: torch.Tensor,
        sink_count: int,
        recent_start: int,
        seq_len: int,
        kept: int,
        buffer: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Slice sink and recent regions (MIT-style) without gather.
        """
        parts = []
        if sink_count > 0:
            parts.append(tensor[:, :, :sink_count, :])
        if recent_start < seq_len:
            parts.append(tensor[:, :, recent_start:seq_len, :])
        if not parts:
            return tensor.new_empty(
                tensor.shape[0], tensor.shape[1], 0, tensor.shape[-1]
            )
        if len(parts) == 1 and buffer is None:
            return parts[0].contiguous()
        if buffer is None:
            return torch.cat(parts, dim=2).contiguous()

        view = buffer[:, :, :kept, :]
        offset = 0
        if sink_count > 0:
            length = min(sink_count, kept)
            view[:, :, :length, :].copy_(tensor[:, :, :length, :])
            offset += length
        if recent_start < seq_len and offset < kept:
            recent_slice = tensor[:, :, recent_start:seq_len, :]
            length = min(recent_slice.shape[2], kept - offset)
            view[:, :, offset : offset + length, :].copy_(recent_slice[:, :, :length, :])
            offset += length
        return view[:, :, :kept, :]
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
    def _infer_cache_capacity(self) -> int:
        for attr in ("cache_size", "max_size", "max_cache_size"):
            if hasattr(self.cache, attr):
                return int(getattr(self.cache, attr))
        raise ValueError(f"Cache {self.cache_name} must expose cache_size/max_size.")

    def _gather_start_recent_positions(
        self, sink_count: int, recent_start: int, seq_len: int
    ) -> torch.Tensor:
        parts = []
        if sink_count > 0:
            parts.append(self._positions[:, :sink_count])
        if recent_start < seq_len:
            parts.append(self._positions[:, recent_start:seq_len])
        if not parts:
            return self._positions.new_empty(self._positions.shape[0], 0)
        if len(parts) == 1:
            return parts[0].clone()
        return torch.cat(parts, dim=1).clone()

    def _get_or_init_buffer(
        self,
        store: List[Optional[torch.Tensor]],
        layer_idx: int,
        template: torch.Tensor,
    ) -> torch.Tensor:
        while len(store) <= layer_idx:
            store.append(None)
        buf = store[layer_idx]
        needed_shape = (
            template.shape[0],
            template.shape[1],
            self._cache_capacity,
            template.shape[-1],
        )
        if (
            buf is None
            or buf.shape != needed_shape
            or buf.dtype != template.dtype
            or buf.device != template.device
        ):
            buf = torch.empty(needed_shape, dtype=template.dtype, device=template.device)
            store[layer_idx] = buf
        return buf

    def get_compression_ratio(self, seq_len: int) -> float:
        return self.cache.get_compression_ratio(seq_len)

    def __repr__(self) -> str:
        return (
            f"StreamingLLMWrapper(\n"
            f"  model={type(self.model).__name__},\n"
            f"  n_sink={self.n_sink},\n"
            f"  window_size={self.window_size},\n"
            f"  compress_every={self.compress_every},\n"
            f"  enabled={self._enabled}\n"
            f")"
        )
