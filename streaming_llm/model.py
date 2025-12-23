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
from .rope_utils import rerotate_keys, build_rotation_cache, apply_rotary_shift_inplace


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
        max_drop: int = 0,
        cache_slack: int = 0,
        overlap: int = 0,
        refresh_budget: int = 0,
        refresh_policy: str = "none",
        reuse_rotation: bool = True,
    ):
        self.model = model
        self.n_sink = n_sink
        self.window_size = window_size
        self.overlap = max(0, int(overlap))
        self.refresh_budget = max(0, int(refresh_budget))
        self.refresh_policy = refresh_policy
        cache_window = window_size + self.overlap
        self.cache = cache if cache is not None else StreamingKVCache(
            n_sink=n_sink,
            window_size=cache_window,
        )
        self.cache_name = type(self.cache).__name__
        self._supports_slice_api = hasattr(self.cache, "get_slice_info")
        # Allow KV to grow by up to (compress_every - 1) tokens before pruning.
        # compress_every=0 disables pruning (no eviction).
        compress_every = int(compress_every)
        if compress_every < 0:
            raise ValueError(f"compress_every must be >= 0, got {compress_every}")
        self.compress_every = compress_every
        max_drop = int(max_drop)
        if max_drop < 0:
            raise ValueError(f"max_drop must be >= 0, got {max_drop}")
        self.max_drop = max_drop
        cache_slack = int(cache_slack)
        if cache_slack < 0:
            raise ValueError(f"cache_slack must be >= 0, got {cache_slack}")
        self.cache_slack = cache_slack
        self.reuse_rotation = reuse_rotation

        self._enabled = False
        self._layer_infos = self._collect_layer_infos()
        self._soft_capacity = self._infer_cache_capacity() + self.refresh_budget
        self._hard_capacity = self._soft_capacity + self.cache_slack
        self._cache_capacity = self._hard_capacity
        self._layer_key_buffers: List[Optional[torch.Tensor]] = []
        self._layer_value_buffers: List[Optional[torch.Tensor]] = []
        self._shift_rotation_cache: dict = {}
        self._step = 0
        self._last_prune: Optional[dict] = None
        self._prune_events: List[dict] = []

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
        self._shift_rotation_cache.clear()
        self._step = 0
        self._last_prune = None
        self._prune_events.clear()
        return

    # ---------------------------------------------------------------------
    # Core logic
    # ---------------------------------------------------------------------
    def update(self, past_key_values: Cache | None):
        """
        对当前的 past_key_values 执行 StreamingLLM 压缩,并重新计算 RoPE。
        """
        if not self._enabled or past_key_values is None:
            return

        self._last_prune = None

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

        soft_limit = self._soft_capacity
        hard_limit = self._hard_capacity
        if self.compress_every == 0:
            self._step += 1
            return

        overflow_soft = seq_len - soft_limit
        if overflow_soft <= 0:
            self._step += 1
            return
        if overflow_soft < self.compress_every:
            self._step += 1
            return

        indices = None
        slice_info = None
        use_middle_refresh = self.refresh_policy == "middle"
        if self.max_drop > 0:
            target_len = max(seq_len - self.max_drop, soft_limit)
            target_len = min(target_len, hard_limit)
        else:
            target_len = soft_limit
        if self._supports_slice_api:
            sink_count = min(self.n_sink, seq_len)
            recent_keep = max(0, target_len - sink_count)
            recent_keep = min(recent_keep, max(0, seq_len - sink_count))
            recent_start_base = seq_len - recent_keep
            kept = sink_count + recent_keep
            slice_info = (sink_count, recent_start_base, seq_len)
        else:
            indices = self.cache.get_keep_indices(seq_len, device)
            if indices is None:
                return
            old_positions_1d = indices.to(dtype=torch.long)
            kept = int(old_positions_1d.numel())

        refresh_indices = None
        if self._supports_slice_api:
            window_count = max(0, seq_len - recent_start_base)
            refresh_budget_cap = window_count if use_middle_refresh else None
            refresh_indices = self._select_refresh_indices(
                sink_count=sink_count,
                recent_start=recent_start_base,
                seq_len=seq_len,
                device=device,
                max_budget=refresh_budget_cap,
            )
        if refresh_indices is not None and refresh_indices.numel() > 0:
            if use_middle_refresh:
                refresh_count = int(refresh_indices.numel())
                refresh_count = min(refresh_count, window_count)
                refresh_indices = refresh_indices[:refresh_count]
                kept = sink_count + window_count
            else:
                kept += int(refresh_indices.numel())
            if indices is not None:
                indices = torch.cat([indices, refresh_indices], dim=0)

        for layer_idx, layer in enumerate(past_key_values.layers):
            key_states = layer.keys
            value_states = layer.values

            if self._supports_slice_api and slice_info is not None:
                sink_count, recent_start_base, total_len = slice_info
                window_count = max(0, total_len - recent_start_base)
                refresh_count = int(refresh_indices.numel()) if refresh_indices is not None else 0
                if use_middle_refresh and refresh_count > 0:
                    refresh_count = min(refresh_count, window_count)
                    refresh_indices = refresh_indices[:refresh_count]
                    recent_keep = max(0, window_count - refresh_count)
                    recent_start = total_len - recent_keep
                    kept = sink_count + refresh_count + recent_keep
                else:
                    recent_keep = window_count
                    recent_start = recent_start_base
                    kept = sink_count + recent_keep + refresh_count

                buffer_key = self._get_or_init_buffer(
                    self._layer_key_buffers, layer_idx, key_states
                )
                buffer_value = self._get_or_init_buffer(
                    self._layer_value_buffers, layer_idx, value_states
                )
                if use_middle_refresh and refresh_count > 0:
                    view_key = buffer_key[:, :, :kept, :]
                    view_value = buffer_value[:, :, :kept, :]
                    offset = 0
                    if sink_count > 0:
                        view_key[:, :, :sink_count, :].copy_(key_states[:, :, :sink_count, :])
                        view_value[:, :, :sink_count, :].copy_(value_states[:, :, :sink_count, :])
                        offset = sink_count
                    refresh_key = torch.index_select(key_states, 2, refresh_indices)
                    refresh_value = torch.index_select(value_states, 2, refresh_indices)
                    new_positions_1d = torch.arange(
                        offset,
                        offset + refresh_count,
                        device=view_key.device,
                        dtype=torch.long,
                    )
                    layer_info = self._layer_infos[min(layer_idx, len(self._layer_infos) - 1)]
                    refresh_key = self._rerotate_refresh_keys(
                        refresh_key,
                        old_positions_1d=refresh_indices.to(dtype=torch.long),
                        new_positions_1d=new_positions_1d,
                        layer_info=layer_info,
                    )
                    view_key[:, :, offset : offset + refresh_count, :].copy_(refresh_key)
                    view_value[:, :, offset : offset + refresh_count, :].copy_(refresh_value)
                    offset += refresh_count
                    if recent_start < total_len and offset < kept:
                        recent_slice = key_states[:, :, recent_start:total_len, :]
                        view_key[:, :, offset:kept, :].copy_(recent_slice[:, :, : kept - offset, :])
                        recent_value = value_states[:, :, recent_start:total_len, :]
                        view_value[:, :, offset:kept, :].copy_(recent_value[:, :, : kept - offset, :])
                    compressed_key = view_key
                    compressed_value = view_value
                else:
                    kept_base = sink_count + recent_keep
                    compressed_key = self._slice_cache_tensor(
                        key_states, sink_count, recent_start, total_len, kept_base, buffer_key
                    )
                    compressed_value = self._slice_cache_tensor(
                        value_states, sink_count, recent_start, total_len, kept_base, buffer_value
                    )
            else:
                compressed_key = torch.index_select(key_states, 2, indices).contiguous()
                compressed_value = torch.index_select(value_states, 2, indices).contiguous()

            layer_info = self._layer_infos[min(layer_idx, len(self._layer_infos) - 1)]
            if self._supports_slice_api and slice_info is not None:
                # Start+Recent pruning implies a constant delta for the recent segment:
                # delta = new_pos - old_pos = sink_count - recent_start
                if (
                    layer_info.inv_freq is not None
                    and layer_info.rotary_ndims
                    and recent_start < total_len
                ):
                    shift_start = sink_count + (refresh_count if use_middle_refresh else 0)
                    shift = int(shift_start - recent_start)
                    if shift != 0:
                        rotary_dim = int(layer_info.rotary_ndims)
                        half = min(rotary_dim, compressed_key.shape[-1]) // 2
                        if half > 0:
                            cache_key = (
                                id(layer_info.inv_freq),
                                shift,
                                half,
                                compressed_key.dtype,
                                compressed_key.device,
                            )
                            cached = self._shift_rotation_cache.get(cache_key)
                            if cached is None:
                                inv = layer_info.inv_freq.to(
                                    device=compressed_key.device, dtype=torch.float32
                                )[:half]
                                angle = inv * float(shift)
                                cos = angle.cos().to(dtype=compressed_key.dtype)
                                sin = angle.sin().to(dtype=compressed_key.dtype)
                                cached = (cos, sin, rotary_dim)
                                self._shift_rotation_cache[cache_key] = cached
                            cos, sin, rotary_dim = cached
                            apply_rotary_shift_inplace(
                                compressed_key,
                                start_index=shift_start,
                                cos=cos,
                                sin=sin,
                                rotary_dim=rotary_dim,
                            )
                if refresh_count > 0 and refresh_indices is not None and not use_middle_refresh:
                    refresh_key = torch.index_select(key_states, 2, refresh_indices)
                    refresh_value = torch.index_select(value_states, 2, refresh_indices)
                    new_positions_1d = torch.arange(
                        kept_base,
                        kept_base + refresh_count,
                        device=compressed_key.device,
                        dtype=torch.long,
                    )
                    refresh_key = self._rerotate_refresh_keys(
                        refresh_key,
                        old_positions_1d=refresh_indices.to(dtype=torch.long),
                        new_positions_1d=new_positions_1d,
                        layer_info=layer_info,
                    )
                    buffer_key[:, :, kept_base:kept_base + refresh_count, :].copy_(refresh_key)
                    buffer_value[:, :, kept_base:kept_base + refresh_count, :].copy_(refresh_value)
                    compressed_key = buffer_key[:, :, :kept, :]
                    compressed_value = buffer_value[:, :, :kept, :]
            else:
                old_positions = old_positions_1d.unsqueeze(0).expand(batch_size, -1)
                new_positions = (
                    torch.arange(kept, device=device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
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
                        compressed_key.device,
                    )
                    rotation = self._shift_rotation_cache.get(cache_key)
                    if rotation is None:
                        rotation = build_rotation_cache(
                            old_positions=old_positions,
                            new_positions=new_positions,
                            inv_freq=layer_info.inv_freq.to(device=device),
                            rotary_dim=layer_info.rotary_ndims,
                            dtype=compressed_key.dtype,
                        )
                        self._shift_rotation_cache[cache_key] = rotation
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

        dropped = max(0, seq_len - kept)
        self._last_prune = {
            "step": self._step,
            "seq_len": seq_len,
            "kept": kept,
            "dropped": dropped,
            "overflow": overflow_soft,
            "compress_every": self.compress_every,
            "max_drop": self.max_drop,
            "cache_slack": self.cache_slack,
        }
        self._prune_events.append(self._last_prune)
        self._step += 1

    def pop_last_prune(self) -> Optional[dict]:
        """Return and clear the last prune event (if any)."""
        last = self._last_prune
        self._last_prune = None
        return last

    def get_prune_events(self) -> List[dict]:
        """Return all prune events observed so far."""
        return list(self._prune_events)

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

    def _select_refresh_indices(
        self,
        sink_count: int,
        recent_start: int,
        seq_len: int,
        device: torch.device,
        max_budget: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if self.refresh_budget <= 0:
            return None
        if self.refresh_policy == "none":
            return None
        if recent_start <= sink_count:
            return None
        drop_len = max(0, recent_start - sink_count)
        if drop_len <= 0:
            return None
        budget = min(self.refresh_budget, drop_len)
        if max_budget is not None:
            budget = min(budget, max_budget)
        if budget <= 0:
            return None
        if self.refresh_policy in {"uniform", "middle"}:
            if budget == 1:
                index = sink_count + drop_len // 2
                return torch.tensor([index], device=device, dtype=torch.long)
            positions = torch.linspace(
                sink_count,
                recent_start - 1,
                steps=budget,
                device=device,
            )
            indices = positions.round().to(dtype=torch.long)
            return torch.unique_consecutive(indices)
        raise ValueError(f"Unsupported refresh_policy: {self.refresh_policy}")

    def _rerotate_refresh_keys(
        self,
        refresh_keys: torch.Tensor,
        old_positions_1d: torch.Tensor,
        new_positions_1d: torch.Tensor,
        layer_info: LayerInfo,
    ) -> torch.Tensor:
        if layer_info.inv_freq is None or not layer_info.rotary_ndims:
            return refresh_keys
        batch_size = refresh_keys.shape[0]
        old_positions = old_positions_1d.unsqueeze(0).expand(batch_size, -1)
        new_positions = new_positions_1d.unsqueeze(0).expand(batch_size, -1)
        return rerotate_keys(
            refresh_keys,
            old_positions=old_positions,
            new_positions=new_positions,
            inv_freq=layer_info.inv_freq,
            rotary_dim=layer_info.rotary_ndims,
            precomputed=None,
        )

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
        if seq_len <= self._cache_capacity:
            return 0.0
        return 1.0 - (self._cache_capacity / seq_len)

    def __repr__(self) -> str:
        return (
            f"StreamingLLMWrapper(\n"
            f"  model={type(self.model).__name__},\n"
            f"  n_sink={self.n_sink},\n"
            f"  window_size={self.window_size},\n"
            f"  compress_every={self.compress_every},\n"
            f"  max_drop={self.max_drop},\n"
            f"  cache_slack={self.cache_slack},\n"
            f"  enabled={self._enabled}\n"
            f")"
        )
