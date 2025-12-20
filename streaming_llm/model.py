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


@dataclass
class TupleLayerWrapper:
    """Helper to wrap legacy tuple (key, value) as an object with .keys and .values"""
    keys: torch.Tensor
    values: torch.Tensor


class LegacyCacheWrapper:
    """Helper to wrap legacy tuple of tuples as a Cache-like object"""
    def __init__(self, past_key_values: tuple):
        self.layers = [TupleLayerWrapper(k, v) for k, v in past_key_values]

    def to_tuple(self) -> tuple:
        return tuple((l.keys, l.values) for l in self.layers)


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
        self._layer_infos = self._collect_layer_infos()
        self._cache_capacity = self._infer_cache_capacity()
        self._layer_key_buffers: List[Optional[torch.Tensor]] = []
        self._layer_value_buffers: List[Optional[torch.Tensor]] = []
        self._shift_rotation_cache: dict = {}

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
        return

    # ---------------------------------------------------------------------
    # Core logic
    # ---------------------------------------------------------------------
    def update(self, past_key_values: Cache | tuple | None):
        """
        Debug-enhanced StreamingLLM update().
        Prints key diagnostics to locate why PPL explodes (150+).
        Controlled by env var:
        - STREAMING_DEBUG=1
        - STREAMING_DEBUG_EVERY=200  (print every N updates)
        """
        def _dbg(msg: str):
            if os.environ.get("STREAMING_DEBUG", "0") == "1":
                print(msg, flush=True)

        dbg_every = int(os.environ.get("STREAMING_DEBUG_EVERY", "200"))

        if not self._enabled or past_key_values is None:
            return past_key_values

        # Handle legacy tuple format
        is_legacy = isinstance(past_key_values, tuple)
        if is_legacy:
            cache_obj = LegacyCacheWrapper(past_key_values)
        elif isinstance(past_key_values, Cache):
            cache_obj = past_key_values
        else:
            if not hasattr(past_key_values, "layers"):
                raise TypeError(
                    f"StreamingLLMWrapper received unsupported cache type: {type(past_key_values)}. "
                    "Expected transformers.Cache or tuple."
                )
            cache_obj = past_key_values

        if len(cache_obj.layers) == 0:
            return past_key_values

        first_layer = cache_obj.layers[0]
        if first_layer.keys.numel() == 0:
            return past_key_values

        device = first_layer.keys.device
        batch_size = first_layer.keys.shape[0]
        seq_len = first_layer.keys.shape[2]

        cache_limit = self._cache_capacity
        overflow = seq_len - cache_limit

        # --- debug header (occasionally) ---
        # Use an internal counter so we can print every N calls
        if not hasattr(self, "_dbg_step"):
            self._dbg_step = 0
        self._dbg_step += 1

        if self._dbg_step % dbg_every == 0:
            _dbg(
                f"[WRAP] step={self._dbg_step} enabled={self._enabled} "
                f"type={'tuple' if is_legacy else type(past_key_values)} "
                f"seq_len={seq_len} cache_limit={cache_limit} overflow={overflow} "
                f"compress_every={self.compress_every} supports_slice_api={self._supports_slice_api}"
            )

        if overflow <= 0:
            return past_key_values
        if overflow < self.compress_every:
            # not enough overflow to trigger a compress this call
            return past_key_values

        indices = None
        slice_info = None
        kept = None

        if self._supports_slice_api:
            slice_info = self.cache.get_slice_info(seq_len)
            sink_count, recent_start, total_len = slice_info
            kept = sink_count + max(0, seq_len - recent_start)

            if self._dbg_step % dbg_every == 0:
                _dbg(
                    f"[WRAP] slice_info: sink={sink_count}, recent_start={recent_start}, total_len={total_len}, kept={kept}, "
                    f"shift(sink-recent)={sink_count - recent_start}"
                )
        else:
            indices = self.cache.get_keep_indices(seq_len, device)
            if indices is None:
                return past_key_values
            old_positions_1d = indices.to(dtype=torch.long)
            kept = int(old_positions_1d.numel())

            if self._dbg_step % dbg_every == 0:
                _dbg(
                    f"[WRAP] keep_indices: kept={kept}, old_pos[0]={int(old_positions_1d[0])}, old_pos[-1]={int(old_positions_1d[-1])}"
                )

        # ---- per-layer compress + RoPE fix ----
        for layer_idx, layer in enumerate(cache_obj.layers):
            key_states = layer.keys
            value_states = layer.values

            # Basic shape sanity (assume [B,H,T,D])
            if key_states.ndim != 4:
                _dbg(f"[WRAP][ERROR] layer{layer_idx} key ndim={key_states.ndim} shape={tuple(key_states.shape)} (expected 4D [B,H,T,D])")

            if self._dbg_step % dbg_every == 0 and layer_idx == 0:
                _dbg(f"[WRAP] layer0 key shape={tuple(key_states.shape)} val shape={tuple(value_states.shape)} dtype={key_states.dtype}")

            if self._supports_slice_api and slice_info is not None:
                sink_count, recent_start, total_len = slice_info

                buffer_key = self._get_or_init_buffer(self._layer_key_buffers, layer_idx, key_states)
                compressed_key = self._slice_cache_tensor(
                    key_states, sink_count, recent_start, total_len, kept, buffer_key
                )
                buffer_value = self._get_or_init_buffer(self._layer_value_buffers, layer_idx, value_states)
                compressed_value = self._slice_cache_tensor(
                    value_states, sink_count, recent_start, total_len, kept, buffer_value
                )
            else:
                compressed_key = torch.index_select(key_states, 2, indices).contiguous()
                compressed_value = torch.index_select(value_states, 2, indices).contiguous()

            layer_info = self._layer_infos[min(layer_idx, len(self._layer_infos) - 1)]
            rotary_ndims = layer_info.rotary_ndims
            inv_freq = layer_info.inv_freq

            if self._dbg_step % dbg_every == 0 and layer_idx == 0:
                inv_shape = tuple(inv_freq.shape) if isinstance(inv_freq, torch.Tensor) else None
                _dbg(f"[WRAP] layer0 rotary_ndims={rotary_ndims} inv_freq_shape={inv_shape} head_dim={compressed_key.shape[-1]}")

            # Debug: measure whether RoPE fix actually changes something
            # Take one representative vector (layer0, head0, token=sink_count if exists) from rotary dims
            def _sample_rotary_vec(x: torch.Tensor, token_index: int, rotary_dim: int):
                # x: [B,H,T,D]
                t = x.shape[2]
                if t == 0:
                    return None
                ti = min(max(token_index, 0), t - 1)
                rd = min(int(rotary_dim), x.shape[-1])
                if rd <= 0:
                    return None
                return x[0, 0, ti, :rd].detach().float().cpu()

            if self._supports_slice_api and slice_info is not None:
                # Start+Recent pruning implies a constant delta for the recent segment:
                # delta = new_pos - old_pos = sink_count - recent_start
                if inv_freq is not None and rotary_ndims and recent_start < total_len:
                    shift = int(sink_count - recent_start)

                    if shift != 0:
                        rotary_dim = int(rotary_ndims)

                        # IMPORTANT: half uses rotary_dim//2; inv_freq is usually of length rotary_dim//2
                        half = min(rotary_dim, compressed_key.shape[-1]) // 2

                        if self._dbg_step % dbg_every == 0 and layer_idx == 0:
                            _dbg(f"[WRAP] layer0 shift-path: shift={shift} rotary_dim={rotary_dim} half={half}")

                        # sample before
                        before = None
                        if self._dbg_step % dbg_every == 0 and layer_idx == 0:
                            before = _sample_rotary_vec(compressed_key, sink_count, rotary_dim)

                        if half > 0:
                            cache_key = (id(inv_freq), shift, half, compressed_key.dtype, compressed_key.device)
                            cached = self._shift_rotation_cache.get(cache_key)

                            if cached is None:
                                inv = inv_freq.to(device=compressed_key.device, dtype=torch.float32)

                                # Debug: inv_freq length should be >= half
                                if self._dbg_step % dbg_every == 0 and layer_idx == 0:
                                    _dbg(f"[WRAP] layer0 inv_freq len={inv.numel()} need_half={half}")

                                inv = inv[:half]
                                angle = inv * float(shift)
                                cos = angle.cos().to(dtype=compressed_key.dtype)
                                sin = angle.sin().to(dtype=compressed_key.dtype)
                                cached = (cos, sin, rotary_dim)
                                self._shift_rotation_cache[cache_key] = cached

                            cos, sin, rotary_dim = cached

                            # apply in-place rotation to RECENT segment (starts at sink_count)
                            apply_rotary_shift_inplace(
                                compressed_key,
                                start_index=sink_count,
                                cos=cos,
                                sin=sin,
                                rotary_dim=rotary_dim,
                            )

                        # sample after
                        if self._dbg_step % dbg_every == 0 and layer_idx == 0 and before is not None:
                            after = _sample_rotary_vec(compressed_key, sink_count, rotary_dim)
                            if after is not None:
                                delta = (after - before).pow(2).sum().sqrt().item()
                                _dbg(f"[WRAP] layer0 shift-path rotary delta(L2)={delta:.6f}  (expect >0 if rotation applied)")
                    else:
                        if self._dbg_step % dbg_every == 0 and layer_idx == 0:
                            _dbg("[WRAP] layer0 shift == 0, skip apply_rotary_shift_inplace")
                else:
                    if self._dbg_step % dbg_every == 0 and layer_idx == 0:
                        _dbg("[WRAP] layer0 shift-path conditions not met; skip RoPE fix (THIS IS SUSPICIOUS if model uses RoPE)")
            else:
                # General gather path: rerotate based on old/new positions
                old_positions = old_positions_1d.unsqueeze(0).expand(batch_size, -1)
                new_positions = torch.arange(kept, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

                rotation = None
                if self.reuse_rotation and inv_freq is not None and rotary_ndims:
                    cache_key = (id(inv_freq), rotary_ndims, compressed_key.dtype, compressed_key.device)
                    rotation = self._shift_rotation_cache.get(cache_key)
                    if rotation is None:
                        rotation = build_rotation_cache(
                            old_positions=old_positions,
                            new_positions=new_positions,
                            inv_freq=inv_freq.to(device=device),
                            rotary_dim=rotary_ndims,
                            dtype=compressed_key.dtype,
                        )
                        self._shift_rotation_cache[cache_key] = rotation

                before = None
                if self._dbg_step % dbg_every == 0 and layer_idx == 0 and rotary_ndims:
                    before = _sample_rotary_vec(compressed_key, 0, int(rotary_ndims))

                compressed_key = rerotate_keys(
                    compressed_key,
                    old_positions,
                    new_positions,
                    inv_freq=inv_freq,
                    rotary_dim=rotary_ndims,
                    precomputed=rotation,
                )

                if self._dbg_step % dbg_every == 0 and layer_idx == 0 and before is not None:
                    after = _sample_rotary_vec(compressed_key, 0, int(rotary_ndims))
                    if after is not None:
                        delta = (after - before).pow(2).sum().sqrt().item()
                        _dbg(f"[WRAP] layer0 rerotate-path rotary delta(L2)={delta:.6f} (expect >0)")

            layer.keys = compressed_key
            layer.values = compressed_value

            if self._dbg_step % dbg_every == 0 and layer_idx == 0:
                _dbg(f"[WRAP] layer0 writeback keys shape={tuple(layer.keys.shape)} (expect T=={kept})")

        if is_legacy:
            return cache_obj.to_tuple()
        return past_key_values


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
