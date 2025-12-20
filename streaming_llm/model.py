"""
StreamingLLMWrapper: KV cache 管理器

负责在 HuggingFace 模型上实现 StreamingLLM 的压缩逻辑,并在压缩后重新
计算 RoPE 位置编码。

修复点：
1) 压缩触发：按 step 每 compress_every 步触发（避免 overflow<compress_every 永不压缩）
2) GPT-NeoX/Pythia RoPE 配置：
   - rotary_ndims 从 attn.rotary_ndims 或 rotary_pct*head_dim 推导
   - inv_freq 若无法从模型中取到，则按 RoPE 公式自行生成（base 默认 10000 或从 config 取 rope_theta/rotary_emb_base）
3) Debug 输出：
   - STREAMING_DEBUG=1
   - STREAMING_DEBUG_EVERY=200
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache

from .kv_cache import StreamingKVCache
from .rope_utils import rerotate_keys, build_rotation_cache, apply_rotary_shift_inplace


def _dbg(msg: str):
    if os.environ.get("STREAMING_DEBUG", "0") == "1":
        print(msg, flush=True)


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
        self._cache_capacity = self._infer_cache_capacity()

        # Buffer reuse
        self._layer_key_buffers: List[Optional[torch.Tensor]] = []
        self._layer_value_buffers: List[Optional[torch.Tensor]] = []
        self._shift_rotation_cache: dict = {}

        # ✅ cache for computed inv_freq: (rotary_dim, base, device, dtype) -> tensor
        self._computed_inv_freq: Dict[Tuple[int, float, torch.device], torch.Tensor] = {}

        # step counter
        self._step = 0

        # collect RoPE info (may compute inv_freq if missing)
        self._layer_infos = self._collect_layer_infos()

    # ---------------------------------------------------------------------
    # Context manager helpers
    # ---------------------------------------------------------------------
    def enable(self):
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
        self._shift_rotation_cache.clear()
        self._step = 0
        return

    # ---------------------------------------------------------------------
    # RoPE helpers
    # ---------------------------------------------------------------------
    def _get_rope_base(self) -> float:
        """
        Try to infer RoPE base/theta from config.
        Common names:
          - rope_theta (LLaMA-like)
          - rotary_emb_base (GPTNeoX-like in some versions)
        Default to 10000.0 if missing.
        """
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return 10000.0
        for name in ("rope_theta", "rotary_emb_base", "rotary_embedding_base", "rotary_base"):
            if hasattr(cfg, name):
                try:
                    return float(getattr(cfg, name))
                except Exception:
                    pass
        return 10000.0

    def _build_inv_freq(self, rotary_dim: int, device: torch.device) -> torch.Tensor:
        """
        Build inv_freq for RoPE:
          inv_freq[i] = 1 / base^(2i/rotary_dim), i=0..rotary_dim/2-1
        """
        base = self._get_rope_base()
        key = (int(rotary_dim), float(base), device)
        cached = self._computed_inv_freq.get(key)
        if cached is not None:
            return cached

        half = rotary_dim // 2
        # must be float32 for stability during trig, later cast
        inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32, device=device) * (2.0 / rotary_dim)))
        self._computed_inv_freq[key] = inv_freq
        return inv_freq

    # ---------------------------------------------------------------------
    # Core logic
    # ---------------------------------------------------------------------
    def update(self, past_key_values: Cache | tuple | None):
        if not self._enabled or past_key_values is None:
            return past_key_values

        dbg_every = int(os.environ.get("STREAMING_DEBUG_EVERY", "200"))

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

        self._step += 1
        if (self._step % dbg_every) == 0:
            _dbg(
                f"[WRAP] step={self._step} enabled={self._enabled} type={'tuple' if is_legacy else type(past_key_values)} "
                f"seq_len={seq_len} cache_limit={cache_limit} overflow={overflow} compress_every={self.compress_every} "
                f"supports_slice_api={self._supports_slice_api}"
            )

        if overflow <= 0:
            return past_key_values

        # ✅ Trigger compression by step frequency
        if (self._step % self.compress_every) != 0:
            return past_key_values

        indices = None
        slice_info = None
        kept = None

        if self._supports_slice_api:
            slice_info = self.cache.get_slice_info(seq_len)
            sink_count, recent_start, total_len = slice_info
            kept = sink_count + max(0, seq_len - recent_start)
            if (self._step % dbg_every) == 0:
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
            if (self._step % dbg_every) == 0:
                _dbg(
                    f"[WRAP] keep_indices: kept={kept}, old_pos[0]={int(old_positions_1d[0])}, old_pos[-1]={int(old_positions_1d[-1])}"
                )

        for layer_idx, layer in enumerate(cache_obj.layers):
            key_states = layer.keys
            value_states = layer.values

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

            # ✅ If inv_freq is missing but rotary_ndims exists, generate it on the fly
            inv_freq = layer_info.inv_freq
            if inv_freq is None and layer_info.rotary_ndims is not None and int(layer_info.rotary_ndims) > 0:
                inv_freq = self._build_inv_freq(int(layer_info.rotary_ndims), device=device)

            if (self._step % dbg_every) == 0 and layer_idx == 0:
                inv_shape = tuple(inv_freq.shape) if isinstance(inv_freq, torch.Tensor) else None
                _dbg(
                    f"[WRAP] layer0 key shape={tuple(key_states.shape)} -> {tuple(compressed_key.shape)} "
                    f"rotary_ndims={layer_info.rotary_ndims} inv_freq_shape={inv_shape} head_dim={compressed_key.shape[-1]}"
                )

            # ---- RoPE fix ----
            if self._supports_slice_api and slice_info is not None:
                if (
                    inv_freq is not None
                    and layer_info.rotary_ndims is not None
                    and int(layer_info.rotary_ndims) > 0
                    and recent_start < total_len
                ):
                    shift = int(sink_count - recent_start)
                    if shift != 0:
                        rotary_dim = int(layer_info.rotary_ndims)
                        half = min(rotary_dim, compressed_key.shape[-1]) // 2
                        if half > 0:
                            cache_key = (
                                id(inv_freq),  # note: inv_freq may be computed
                                shift,
                                half,
                                compressed_key.dtype,
                                compressed_key.device,
                            )
                            cached = self._shift_rotation_cache.get(cache_key)
                            if cached is None:
                                inv = inv_freq.to(device=compressed_key.device, dtype=torch.float32)[:half]
                                angle = inv * float(shift)
                                cos = angle.cos().to(dtype=compressed_key.dtype)
                                sin = angle.sin().to(dtype=compressed_key.dtype)
                                cached = (cos, sin, rotary_dim)
                                self._shift_rotation_cache[cache_key] = cached
                            cos, sin, rotary_dim = cached
                            apply_rotary_shift_inplace(
                                compressed_key,
                                start_index=sink_count,
                                cos=cos,
                                sin=sin,
                                rotary_dim=rotary_dim,
                            )
            else:
                # (rare in your run; you are on slice_api path)
                old_positions = old_positions_1d.unsqueeze(0).expand(batch_size, -1)
                new_positions = (
                    torch.arange(kept, device=device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                rotation = None
                if (
                    self.reuse_rotation
                    and inv_freq is not None
                    and layer_info.rotary_ndims is not None
                    and int(layer_info.rotary_ndims) > 0
                ):
                    cache_key = (
                        id(inv_freq),
                        int(layer_info.rotary_ndims),
                        compressed_key.dtype,
                        compressed_key.device,
                    )
                    rotation = self._shift_rotation_cache.get(cache_key)
                    if rotation is None:
                        rotation = build_rotation_cache(
                            old_positions=old_positions,
                            new_positions=new_positions,
                            inv_freq=inv_freq.to(device=device),
                            rotary_dim=int(layer_info.rotary_ndims),
                            dtype=compressed_key.dtype,
                        )
                        self._shift_rotation_cache[cache_key] = rotation
                compressed_key = rerotate_keys(
                    compressed_key,
                    old_positions,
                    new_positions,
                    inv_freq=inv_freq,
                    rotary_dim=layer_info.rotary_ndims,
                    precomputed=rotation,
                )

            layer.keys = compressed_key
            layer.values = compressed_value

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
        parts = []
        if sink_count > 0:
            parts.append(tensor[:, :, :sink_count, :])
        if recent_start < seq_len:
            parts.append(tensor[:, :, recent_start:seq_len, :])
        if not parts:
            return tensor.new_empty(tensor.shape[0], tensor.shape[1], 0, tensor.shape[-1])
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
            view[:, :, offset: offset + length, :].copy_(recent_slice[:, :, :length, :])
            offset += length
        return view[:, :, :kept, :]

    # ---------------------------------------------------------------------
    # RoPE config collection
    # ---------------------------------------------------------------------
    def _collect_layer_infos(self) -> List[LayerInfo]:
        infos: List[LayerInfo] = []

        # --- GPT-NeoX (Pythia) ---
        if hasattr(self.model, "gpt_neox"):
            layers = self.model.gpt_neox.layers

            # Try to locate inv_freq, may be missing -> we'll compute later anyway.
            inv_freq = None
            rotary_emb = getattr(self.model.gpt_neox, "rotary_emb", None)
            if rotary_emb is not None:
                inv_freq = getattr(rotary_emb, "inv_freq", None)

            if inv_freq is None and len(layers) > 0:
                att0 = layers[0].attention
                re0 = getattr(att0, "rotary_emb", None)
                if re0 is not None:
                    inv_freq = getattr(re0, "inv_freq", None)

            for layer in layers:
                attn = layer.attention

                # Determine head_dim
                head_dim = None
                if hasattr(attn, "head_size"):
                    head_dim = int(getattr(attn, "head_size"))
                elif hasattr(attn, "hidden_size") and hasattr(attn, "num_attention_heads"):
                    head_dim = int(attn.hidden_size) // int(attn.num_attention_heads)
                elif hasattr(self.model, "config") and hasattr(self.model.config, "hidden_size") and hasattr(self.model.config, "num_attention_heads"):
                    head_dim = int(self.model.config.hidden_size) // int(self.model.config.num_attention_heads)

                rotary_ndims = getattr(attn, "rotary_ndims", None)

                # Fallback: compute from rotary_pct
                if rotary_ndims is None:
                    rotary_pct = getattr(attn, "rotary_pct", None)
                    if rotary_pct is None and hasattr(self.model, "config") and hasattr(self.model.config, "rotary_pct"):
                        rotary_pct = getattr(self.model.config, "rotary_pct")
                    if rotary_pct is not None and head_dim is not None:
                        rotary_ndims = int(round(float(rotary_pct) * float(head_dim)))
                        rotary_ndims = rotary_ndims - (rotary_ndims % 2)

                # Final fallback: full head_dim
                if rotary_ndims is None and head_dim is not None:
                    rotary_ndims = int(head_dim)
                    rotary_ndims = rotary_ndims - (rotary_ndims % 2)

                # per-layer inv_freq fallback
                layer_inv = inv_freq
                if layer_inv is None:
                    re = getattr(attn, "rotary_emb", None)
                    if re is not None:
                        layer_inv = getattr(re, "inv_freq", None)

                infos.append(LayerInfo(rotary_ndims=rotary_ndims, inv_freq=layer_inv))

            if os.environ.get("STREAMING_DEBUG", "0") == "1" and len(infos) > 0:
                inv_len = None
                if infos[0].inv_freq is not None:
                    try:
                        inv_len = int(infos[0].inv_freq.numel())
                    except Exception:
                        inv_len = None
                _dbg(
                    f"[WRAP] collect_layer_infos GPTNeoX: layers={len(infos)} "
                    f"rotary_ndims[0]={infos[0].rotary_ndims} inv_freq_len={inv_len} rope_base={self._get_rope_base()}"
                )

            return infos

        # --- LLaMA-like ---
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

        # --- GPT2/GPTJ (no RoPE) ---
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
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
