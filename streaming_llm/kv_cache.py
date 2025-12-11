"""
StreamingKVCache: 固定大小的 KV Cache 实现

实现 StreamingLLM 的核心压缩逻辑:
- 保留前 n_sink 个 token (attention sink)
- 保留最近 window_size 个 token
- 丢弃中间所有 token
"""

from typing import Optional, Tuple
import torch
from torch import Tensor


class StreamingKVCache:
    """
    固定大小的 KV Cache,实现 StreamingLLM 核心逻辑
    
    核心思想:
        原始序列: [sink_0, sink_1, ..., sink_n, middle_tokens..., recent_0, recent_1, ...]
        压缩序列: [sink_0, sink_1, ..., sink_n, recent_0, recent_1, ..., recent_m]
    
    Attributes:
        n_sink (int): sink token 数量,默认 4
        window_size (int): 滑动窗口大小,默认 1024
        max_size (int): 最大 cache 大小 = n_sink + window_size
    
    Example:
        >>> cache = StreamingKVCache(n_sink=4, window_size=1024)
        >>> compressed_key, compressed_value = cache.compress(key, value)
    """
    
    def __init__(self, n_sink: int = 4, window_size: int = 1024):
        """
        初始化 StreamingKVCache
        
        Args:
            n_sink: sink token 数量 (默认 4)
            window_size: 滑动窗口大小 (默认 1024)
        """
        if n_sink < 0:
            raise ValueError(f"n_sink must be non-negative, got {n_sink}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        
        self.n_sink = n_sink
        self.window_size = window_size
        self.max_size = n_sink + window_size

    @property
    def cache_size(self) -> int:
        """Total number of tokens retained after compression."""
        return self.max_size
    
    def get_keep_indices(self, seq_len: int, device: torch.device) -> Optional[Tensor]:
        """
        计算需要保留的 token 索引

        Args:
            seq_len: 当前 cache 序列长度
            device: 输出张量所在设备

        Returns:
            indices: [n_kept] 或 None (无需压缩)
        """
        if seq_len <= self.max_size:
            return None

        sink_count = min(self.n_sink, seq_len)
        window_count = min(self.window_size, max(seq_len - sink_count, 0))

        if window_count == 0:
            # 只需要 sinks, 但如果 seq_len <= max_size, 不会进入这里
            return torch.arange(sink_count, device=device)

        recent_start = seq_len - window_count
        # 确保 recent 部分不与 sink 重叠, overlap 会被 sink 覆盖
        recent_start = max(recent_start, sink_count)

        indices = []
        if sink_count > 0:
            indices.append(torch.arange(0, sink_count, device=device))
        indices.append(torch.arange(recent_start, seq_len, device=device))
        return torch.cat(indices, dim=0)

    def get_slice_info(self, seq_len: int) -> Tuple[int, int, int]:
        """
        Return (sink_count, recent_start, seq_len) for in-place slicing.
        """
        sink_count = min(self.n_sink, seq_len)
        window_count = min(self.window_size, max(seq_len - sink_count, 0))
        recent_start = seq_len - window_count
        recent_start = max(recent_start, sink_count)
        return sink_count, recent_start, seq_len

    def compress(
        self,
        key: Tensor,
        value: Tensor,
        indices: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        压缩 KV cache (保留 sinks + recent 窗口)

        Args:
            key: Key tensor [batch, heads, seq_len, head_dim]
            value: Value tensor [batch, heads, seq_len, head_dim]
            indices: 需要保留的索引,可选
        """
        if key.shape != value.shape:
            raise ValueError(
                f"Key and value shapes must match, got key: {key.shape}, value: {value.shape}"
            )

        seq_len = key.shape[2]
        if indices is None:
            indices = self.get_keep_indices(seq_len, key.device)

        if indices is None:
            return key, value

        compressed_key = torch.index_select(key, 2, indices).contiguous()
        compressed_value = torch.index_select(value, 2, indices).contiguous()
        return compressed_key, compressed_value
    
    def get_compression_ratio(self, original_seq_len: int) -> float:
        """
        计算压缩比
        
        Args:
            original_seq_len: 原始序列长度
        
        Returns:
            compression_ratio: 压缩比 (0-1 之间)
        """
        if original_seq_len <= self.max_size:
            return 0.0
        
        compressed_len = self.max_size
        return 1.0 - (compressed_len / original_seq_len)
    
    def __repr__(self) -> str:
        return (
            f"StreamingKVCache(n_sink={self.n_sink}, "
            f"window_size={self.window_size}, "
            f"max_size={self.max_size})"
        )
