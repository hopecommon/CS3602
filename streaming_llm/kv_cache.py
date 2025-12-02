"""
StreamingKVCache: 固定大小的 KV Cache 实现

实现 StreamingLLM 的核心压缩逻辑:
- 保留前 n_sink 个 token (attention sink)
- 保留最近 window_size 个 token
- 丢弃中间所有 token
"""

from typing import Tuple
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
    
    def compress(
        self, 
        key: Tensor, 
        value: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        压缩 KV cache
        
        Args:
            key: Key tensor, shape [batch, n_heads, seq_len, head_dim]
            value: Value tensor, shape [batch, n_heads, seq_len, head_dim]
        
        Returns:
            compressed_key: 压缩后的 key
            compressed_value: 压缩后的 value
        
        Raises:
            ValueError: 如果 key 和 value 的形状不匹配
        """
        if key.shape != value.shape:
            raise ValueError(
                f"Key and value shapes must match, got key: {key.shape}, value: {value.shape}"
            )
        
        seq_len = key.shape[2]
        
        # 如果序列长度小于等于最大大小,不需要压缩
        if seq_len <= self.max_size:
            return key, value
        
        # 保留 sink tokens (前 n_sink 个)
        if self.n_sink > 0:
            sink_key = key[:, :, :self.n_sink, :]
            sink_value = value[:, :, :self.n_sink, :]
        
        # 保留 recent tokens (最后 window_size 个)
        recent_key = key[:, :, -self.window_size:, :]
        recent_value = value[:, :, -self.window_size:, :]
        
        # 拼接 sink + recent
        if self.n_sink > 0:
            compressed_key = torch.cat([sink_key, recent_key], dim=2)
            compressed_value = torch.cat([sink_value, recent_value], dim=2)
        else:
            compressed_key = recent_key
            compressed_value = recent_value
        
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