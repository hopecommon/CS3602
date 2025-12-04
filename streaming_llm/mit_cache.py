"""
MIT-style Start+Recent KV cache helper

This cache mirrors the slicing logic used in MIT's StreamingLLM reference:
always keep the first `start_size` tokens plus the most recent `recent_size`
tokens when trimming the KV cache.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


class StartRecentKVCache:
    """
    Start+Recent cache selector for StreamingLLM.

    Attributes:
        start_size: Number of sink tokens to always keep.
        recent_size: Number of most recent tokens to keep.
        cache_size: Total cache size = start_size + recent_size.
        k_seq_dim/v_seq_dim: Sequence dimension index for keys/values.
    """

    def __init__(
        self,
        start_size: int = 4,
        recent_size: int = 1024,
        k_seq_dim: int = 2,
        v_seq_dim: int = 2,
    ):
        if start_size < 0:
            raise ValueError("start_size must be non-negative")
        if recent_size <= 0:
            raise ValueError("recent_size must be positive")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim

    def get_slice_info(self, seq_len: int) -> tuple[int, int, int]:
        """
        Return slicing info for cache pruning: (sink_count, recent_start, seq_len)
        """
        sink_count = min(self.start_size, seq_len)
        window_count = min(self.recent_size, max(seq_len - sink_count, 0))
        recent_start = seq_len - window_count
        recent_start = max(recent_start, sink_count)
        return sink_count, recent_start, seq_len

    def get_keep_indices(self, seq_len: int, device: torch.device) -> Optional[Tensor]:
        """
        Return sink + recent keep indices (MIT style). Returns None when the cache
        can already fit without trimming.
        """
        if seq_len <= self.cache_size:
            return None

        sink_count = min(self.start_size, seq_len)
        window_count = min(self.recent_size, max(seq_len - sink_count, 0))
        if window_count == 0:
            return torch.arange(sink_count, device=device)

        recent_start = seq_len - window_count
        recent_start = max(recent_start, sink_count)

        parts = []
        if sink_count > 0:
            parts.append(torch.arange(0, sink_count, device=device))
        parts.append(torch.arange(recent_start, seq_len, device=device))
        return torch.cat(parts, dim=0)

    def get_compression_ratio(self, original_seq_len: int) -> float:
        """
        Return compression ratio (1 - compressed/original).
        """
        if original_seq_len <= self.cache_size:
            return 0.0
        return 1.0 - (self.cache_size / original_seq_len)

