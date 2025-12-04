"""
StreamingLLM: 从零复现 KV Cache 压缩算法

基于论文: Efficient Streaming Language Models with Attention Sinks
https://arxiv.org/abs/2309.17453

核心思想:
- 保留前 n_sink 个 token (attention sink)
- 保留最近 window_size 个 token
- 丢弃中间所有 token,实现固定大小的 KV cache
"""

from .kv_cache import StreamingKVCache
from .mit_cache import StartRecentKVCache
from .model import StreamingLLMWrapper

__version__ = "0.1.0"
__all__ = ["StreamingKVCache", "StartRecentKVCache", "StreamingLLMWrapper"]
