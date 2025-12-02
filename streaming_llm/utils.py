"""
工具函数模块
"""

import torch
from typing import Dict, Any


def get_model_memory_usage(model: torch.nn.Module) -> Dict[str, float]:
    """
    获取模型的显存使用情况
    
    Args:
        model: PyTorch 模型
    
    Returns:
        memory_info: 包含显存信息的字典
            - allocated_mb: 已分配显存 (MB)
            - reserved_mb: 保留显存 (MB)
            - peak_mb: 峰值显存 (MB)
    """
    if not torch.cuda.is_available():
        return {
            'allocated_mb': 0.0,
            'reserved_mb': 0.0,
            'peak_mb': 0.0
        }
    
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return {
        'allocated_mb': allocated,
        'reserved_mb': reserved,
        'peak_mb': peak
    }


def reset_peak_memory():
    """重置峰值显存统计"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def estimate_kv_cache_size(
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    dtype_bytes: int = 2  # fp16
) -> float:
    """
    估算 KV cache 的显存占用
    
    Args:
        batch_size: batch 大小
        seq_len: 序列长度
        n_layers: 层数
        n_heads: 注意力头数
        head_dim: 每个头的维度
        dtype_bytes: 数据类型字节数 (fp16=2, fp32=4)
    
    Returns:
        size_mb: KV cache 大小 (MB)
    """
    # KV cache = 2 (key + value) × batch × layers × heads × seq_len × head_dim × dtype_bytes
    size_bytes = 2 * batch_size * n_layers * n_heads * seq_len * head_dim * dtype_bytes
    size_mb = size_bytes / 1024 / 1024
    return size_mb


def format_metrics(metrics: Dict[str, Any]) -> str:
    """
    格式化指标输出
    
    Args:
        metrics: 指标字典
    
    Returns:
        formatted: 格式化的字符串
    """
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'ppl' in key.lower() or 'perplexity' in key.lower():
                lines.append(f"{key}: {value:.2f}")
            elif 'time' in key.lower() or 'sec' in key.lower():
                lines.append(f"{key}: {value:.3f}s")
            elif 'mb' in key.lower() or 'memory' in key.lower():
                lines.append(f"{key}: {value:.1f}MB")
            elif 'ratio' in key.lower():
                lines.append(f"{key}: {value:.2%}")
            else:
                lines.append(f"{key}: {value:.4f}")
        else:
            lines.append(f"{key}: {value}")
    
    return "\n".join(lines)