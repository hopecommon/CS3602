"""
RoPE 辅助函数

提供旋转相关的工具用于在压缩 KV cache 后重新计算位置编码
"""

from __future__ import annotations

import torch
from torch import Tensor


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half of the hidden dimensions (RoPE helper)"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rerotate_keys(
    keys: Tensor,
    old_positions: Tensor,
    new_positions: Tensor,
    inv_freq: Tensor | None,
    rotary_dim: int | None,
) -> Tensor:
    """
    将压缩后的 keys 重新旋转,使其位置编码与新的 cache 位置对齐

    Args:
        keys: [batch, heads, seq_len, head_dim]
        old_positions: [batch, seq_len_kept] 压缩前的相对位置
        new_positions: [batch, seq_len_kept] 压缩后的连续位置
        inv_freq: RoPE 频率张量
        rotary_dim: 需要应用 RoPE 的维度数
    """
    if inv_freq is None or not torch.is_tensor(inv_freq):
        return keys

    if rotary_dim is None or rotary_dim <= 0:
        return keys

    rotary_dim = min(rotary_dim, keys.shape[-1])
    if rotary_dim == 0:
        return keys

    # inv_freq 只需要前 rotary_dim // 2 个频率
    freq = inv_freq[: rotary_dim // 2].to(device=keys.device, dtype=torch.float32)

    # 计算位置增量并广播
    delta = (new_positions - old_positions).to(device=keys.device, dtype=torch.float32)
    delta = delta.unsqueeze(-1)  # [batch, seq_len_kept, 1]

    freqs = delta * freq  # [batch, seq_len_kept, rotary_dim//2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [batch, seq_len_kept, rotary_dim]
    cos = emb.cos().to(dtype=keys.dtype).unsqueeze(1)  # [batch, 1, seq_len_kept, rotary_dim]
    sin = emb.sin().to(dtype=keys.dtype).unsqueeze(1)

    k_rot = keys[:, :, :, :rotary_dim]
    k_pass = keys[:, :, :, rotary_dim:]
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    return torch.cat([k_rot, k_pass], dim=-1)

