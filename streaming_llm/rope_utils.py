"""
RoPE 辅助函数

提供旋转相关的工具用于在压缩 KV cache 后重新计算位置编码
"""

from __future__ import annotations

import torch
from torch import Tensor


def apply_rotary_shift_inplace(
    keys: Tensor,
    *,
    start_index: int,
    cos: Tensor,
    sin: Tensor,
    rotary_dim: int,
) -> Tensor:
    """
    Apply a constant RoPE shift to a contiguous suffix of keys in-place.

    This is used for StreamingLLM start+recent pruning: sinks keep delta=0,
    and all recent tokens share the same position delta, so we can rotate
    the recent segment with a constant shift.

    Args:
        keys: [batch, heads, seq_len, head_dim]
        start_index: starting token index (e.g., sink_count) to rotate.
        cos/sin: [half] or broadcastable to [1,1,1,half]
        rotary_dim: number of dims in head_dim using RoPE.
    """
    if rotary_dim <= 0:
        return keys
    rotary_dim = min(int(rotary_dim), int(keys.shape[-1]))
    half = rotary_dim // 2
    if half <= 0:
        return keys

    if start_index >= keys.shape[2]:
        return keys

    cos = cos.to(device=keys.device, dtype=keys.dtype).view(1, 1, 1, half)
    sin = sin.to(device=keys.device, dtype=keys.dtype).view(1, 1, 1, half)

    segment = keys[:, :, start_index:, : rotary_dim]
    x1 = segment[..., :half]
    x2 = segment[..., half : 2 * half]

    # Use temporaries to avoid in-place overwrite hazards.
    new1 = x1 * cos - x2 * sin
    new2 = x1 * sin + x2 * cos
    x1.copy_(new1)
    x2.copy_(new2)
    return keys


def build_rotation_cache(
    old_positions: Tensor,
    new_positions: Tensor,
    inv_freq: Tensor,
    rotary_dim: int,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """
    预计算 RoPE 旋转所需的 cos/sin, 供多层复用。
    """
    delta = (new_positions - old_positions).to(device=inv_freq.device, dtype=torch.float32)
    freq = inv_freq[: rotary_dim // 2].to(device=inv_freq.device, dtype=torch.float32)
    freq = freq.view(1, 1, -1)
    delta = delta.unsqueeze(-1)
    freqs = delta * freq
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(dtype=dtype).unsqueeze(1)
    sin = emb.sin().to(dtype=dtype).unsqueeze(1)
    return cos, sin


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
    precomputed: tuple[Tensor, Tensor] | None = None,
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
    if precomputed is None:
        cos, sin = build_rotation_cache(
            old_positions=old_positions.to(device=keys.device),
            new_positions=new_positions.to(device=keys.device),
            inv_freq=inv_freq.to(device=keys.device),
            rotary_dim=rotary_dim,
            dtype=keys.dtype,
        )
    else:
        cos, sin = precomputed

    k_rot = keys[:, :, :, :rotary_dim]
    k_pass = keys[:, :, :, rotary_dim:]
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    return torch.cat([k_rot, k_pass], dim=-1)
