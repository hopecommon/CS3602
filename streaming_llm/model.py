"""
StreamingLLMWrapper: 模型包装器

使用 Hook 机制将 StreamingLLM 注入到 HuggingFace 模型中
"""

from typing import List, Optional
import torch
from torch import nn

from .kv_cache import StreamingKVCache


class StreamingLLMWrapper:
    """
    包装 HuggingFace 模型,注入 StreamingLLM 逻辑
    
    使用 forward hook 机制在 attention 层后拦截和压缩 KV cache
    
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
        >>> wrapper = StreamingLLMWrapper(model, n_sink=4, window_size=1024)
        >>> 
        >>> # 使用 context manager
        >>> with wrapper.enable():
        >>>     outputs = model(input_ids, use_cache=True)
    
    Attributes:
        model: HuggingFace 模型
        cache: StreamingKVCache 实例
        hooks: 注册的 hook 列表
        n_sink: sink token 数量
        window_size: 滑动窗口大小
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_sink: int = 4,
        window_size: int = 1024
    ):
        """
        初始化 StreamingLLMWrapper
        
        Args:
            model: HuggingFace 模型 (如 GPTNeoXForCausalLM)
            n_sink: sink token 数量 (默认 4)
            window_size: 滑动窗口大小 (默认 1024)
        """
        self.model = model
        self.n_sink = n_sink
        self.window_size = window_size
        self.cache = StreamingKVCache(n_sink=n_sink, window_size=window_size)
        self.hooks: List = []
        self._enabled = False
    
    def _create_hook(self):
        """
        创建 forward hook 函数
        
        Hook 会在 attention 层的 forward 后被调用,
        拦截 (key, value) 并进行压缩
        
        Returns:
            hook 函数
        """
        def hook(module: nn.Module, input, output):
            """
            Forward hook 函数
            
            Args:
                module: attention 模块
                input: forward 的输入
                output: forward 的输出
            
            Returns:
                修改后的 output (压缩 KV cache)
            """
            # GPTNeoX attention 输出格式:
            # output = (attn_output, present_key_value, attn_weights)
            # present_key_value = (key, value) 或 None
            
            if not isinstance(output, tuple) or len(output) < 2:
                return output
            
            attn_output = output[0]
            present_kv = output[1]
            
            # 如果没有 KV cache,直接返回
            if present_kv is None or not isinstance(present_kv, tuple):
                return output
            
            # 提取 key 和 value
            key, value = present_kv
            
            # 压缩 KV cache
            compressed_key, compressed_value = self.cache.compress(key, value)
            
            # 构造新的 output
            new_output = (attn_output, (compressed_key, compressed_value))
            
            # 如果有 attention weights,也要保留
            if len(output) > 2:
                new_output = new_output + output[2:]
            
            return new_output
        
        return hook
    
    def _register_hooks(self):
        """
        注册 hooks 到所有 attention 层
        
        支持 GPTNeoX (Pythia) 架构
        """
        # 检测模型架构
        if hasattr(self.model, 'gpt_neox'):
            # GPTNeoX / Pythia
            layers = self.model.gpt_neox.layers
        elif hasattr(self.model, 'transformer'):
            # GPT-2 / GPT-J
            layers = self.model.transformer.h
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # LLaMA / Mistral
            layers = self.model.model.layers
        else:
            raise ValueError(
                f"Unsupported model architecture. "
                f"Model type: {type(self.model).__name__}"
            )
        
        # 注册 hook 到每个 attention 层
        for layer_idx, layer in enumerate(layers):
            # 获取 attention 模块
            if hasattr(layer, 'attention'):
                attention_module = layer.attention
            elif hasattr(layer, 'attn'):
                attention_module = layer.attn
            elif hasattr(layer, 'self_attn'):
                attention_module = layer.self_attn
            else:
                raise ValueError(
                    f"Cannot find attention module in layer {layer_idx}. "
                    f"Layer type: {type(layer).__name__}"
                )
            
            # 注册 hook
            hook_handle = attention_module.register_forward_hook(
                self._create_hook()
            )
            self.hooks.append(hook_handle)
    
    def _remove_hooks(self):
        """移除所有注册的 hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def enable(self):
        """
        启用 StreamingLLM (返回 context manager)
        
        Returns:
            self (用于 context manager)
        """
        return self
    
    def __enter__(self):
        """Context manager 入口"""
        if not self._enabled:
            self._register_hooks()
            self._enabled = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 出口"""
        if self._enabled:
            self._remove_hooks()
            self._enabled = False
        return False
    
    def get_compression_ratio(self, seq_len: int) -> float:
        """
        获取给定序列长度的压缩比
        
        Args:
            seq_len: 序列长度
        
        Returns:
            compression_ratio: 压缩比 (0-1)
        """
        return self.cache.get_compression_ratio(seq_len)
    
    def __repr__(self) -> str:
        return (
            f"StreamingLLMWrapper(\n"
            f"  model={type(self.model).__name__},\n"
            f"  n_sink={self.n_sink},\n"
            f"  window_size={self.window_size},\n"
            f"  enabled={self._enabled}\n"
            f")"
        )