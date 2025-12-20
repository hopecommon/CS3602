"""
评估工具函数

提供通用的 PPL 评估、数据加载等功能
"""

import time
import random
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer
from torch.backends.cuda import sdp_kernel
import torch.nn as nn
from tqdm import tqdm


def _load_json_entries(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        entries: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
        return entries

@dataclass
class PerplexityResult:
    perplexity: float
    runtime_sec: float
    prefill_sec: float
    first_token_latency_sec: float

    def __iter__(self):
        yield self.perplexity
        yield self.runtime_sec
        yield self.prefill_sec


# 设置随机种子以确保可复现
random.seed(42)

# PG19 本地缓存目录
PG19_CACHE_DIR = Path("data/pg19")
PG19_CACHE_FILE = PG19_CACHE_DIR / "sample.json"
PG19_SAMPLES_PATTERN = PG19_CACHE_DIR / "long_context_*.json"

# Utility functions
def _extract_length_from_name(path: Path) -> Optional[int]:
    match = re.search(r"long_context_(\d+)", path.name)
    if match:
        return int(match.group(1))
    return None


def _resolve_pg19_sample_path() -> Optional[Path]:
    env_override = os.environ.get("PG19_SAMPLE_FILE")
    if env_override:
        candidate = Path(env_override)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Specified PG19_SAMPLE_FILE does not exist: {candidate}")

    candidates = sorted(PG19_CACHE_DIR.glob("long_context_*.json"))
    if not candidates:
        return None

    target_length = os.environ.get("PG19_SAMPLE_LENGTH")
    if target_length:
        chosen = [p for p in candidates if _extract_length_from_name(p) == int(target_length)]
        if chosen:
            return chosen[0]

    candidates.sort(key=lambda p: (_extract_length_from_name(p) or 0, p.name), reverse=True)
    return candidates[0]


def _resolve_wikitext_sample_path() -> Optional[Path]:
    env_override = os.environ.get("WIKITEXT_SAMPLE_FILE")
    if env_override:
        candidate = Path(env_override)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Specified WIKITEXT_SAMPLE_FILE does not exist: {candidate}")
    candidates = sorted(WIKITEXT_CACHE_DIR.glob("long_context_*.json"))
    if not candidates:
        return None
    target_length = os.environ.get("WIKITEXT_SAMPLE_LENGTH")
    if target_length:
        chosen = [p for p in candidates if _extract_length_from_name(p) == int(target_length)]
        if chosen:
            return chosen[0]
    candidates.sort(key=lambda p: (_extract_length_from_name(p) or 0, p.name), reverse=True)
    return candidates[0]


# WikiText 本地缓存目录
WIKITEXT_CACHE_DIR = Path("data/wikitext")
WIKITEXT_CACHE_FILE = WIKITEXT_CACHE_DIR / "sample.json"
WIKITEXT_SAMPLES_PATTERN = WIKITEXT_CACHE_DIR / "long_context_*.json"


def load_tokenized_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    text_column: str,
    max_samples: Optional[int],
    tokenizer,
    max_eval_tokens: Optional[int] = None,
    trust_remote_code: bool = False,
    use_streaming: bool = False,
) -> Tensor:
    """
    加载并 tokenize 数据集
    
    Args:
        dataset_name: 数据集名称 (如 'wikitext', 'pg19')
        dataset_config: 数据集配置 (如 'wikitext-103-v1')
        split: 数据集分割 (如 'test')
        text_column: 文本列名 (如 'text')
        max_samples: 最大样本数
        tokenizer: tokenizer 实例
        max_eval_tokens: 最大评估 token 数
        trust_remote_code: 是否信任远程代码
        use_streaming: 是否使用流式加载
    
    Returns:
        input_ids: tokenized 输入 [1, seq_len]
    """
    # 特殊处理 WikiText: 优先读取预采样样本
    if dataset_name.lower() == "wikitext":
        print("检测到 WikiText 数据集...")
        sample_path = _resolve_wikitext_sample_path()
        if sample_path:
            print(f"✓ 使用预采样 WikiText 样本: {sample_path.name}")
            entries = _load_json_entries(sample_path)
            if not entries:
                raise ValueError(f"{sample_path} 中未包含有效样本")
            if max_samples:
                entries = entries[:max_samples]
            texts = []
            for entry in entries:
                text = entry.get(text_column, "") or entry.get("text", "")
                if text and not text.isspace():
                    texts.append(text)
            if not texts:
                raise ValueError(f"{sample_path} 中没有可用文本")
            concatenated = "\n\n".join(texts)
            encodings = tokenizer(concatenated, return_tensors="pt")
            input_ids = encodings.input_ids
            if max_eval_tokens:
                input_ids = input_ids[:, :max_eval_tokens]
            print(f"  最终 token 数: {input_ids.shape[1]}")
            return input_ids
        print("  未找到采样文件，继续使用原始 WikiText 数据集")

    # 特殊处理 PG19: 下载一条数据到本地，后续使用本地缓存
    if dataset_name.lower() == "pg19":
        print("检测到 PG19 数据集...")

        sample_path = _resolve_pg19_sample_path()
        if sample_path:
            print(f"✓ 使用预采样 PG19 样本: {sample_path.name}")
            entries = _load_json_entries(sample_path)
            if not entries:
                raise ValueError(f"{sample_path} 中未包含有效样本")
            text = entries[0].get(text_column, "") or entries[0].get("text", "")
            if not text:
                raise ValueError(f"{sample_path} 中没有可用文本")
            texts = [text]
        else:
        
            # 检查本地缓存
            if PG19_CACHE_FILE.exists():
                print(f"✓ 使用本地缓存: {PG19_CACHE_FILE}")
                try:
                    with open(PG19_CACHE_FILE, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    text = cached_data.get(text_column, "") or cached_data.get("text", "")
                    if not text:
                         # Try fallback keys or structure
                         if isinstance(cached_data, dict):
                             # Maybe it's nested or different key
                             text = str(cached_data) # Last resort
                    
                    if not text or len(text) < 100: # Basic validity check
                         print("  缓存数据无效或过短，重新下载...")
                         raise ValueError("Invalid cache")
                         
                    texts = [text]
                    print(f"  已加载缓存数据 (长度: {len(texts[0])} 字符)")
                except Exception as e:
                    print(f"  读取缓存失败: {e}")
                    # Fall through to download logic
                    if PG19_CACHE_FILE.exists():
                        PG19_CACHE_FILE.unlink() # Delete bad cache
                    texts = None
            
            if not texts:
                print("本地缓存不存在，开始流式下载 PG19 数据集...")
                print("注意: 只下载一条样本以节省空间和时间")
                
                # 流式加载前 N 条作为候选
                from datasets import load_dataset
                dataset = load_dataset(
                    dataset_name,
                    split=split,
                    streaming=True,
                    trust_remote_code=trust_remote_code
                )
                
                # 收集前 10 条作为候选
                N = 10
                buffer = []
                print(f"正在流式加载前 {N} 条样本...")
                for i, example in enumerate(dataset):
                    buffer.append(example)
                    print(f"  已加载 {i+1}/{N} 条", end='\r')
                    if i + 1 >= N:
                        break
                print()  # 换行
                
                # 随机选择一条 (固定种子确保可复现)
                random_one = random.choice(buffer)
                texts = [random_one.get(text_column, "")]
                print(f"✓ 已从前 {N} 条中随机选择 1 条样本 (种子=42)")
                print(f"  样本长度: {len(texts[0])} 字符")
                
                # 保存到本地缓存
                PG19_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                with open(PG19_CACHE_FILE, 'w', encoding='utf-8') as f:
                    json.dump(random_one, f, ensure_ascii=False, indent=2)
                print(f"✓ 已保存到本地缓存: {PG19_CACHE_FILE}")
                print(f"  后续运行将直接使用本地缓存，无需重新下载")
    
    else:
        # 其他数据集的正常加载逻辑
        from datasets import load_dataset
        dataset_kwargs = {
            "split": split,
            "trust_remote_code": trust_remote_code
        }
        if use_streaming:
            dataset_kwargs["streaming"] = True
        
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, **dataset_kwargs)
        else:
            dataset = load_dataset(dataset_name, **dataset_kwargs)
        
        # 限制样本数
        if not use_streaming and max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        # 收集文本
        texts = []
        for idx, row in enumerate(dataset):
            if use_streaming and max_samples and idx >= max_samples:
                break
            text = row.get(text_column, "")
            if text and not text.isspace():
                texts.append(text)
    
    if not texts:
        raise ValueError(f"No non-empty rows found for dataset {dataset_name}")
    
def check_sdpa_flash_available(dtype=torch.bfloat16):
    if not torch.cuda.is_available():
        return {"flash_enabled": False, "reason": "cuda_unavailable"}
    try:
        enabled = torch.backends.cuda.flash_sdp_enabled()
    except Exception:
        enabled = False
    result = {"flash_enabled": bool(enabled)}
    if enabled:
        try:
            q = torch.randn(1, 4, 8, 64, device="cuda", dtype=dtype)
            k = torch.randn(1, 4, 8, 64, device="cuda", dtype=dtype)
            v = torch.randn(1, 4, 8, 64, device="cuda", dtype=dtype)
            with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                _ = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
            result["smoke_test"] = True
        except Exception as e:
            result["smoke_test"] = False
            result["error"] = str(e)
    return result

class NeoXFlashAttentionAdapter(nn.Module):
    def __init__(self, original_module: nn.Module, backend: str = "auto"):
        super().__init__()
        self.original = original_module
        self.backend = backend
        
        # Extract components from original module
        if hasattr(original_module, "query_key_value"):
             self.query_key_value = original_module.query_key_value
        else:
             raise ValueError("NeoXFlashAttentionAdapter requires original_module to have query_key_value")
             
        if hasattr(original_module, "dense"):
             self.dense = original_module.dense
        else:
             raise ValueError("NeoXFlashAttentionAdapter requires original_module to have dense")
             
        if hasattr(original_module, "rotary_emb"):
             self.rotary_emb = original_module.rotary_emb
        else:
             # Try to find rotary embedding from parent model if not in module
             # This is tricky without reference to model, so we assume standard NeoX structure
             self.rotary_emb = None 

        self.num_attention_heads = original_module.num_attention_heads
        self.hidden_size = original_module.hidden_size
        self.head_size = original_module.hidden_size // original_module.num_attention_heads
        if hasattr(original_module, "rotary_pct"):
            self.rotary_ndims = int(self.head_size * original_module.rotary_pct)
        elif hasattr(original_module, "rotary_ndims"):
            self.rotary_ndims = original_module.rotary_ndims
        else:
            # Default for GPT-NeoX / Pythia often is partial rotary or full
            # We'll try to infer or assume based on common configs if not present
            # For Pythia 2.8b, it's typically rotary_pct=0.25
            # Safest is to check if there is a rotary_emb module and inspect it
            if self.rotary_emb is not None and hasattr(self.rotary_emb, "dim"):
                 self.rotary_ndims = self.rotary_emb.dim
            else:
                 # Fallback: Assume full rotation or partial (Pythia default 0.25 * 128 = 32)
                 # This is risky without config access.
                 # Let's try to get config from original module if possible
                 self.rotary_ndims = int(self.head_size * 0.25) # Default guess for Pythia

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        layer_past=None,
        use_cache=False,
        output_attentions=False,
        position_ids=None,
        **kwargs,
    ):
        mode = self.backend
        env = os.environ.get("FLASH_SDPA", None)
        if env == "1":
            mode = "flash"
        elif env == "0":
            mode = "math"
        
        # Fallback to original implementation if not flash mode or not using SDPA
        if mode != "flash" and mode != "auto":
             return self.original(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
                position_ids=position_ids,
                **kwargs,
            )

        # Flash/Auto mode implementation
        bsz, q_len, _ = hidden_states.size()
        
        # 1. QKV Projection
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, 3, self.num_attention_heads, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4) # [3, bsz, num_heads, q_len, head_size]
        query, key, value = qkv[0], qkv[1], qkv[2]
        
        # 2. RoPE (Manual implementation since we bypass original forward)
        # Note: This relies on streaming_llm already handling position updates for cache
        # If position_ids is None, infer from cache or sequence
        if hasattr(self.original, "rotary_emb"):
             rotary_emb = self.original.rotary_emb
        elif self.rotary_emb:
             rotary_emb = self.rotary_emb
        else:
             # Fallback: try to find it attached to self.original (some implementations)
             rotary_emb = getattr(self.original, "rotary_emb", None)

        if rotary_emb is not None:
             if position_ids is None:
                  # Infer position_ids
                  past_length = 0
                  if layer_past is not None:
                      # past_key_values[0] is keys: [bsz, num_heads, past_len, head_size]
                      if hasattr(layer_past[0], "device"): # Tuple or Tensor
                          past_length = layer_past[0].shape[-2]
                      # If it's something else (like Cache object), we might need logic, 
                      # but here layer_past usually comes from legacy tuple or our wrapper
                  
                  position_ids = torch.arange(past_length, past_length + q_len, dtype=torch.long, device=hidden_states.device)
                  position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
             
             cos, sin = rotary_emb(value, seq_len=position_ids.max() + 1)
             # Pythia/NeoX RoPE usually applies to query and key
             # Need to match dimensions for apply_rotary_pos_emb
             # Standard implementation:
             from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
             
             # IMPORTANT FIX: Ensure dimensions match for RoPE
             # query/key: [bsz, num_heads, q_len, head_size]
             # cos/sin: [q_len, rotary_dim] or [1, q_len, rotary_dim] depending on implementation
             
             # If partial rotary (e.g. Pythia 2.8B rotates 32 dims out of 128), we must split
             rotary_dim = cos.shape[-1]
             head_dim = query.shape[-1]
             
             if rotary_dim < head_dim:
                 q_rot = query[..., :rotary_dim]
                 q_pass = query[..., rotary_dim:]
                 k_rot = key[..., :rotary_dim]
                 k_pass = key[..., rotary_dim:]
                 
                 q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, position_ids)
                 
                 query = torch.cat((q_rot, q_pass), dim=-1)
                 key = torch.cat((k_rot, k_pass), dim=-1)
             else:
                 query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        # 3. KV Cache Management
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        
        if use_cache:
            present = (key, value)
        else:
            present = None

        # 4. SDPA Call
        # SDPA expects [bsz, num_heads, q_len, head_size]
        # Our Q/K/V are currently [bsz, num_heads, seq_len, head_size]
        
        # Ensure dimensions match for SDPA
        # Usually they do, but if q_len is different from k_len (e.g. cross attn or cache mismatch), we need to be careful.
        # Here we concatenated past_key/value to key/value, so key/value seq_len >= query seq_len
        # F.scaled_dot_product_attention handles this (Q: [B, H, Lq, D], K, V: [B, H, Lk, D])
        # BUT, if head_size mismatch occurs (e.g. some broadcast issue), we check.
        
        # Explicit check for head_size match (last dimension)
        if query.shape[-1] != key.shape[-1]:
             # This shouldn't happen with standard RoPE/Projection unless something weird happened
             raise ValueError(f"Head size mismatch: Query {query.shape}, Key {key.shape}")

        # Context manager for SDPA
        ctx_manager = sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False)
        
        with ctx_manager:
            attn_output = F.scaled_dot_product_attention(
                query, 
                key, 
                value, 
                attn_mask=None, 
                dropout_p=0.0, 
                is_causal=True if q_len > 1 else False
            )
        
        # 5. Output Projection
        attn_output = attn_output.transpose(1, 2).contiguous() # [bsz, q_len, num_heads, head_size]
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        
        output = self.dense(attn_output)
        
        if output_attentions:
            return (output, present, None) # No attention weights from SDPA
            
        return (output, present)

def wrap_attention_modules(model: nn.Module, backend: str = "auto"):
    for name, module in model.named_modules():
        has_qkv = hasattr(module, "query_key_value") and isinstance(getattr(module, "query_key_value"), nn.Linear)
        has_out = hasattr(module, "dense") and isinstance(getattr(module, "dense"), nn.Linear)
        if has_qkv and has_out:
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr = parts[-1]
            adapter = NeoXFlashAttentionAdapter(module, backend=backend)
            setattr(parent, attr, adapter)
    return model


def compute_perplexity(
    model,
    encoded_dataset: Tensor,
    device: torch.device,
    max_length: int,
    stride: int,
    use_streaming: bool = False,
    streaming_wrapper = None,
    max_cache_size: Optional[int] = None,
) -> PerplexityResult:
    """
    计算 perplexity
    
    Args:
        model: 语言模型
        encoded_dataset: tokenized 数据集 [1, seq_len]
        device: 设备
        max_length: 单次评估的最大长度
        stride: 滑动步长
        use_streaming: 是否使用 StreamingLLM
        streaming_wrapper: StreamingLLMWrapper 实例
    
    Returns:
        perplexity: PPL 值
        total_time: 总时间 (秒)
        prefill_time: prefill 时间 (秒)
    """
    if encoded_dataset.device != device:
        encoded_dataset = encoded_dataset.to(device)

    if max_cache_size is not None:
        wrapper = streaming_wrapper if use_streaming else None
        return _compute_streaming_decode_perplexity(
            model=model,
            encoded_dataset=encoded_dataset,
            device=device,
            max_cache_size=max_cache_size,
            streaming_wrapper=wrapper,
        )

    nlls = []
    total_tokens = 0
    seq_len = encoded_dataset.size(1)
    
    # 重置显存统计
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    total_start = time.perf_counter()
    prefill_start = time.perf_counter()
    
    # 使用 StreamingLLM 或普通模式
    if use_streaming and streaming_wrapper is not None:
        context = streaming_wrapper.enable()
    else:
        from contextlib import nullcontext
        context = nullcontext()
    
    with context:
        for start_idx in range(0, seq_len, stride):
            begin_loc = max(start_idx + stride - max_length, 0)
            end_loc = min(start_idx + stride, seq_len)
            trg_len = end_loc - start_idx
            
            input_ids = encoded_dataset[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=target_ids, use_cache=True)
            
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood.detach().to("cpu"))
            total_tokens += trg_len
            
            if end_loc == seq_len:
                break
    
    prefill_time = time.perf_counter() - prefill_start
    total_time = time.perf_counter() - total_start
    
    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)

    return PerplexityResult(
        perplexity=ppl.item(),
        runtime_sec=total_time,
        prefill_sec=prefill_time,
        first_token_latency_sec=0.0,
    )


def _compute_streaming_decode_perplexity(
    model,
    encoded_dataset: Tensor,
    device: torch.device,
    max_cache_size: int,
    streaming_wrapper = None,
) -> PerplexityResult:
    """
    使用解码式评估 (逐 token) 计算 PPL 和时间

    Args:
        model: 语言模型
        encoded_dataset: tokenized 数据集
        device: 设备
        max_cache_size: 最多保留的 token 数 (n_sink + window_size)
        streaming_wrapper: 若提供则使用 StreamingLLM, 否则模拟 sliding window baseline
    """
    seq_len = encoded_dataset.size(1)
    if seq_len < 2:
        raise ValueError("Dataset is too short to compute perplexity (seq_len < 2)")

    if encoded_dataset.device != device:
        encoded_dataset = encoded_dataset.to(device)

    max_cache_size = max(2, min(max_cache_size, seq_len))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    use_cuda_timing = device.type == "cuda" and torch.cuda.is_available()
    if use_cuda_timing:
        start_evt = torch.cuda.Event(enable_timing=True)
        prefill_end_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        first_start_evt = torch.cuda.Event(enable_timing=True)
        first_end_evt = torch.cuda.Event(enable_timing=True)
    else:
        total_start = time.perf_counter()
        prefill_start = time.perf_counter()

    total_nll = 0.0
    total_tokens = 0
    first_token_time = 0.0

    if streaming_wrapper is None:
        # Baseline: 使用 sliding window 限制上下文长度
        # 这样可以避免超过模型的 max_position_embeddings 限制
        prefill_len = min(max_cache_size, seq_len)
        input_ids = encoded_dataset[:, :prefill_len]
        if use_cuda_timing:
            start_evt.record()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)
        if use_cuda_timing:
            prefill_end_evt.record()

        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        if labels.numel() > 0:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="sum",
            )
            total_nll += loss.item()
            total_tokens += labels.numel()

        if use_cuda_timing:
            first_token_recorded = False
        else:
            prefill_time = time.perf_counter() - prefill_start
            first_token_recorded = False

        # Add None check for past_key_values if used (Baseline doesn't use cache here, but good practice)
        
        for pos in tqdm(range(prefill_len - 1, seq_len - 1), desc="Decoding (Baseline)"):
            start_idx = max(0, pos + 1 - max_cache_size)
            context = encoded_dataset[:, start_idx:pos + 1]
            target = encoded_dataset[:, pos + 1]

            if use_cuda_timing and not first_token_recorded:
                first_start_evt.record()
            with torch.no_grad():
                outputs = model(input_ids=context, use_cache=False)
            if use_cuda_timing and not first_token_recorded:
                first_end_evt.record()

            logits = outputs.logits[:, -1, :]
            loss = F.cross_entropy(
                logits,
                target,
                reduction="sum",
            )
            total_nll += loss.item()
            total_tokens += target.numel()

            if not first_token_recorded:
                first_token_recorded = True
    else:
        past_key_values = None
        prefill_len = min(max_cache_size, seq_len)
        with streaming_wrapper.enable():
            input_ids = encoded_dataset[:, :prefill_len]
            if use_cuda_timing:
                start_evt.record()
            with torch.no_grad():
                outputs = model(input_ids=input_ids, use_cache=True)

            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            if labels.numel() > 0:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    reduction="sum",
                )
                total_nll += loss.item()
                total_tokens += labels.numel()

            past_key_values = outputs.past_key_values
            past_key_values = streaming_wrapper.update(past_key_values)
            if use_cuda_timing:
                prefill_end_evt.record()

            if use_cuda_timing:
                first_token_recorded = False
            else:
                prefill_time = time.perf_counter() - prefill_start
                first_token_recorded = False

            for pos in tqdm(range(prefill_len - 1, seq_len - 1), desc="Decoding"):
                current_input = encoded_dataset[:, pos:pos + 1]
                target = encoded_dataset[:, pos + 1]

                if use_cuda_timing and not first_token_recorded:
                    first_start_evt.record()
                with torch.no_grad():
                    outputs = model(
                        input_ids=current_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                if use_cuda_timing and not first_token_recorded:
                    first_end_evt.record()

                logits = outputs.logits[:, -1, :]
                loss = F.cross_entropy(
                    logits,
                    target,
                    reduction="sum",
                )
                total_nll += loss.item()
                total_tokens += target.numel()
                past_key_values = outputs.past_key_values
                past_key_values = streaming_wrapper.update(past_key_values)

                if not first_token_recorded:
                    first_token_recorded = True

    if use_cuda_timing:
        end_evt.record()
        torch.cuda.synchronize()
        total_time = start_evt.elapsed_time(end_evt) / 1000.0
        # Check if prefill event was recorded (might not be if loop was empty or skipped)
        try:
             prefill_time = start_evt.elapsed_time(prefill_end_evt) / 1000.0
        except RuntimeError:
             prefill_time = 0.0
        if total_tokens > 0:
            try:
                first_token_time = first_start_evt.elapsed_time(first_end_evt) / 1000.0
            except RuntimeError:
                 first_token_time = 0.0
    else:
        total_time = time.perf_counter() - total_start
    ppl = torch.exp(torch.tensor(total_nll / total_tokens))
    return PerplexityResult(
        perplexity=ppl.item(),
        runtime_sec=total_time,
        prefill_sec=prefill_time,
        first_token_latency_sec=first_token_time,
    )


_compute_streaming_perplexity = _compute_streaming_decode_perplexity


def save_results(
    results: Dict[str, Any],
    output_path: Path
):
    """
    保存结果到 JSON 文件
    
    Args:
        results: 结果字典
        output_path: 输出路径
    """
    import json
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n结果已保存到: {output_path}")


def print_results(results: Dict[str, Any]):
    """
    打印结果
    
    Args:
        results: 结果字典
    """
    import json
    print("\n" + "="*60)
    print("实验结果")
    print("="*60)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("="*60)
