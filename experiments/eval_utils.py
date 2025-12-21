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


def _get_cache_len(pkv) -> int:
    """Robustly infer current KV cache length from past_key_values.

    Supports:
      - legacy tuple: pkv[layer] = (k, v), k: [B, H, T, D]
      - transformers Cache-like: pkv.layers[layer].keys
    """
    if pkv is None:
        return 0
    if isinstance(pkv, tuple):
        return int(pkv[0][0].shape[2])
    if hasattr(pkv, "layers") and len(pkv.layers) > 0 and hasattr(pkv.layers[0], "keys"):
        return int(pkv.layers[0].keys.shape[2])
    raise TypeError(f"Unsupported past_key_values type: {type(pkv)}")


def _estimate_kv_size_mb(pkv) -> float:
    """Estimate KV cache size in MB (best-effort)."""
    if pkv is None:
        return 0.0
    total_bytes = 0
    try:
        if isinstance(pkv, tuple):
            for (k, v) in pkv:
                total_bytes += k.numel() * k.element_size()
                total_bytes += v.numel() * v.element_size()
        elif hasattr(pkv, "layers"):
            for layer in pkv.layers:
                k = getattr(layer, "keys", None)
                v = getattr(layer, "values", None)
                if k is not None:
                    total_bytes += k.numel() * k.element_size()
                if v is not None:
                    total_bytes += v.numel() * v.element_size()
    except Exception:
        return 0.0
    return total_bytes / (1024.0 * 1024.0)


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
            if PG19_CACHE_FILE.exists():
                print(f"✓ 使用本地缓存: {PG19_CACHE_FILE}")
                with open(PG19_CACHE_FILE, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                texts = [cached_data.get(text_column, "")]
                print(f"  已加载缓存数据 (长度: {len(texts[0])} 字符)")
            else:
                print("本地缓存不存在，开始流式下载 PG19 数据集...")
                print("注意: 只下载一条样本以节省空间和时间")

                from datasets import load_dataset

                dataset = load_dataset(
                    dataset_name,
                    split=split,
                    streaming=True,
                    trust_remote_code=trust_remote_code,
                )

                N = 10
                buffer = []
                print(f"正在流式加载前 {N} 条样本...")
                for i, example in enumerate(dataset):
                    buffer.append(example)
                    print(f"  已加载 {i+1}/{N} 条", end="\r")
                    if i + 1 >= N:
                        break
                print()

                random_one = random.choice(buffer)
                texts = [random_one.get(text_column, "")]
                print(f"✓ 已从前 {N} 条中随机选择 1 条样本 (种子=42)")
                print(f"  样本长度: {len(texts[0])} 字符")

                PG19_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                with open(PG19_CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(random_one, f, ensure_ascii=False, indent=2)
                print(f"✓ 已保存到本地缓存: {PG19_CACHE_FILE}")
                print("  后续运行将直接使用本地缓存，无需重新下载")

    else:
        from datasets import load_dataset

        dataset_kwargs = {"split": split, "trust_remote_code": trust_remote_code}
        if use_streaming:
            dataset_kwargs["streaming"] = True

        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, **dataset_kwargs)
        else:
            dataset = load_dataset(dataset_name, **dataset_kwargs)

        if not use_streaming and max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        texts = []
        for idx, row in enumerate(dataset):
            if use_streaming and max_samples and idx >= max_samples:
                break
            text = row.get(text_column, "")
            if text and not text.isspace():
                texts.append(text)

    if not texts:
        raise ValueError(f"No non-empty rows found for dataset {dataset_name}")

    concatenated = "\n\n".join(texts)
    encodings = tokenizer(concatenated, return_tensors="pt")
    input_ids = encodings.input_ids
    if max_eval_tokens:
        input_ids = input_ids[:, :max_eval_tokens]
    print(f"  最终 token 数: {input_ids.shape[1]}")
    return input_ids


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
        if mode == "flash":
            ctx = sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False)
        elif mode == "math":
            ctx = sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        else:
            info = check_sdpa_flash_available(dtype=torch.bfloat16)
            if info.get("flash_enabled") and info.get("smoke_test", False):
                ctx = sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False)
            else:
                ctx = sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        with ctx:
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
    streaming_wrapper=None,
    max_cache_size: Optional[int] = None,
) -> PerplexityResult:
    """
    计算 perplexity
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

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    total_start = time.perf_counter()
    prefill_start = time.perf_counter()

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
    streaming_wrapper=None,
) -> PerplexityResult:
    """
    使用解码式评估 (逐 token) 计算 PPL 和时间
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

    # Optional: measure streaming_wrapper.update() time (including CUDA sync), controlled by env.
    measure_update_sync = os.environ.get("MEASURE_UPDATE_SYNC", "0") == "1"
    update_calls = 0
    update_time_sec = 0.0
    kv_cache_first_mb = None
    kv_cache_peak_mb = 0.0

    def _do_update(pkv):
        nonlocal update_calls, update_time_sec, kv_cache_first_mb, kv_cache_peak_mb
        if streaming_wrapper is None:
            return pkv
        update_calls += 1
        if measure_update_sync and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pkv2 = streaming_wrapper.update(pkv)
            torch.cuda.synchronize()
            update_time_sec += (time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            pkv2 = streaming_wrapper.update(pkv)
            update_time_sec += (time.perf_counter() - t0) if measure_update_sync else 0.0

        mb = _estimate_kv_size_mb(pkv2)
        if kv_cache_first_mb is None:
            kv_cache_first_mb = mb
        kv_cache_peak_mb = max(kv_cache_peak_mb, mb)
        return pkv2

    if streaming_wrapper is None:
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

        for pos in range(prefill_len - 1, seq_len - 1):
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
                position_ids = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long).unsqueeze(0)
                outputs = model(input_ids=input_ids, use_cache=True, position_ids=position_ids)

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
            past_key_values = _do_update(past_key_values)
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
                    cache_len = _get_cache_len(past_key_values)
                    position_ids = torch.tensor([[cache_len]], device=current_input.device, dtype=torch.long)
                    attention_mask = torch.ones(
                        (current_input.shape[0], cache_len + current_input.shape[1]),
                        device=current_input.device,
                        dtype=torch.long,
                    )

                    outputs = model(
                        input_ids=current_input,
                        past_key_values=past_key_values,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
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
                past_key_values = _do_update(past_key_values)

                if not first_token_recorded:
                    first_token_recorded = True

    if use_cuda_timing:
        end_evt.record()
        torch.cuda.synchronize()
        total_time = start_evt.elapsed_time(end_evt) / 1000.0
        prefill_time = start_evt.elapsed_time(prefill_end_evt) / 1000.0
        if total_tokens > 0:
            first_token_time = first_start_evt.elapsed_time(first_end_evt) / 1000.0
    else:
        total_time = time.perf_counter() - total_start

    # -------- metrics summary --------
    try:
        flash_env = os.environ.get("FLASH_SDPA", "unset")
    except Exception:
        flash_env = "unset"

    decode_steps = max(0, seq_len - 1 - (prefill_len - 1))
    tok_per_s = (decode_steps / total_time) if total_time > 0 else 0.0
    tpot_ms = (total_time / decode_steps * 1000.0) if decode_steps > 0 else 0.0

    if os.environ.get("PRINT_METRICS", "1") == "1":
        tag = "Streaming" if streaming_wrapper is not None else "Baseline"
        print(
            f"[METRIC] ({tag}) max_cache_size={max_cache_size} prefill_len={prefill_len} seq_len={seq_len} "
            f"decode_steps={decode_steps} FLASH_SDPA={flash_env}"
        )
        if streaming_wrapper is None:
            print(
                f"[METRIC] ({tag}) TTFT={prefill_time:.4f}s  TPOT={tpot_ms:.3f}ms  tok/s={tok_per_s:.2f}  "
                f"prefill={prefill_time:.4f}s  total={total_time:.4f}s"
            )
        else:
            extra = ""
            if measure_update_sync:
                avg_ms = (update_time_sec / update_calls * 1000.0) if update_calls > 0 else 0.0
                extra = f"  update_total={update_time_sec:.4f}s  update_avg={avg_ms:.3f}ms  update_calls={update_calls}"
            print(
                f"[METRIC] ({tag}) TTFT={first_token_time if first_token_time else 0.0:.4f}s  "
                f"TPOT={tpot_ms:.3f}ms  tok/s={tok_per_s:.2f}  prefill={prefill_time:.4f}s  total={total_time:.4f}s{extra}"
            )
            if kv_cache_first_mb is not None:
                print(f"[METRIC] ({tag}) KV_cache_first={kv_cache_first_mb:.2f}MB  KV_cache_peak={kv_cache_peak_mb:.2f}MB")

        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
            peak_res = torch.cuda.max_memory_reserved() / (1024**2)
            print(f"[METRIC] ({tag}) cuda_peak_alloc={peak_alloc:.2f}MB  cuda_peak_reserved={peak_res:.2f}MB")
    # -------- end metrics summary --------

    ppl = torch.exp(torch.tensor(total_nll / total_tokens))
    return PerplexityResult(
        perplexity=ppl.item(),
        runtime_sec=total_time,
        prefill_sec=prefill_time,
        first_token_latency_sec=first_token_time,
    )


_compute_streaming_perplexity = _compute_streaming_decode_perplexity


def save_results(results: Dict[str, Any], output_path: Path):
    """
    保存结果到 JSON 文件
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n结果已保存到: {output_path}")


def print_results(results: Dict[str, Any]):
    """
    打印结果
    """
    print("\n" + "=" * 60)
    print("实验结果")
    print("=" * 60)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("=" * 60)
