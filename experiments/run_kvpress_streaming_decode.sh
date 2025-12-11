#!/bin/bash
set -euo pipefail

# 读取 .env, 提供可共享配置
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
  set -o allexport
  source "$ENV_FILE"
  set +o allexport
fi

# Set offline Hugging Face cache (matches other scripts)
export HF_HOME="${HF_HOME:-/data2/jflin/CS3602/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

PYTHON="${PYTHON_BIN:-kvpress/.venv/bin/python}"

if [ ! -f "$PYTHON" ]; then
  echo "Python interpreter not found: $PYTHON"
  exit 1
fi

DATASET_NAME=${DATASET_NAME:-wikitext}
DATASET_CONFIG=${DATASET_CONFIG:-wikitext-103-v1}
SPLIT=${SPLIT:-test}
TEXT_COLUMN=${TEXT_COLUMN:-text}
MAX_SAMPLES=${MAX_SAMPLES:-64}
MAX_EVAL_TOKENS=${MAX_EVAL_TOKENS:-4096}
N_SINK=${N_SINK:-4}
WINDOW_SIZE=${WINDOW_SIZE:-1024}

export MAX_SAMPLES
export MAX_EVAL_TOKENS
export N_SINK
export WINDOW_SIZE

SCRIPT=$(cat <<'PY'
import importlib.util
import os
import sys
from pathlib import Path
import time

repo = Path("/data2/jflin/CS3602")
sys.path.insert(0, str(repo))

spec = importlib.util.spec_from_file_location("eval_utils", repo / "experiments" / "eval_utils.py")
eval_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_utils)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
from kvpress import StreamingLLMPress, KeyRerotationPress

MODEL_NAME = os.environ.get("MODEL_NAME", "EleutherAI/pythia-2.8b")
DATASET = os.environ.get("DATASET_NAME", "wikitext")
CONFIG = os.environ.get("DATASET_CONFIG", "")
SPLIT = os.environ.get("SPLIT", "test")
TEXT_COLUMN = os.environ.get("TEXT_COLUMN", "text")

MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "64"))
MAX_EVAL_TOKENS = int(os.environ.get("MAX_EVAL_TOKENS", "4096"))
N_SINK = int(os.environ.get("N_SINK", "4"))
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "1024"))
MAX_CACHE = N_SINK + WINDOW_SIZE

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading dataset...")
encoded = eval_utils.load_tokenized_dataset(
    dataset_name=DATASET,
    dataset_config=CONFIG,
    split=SPLIT,
    text_column=TEXT_COLUMN,
    max_samples=MAX_SAMPLES,
    tokenizer=tokenizer,
    max_eval_tokens=MAX_EVAL_TOKENS,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
).cuda()
model.eval()

press = KeyRerotationPress(StreamingLLMPress(compression_ratio=0.498, n_sink=N_SINK))
seq_len = encoded.shape[1]
prefill_len = min(MAX_CACHE, seq_len)

total_nll = 0.0
total_tokens = 0
start_time = time.perf_counter()

with press(model):
    inputs = encoded[:, :prefill_len].cuda()
    with torch.no_grad():
        outputs = model(input_ids=inputs, use_cache=True)
    logits = outputs.logits[:, :-1, :]
    labels = inputs[:, 1:]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="sum",
    )
    total_nll += loss.item()
    total_tokens += labels.numel()
    past = outputs.past_key_values
    prefill_end = time.perf_counter()

    for pos in range(prefill_len - 1, seq_len - 1):
        token = encoded[:, pos : pos + 1].cuda()
        target = encoded[:, pos + 1].cuda()
        with torch.no_grad():
            outputs = model(
                input_ids=token,
                past_key_values=past,
                use_cache=True,
            )
        logits = outputs.logits[:, -1, :]
        loss = F.cross_entropy(logits, target, reduction="sum")
        total_nll += loss.item()
        total_tokens += target.numel()
        past = outputs.past_key_values
end_time = time.perf_counter()

ppl = torch.exp(torch.tensor(total_nll / total_tokens))
print(f"PPL: {ppl.item():.4f}")
print(f"Total runtime: {end_time - start_time:.4f}s")
print(f"Prefill runtime: {prefill_end - start_time:.4f}s")
print(f"Streaming loop runtime: {end_time - prefill_end:.4f}s")
PY
)

eval "$PYTHON" <<PY
$SCRIPT
PY
