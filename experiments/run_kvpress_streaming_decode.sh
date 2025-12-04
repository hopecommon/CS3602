#!/bin/bash
set -euo pipefail

# Set offline Hugging Face cache (matches other scripts)
export HF_HOME="/data2/jflin/CS3602/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHON="kvpress/.venv/bin/python"

if [ ! -f "$PYTHON" ]; then
  echo "Python interpreter not found: $PYTHON"
  exit 1
fi

SCRIPT=$(cat <<'PY'
import importlib.util
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

MODEL_NAME = "EleutherAI/pythia-70m"
DATASET = "wikitext"
CONFIG = "wikitext-103-v1"
SPLIT = "test"
TEXT_COLUMN = "text"
MAX_SAMPLES = 64
MAX_EVAL_TOKENS = 4096
N_SINK = 4
WINDOW_SIZE = 1024
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
