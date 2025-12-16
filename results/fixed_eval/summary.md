# Fixed Eval Summary

## Decode-Loop (comparable)

| Dataset | Method | PPL | Runtime (s) | Speedup vs baseline | First-token (s) |
|---|---|---:|---:|---:|---:|
| pg19_20k | baseline | 8.8245 | 1446.19 | 1.00× | 0.078 |
| pg19_20k | ours | 8.9947 | 294.12 | 4.92× | 0.019 |
| pg19_20k | kvpress | 8.9947 | 387.45 | 3.73× | 0.023 |
| wikitext | baseline | 9.3713 | 164.75 | 1.00× | 0.097 |
| wikitext | ours | 9.6538 | 34.36 | 4.80× | 0.020 |
| wikitext | kvpress | 9.6538 | 46.64 | 3.53× | 0.023 |

## MIT Official Benchmark (non-PPL)

Measures decode throughput (tokens/s) and peak VRAM; not directly comparable to the decode-loop PPL table.

| File | Mode | Prefix | Gen | Decode tok/s | Decode (s) | Prefill (s) | Peak alloc (MB) | Peak reserved (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pg19_20k_benchmark.json | streaming | 20000 | 512 | 36.76 | 13.93 | 4.36 | 8086.0 | 8936.0 |
| pg19_20k_benchmark.json | recompute | 20000 | 512 | 4.17 | 122.72 | 0.00 | 7023.3 | 7222.0 |
