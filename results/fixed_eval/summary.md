# Fixed Eval Summary

## Decode-Loop (comparable)

| Dataset | Method | PPL | Runtime (s) | Speedup vs baseline | First-token (s) |
|---|---|---:|---:|---:|---:|
| pg19_20k | baseline | 19.7529 | 1457.85 | 1.00× | 0.076 |
| pg19_20k | ours | 20.8190 | 274.63 | 5.31× | 0.019 |
| pg19_20k | kvpress | 20.8190 | 358.20 | 4.07× | 0.022 |
| wikitext | baseline | 9.3713 | 163.99 | 1.00× | 0.076 |
| wikitext | ours | 9.6538 | 31.69 | 5.17× | 0.019 |
| wikitext | kvpress | 9.6538 | 42.39 | 3.87× | 0.023 |
