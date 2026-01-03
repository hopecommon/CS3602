$env:PYTHONUTF8=1
python examples/eval_long_ppl.py `
    --model_name_or_path EleutherAI/pythia-2.8b `
    --dataset_path "f:\2025Autumn\CS3602\homework\FinalWork\CS3602\data\pg19\long_context_20000.json" `
    --enable_start_recent_kv_cache `
    --start_size 32 `
    --recent_size 2016 `
    --num_eval_tokens 50000 `
    --output_dir results/pg19_local_test

Write-Host "Done! Check results in results/pg19_local_test"
