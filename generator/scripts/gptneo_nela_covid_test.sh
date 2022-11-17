python run_clm.py \
    --model_name_or_path EleutherAI/gpt-neo-2.7B \
    --validation_file "data/nela-covid-2020-test.json" \
    --do_eval \
    --output_dir "outputs/gptneo-covid-test" \
    --per_device_eval_batch_size 4 \
    --block_size 512 \
