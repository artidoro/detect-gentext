deepspeed run_clm.py \
    --model_name_or_path gpt2-xl \
    --train_file "data/nela-covid-2020-train.json" \
    --validation_file "data/nela-covid-2020-valid.json" \
    --do_train \
    --do_eval \
    --logging_first_step \
    --output_dir "outputs/gpt2-xl-covid" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --logging_steps 10000 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --block_size 512 \
    --overwrite_output_dir \
    --fp16 \
    --deepspeed deepspeed_config.json \

