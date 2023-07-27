python ../finetune.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ../output/experiments/llama_7b_qlora_fp4_fp16 \
    --logging_steps 16 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 512 \
    --save_total_limit 32 \
    --evaluation_strategy steps \
    --eval_dataset_size 1024 \
    --max_eval_samples 1024 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_mmlu_eval \
    --mmlu_dataset mmlu-fs \
    --mmlu_path ../data/mmlu/ \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type fp4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing \
    --dataset oasst1 \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 2048 \
    --eval_steps 256 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 1234