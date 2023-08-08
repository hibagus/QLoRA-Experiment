#!/bin/bash
### Configurable
# Before run, please set environment variable: export DEVICES='DeviceID'
export CPREC='fp16'
export SPREC='nf4'
export METHOD='qlora'
export LOADBIT=4
export MODELHUB='huggyllama'
export MODELNAME='llama-65b'
export DATASET='oasst1'
export DOUBLEQUANT='True'
export PROFILE='False'


### Do not touch
DATE=$(date '+%Y%m%d-%H%M%S')
export LOG=../../log/$DATE-$DEVICE-$METHOD-$DATASET-$SPREC-$CPREC-$DOUBLEQUANT.log

exec   > >(tee -ia $LOG)
exec  2> >(tee -ia $LOG >& 2)
exec 19> $LOG

export BASH_XTRACEFD="19"
set -x

python ../../finetune.py \
    --method $METHOD \
    --profile $PROFILE \
    --model_name_or_path $MODELHUB/$MODELNAME \
    --output_dir ../../output/experiments/$DEVICE/$MODELNAME/$DATE-$METHOD-$DATASET-$SPREC-$CPREC-$DOUBLEQUANT \
    --run_name $DATE-$METHOD-$DATASET-$SPREC-$CPREC-$DOUBLEQUANT \
    --logging_steps 16 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 0 \
    --save_total_limit 0 \
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
    --mmlu_path ../../data/mmlu/ \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant $DOUBLEQUANT \
    --quant_type $SPREC \
    --$CPREC \
    --bits $LOADBIT \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --gradient_checkpointing \
    --dataset $DATASET \
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
    --seed 0



set +x